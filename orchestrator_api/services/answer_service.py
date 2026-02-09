from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import structlog
from dotenv import load_dotenv, find_dotenv
import json
import yaml
import base64
import whisper
from config.model_config_loader import ModelConfig
from services.llm_service import get_llm_model
from TTS.api import TTS
import torch
import os
import tempfile
import time
import re
from shared.utils import log_event
from services.rag_singleton import rag_manager
from services.memory_service import get_chat_history, update_chat_history
from langchain_core.messages import HumanMessage, SystemMessage
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# Load environment variables
load_dotenv(find_dotenv())

# Load model configuration params
model_config = ModelConfig()
xtts_config = model_config.get("xtts")
tutor_config = model_config.get("tutor")
whisper_config = model_config.get("whisper")

# Get the models
stt_model = whisper.load_model("turbo")
question_answerer_model = get_llm_model(tutor_config["model"], tutor_config["temperature"])
tts_model = TTS(xtts_config["model"]).to("cuda")

logger = structlog.get_logger()

@log_event("question_transcription", result_mapper=lambda x: {"transcription": x[0], "language": x[1]})
def transcribe_audio(audio_path):
    result = stt_model.transcribe(audio_path)
    language = result.get("language", "en")
    os.remove(audio_path)
    return result["text"], language

def detect_language(text_input):
    return detect(text_input)

def retrieve_context(question: str, pdf_name: str, session_id: str):
    try:
        if session_id:
            structlog.contextvars.bind_contextvars(session_id=session_id)
        # Retrieve context
        context, sources = rag_manager.retrieve_context(pdf_name, question, k=5)
        structlog.contextvars.bind_contextvars(rag_sources=sources, rag_pdf=pdf_name)
    except Exception as e:
        logger.error(f"RAG Retrieval failed: {e}")
        structlog.contextvars.bind_contextvars(rag_pdf=pdf_name, rag_error="retrieve_failed")
        context, sources = "", []
    finally:
        return context, sources

def stream_text_answer_by_sentence(question: str, language: str, pdf_name: str , session_id: str, context):
    
    logger.info("text_answer_generation_started", question=question)
    start_time = time.perf_counter()
    structlog.contextvars.bind_contextvars(question=question)
    
    # Construct the Message List
    system_text = tutor_config["prompt"].format(context=context, language=language)
    
    messages = [
        SystemMessage(content=system_text),
    ]
    
    # Inject History
    messages.extend(get_chat_history(session_id))
    
    # Add Current Question
    messages.append(HumanMessage(content=question))

    # --- DEFINITION OF MARKERS & SECTIONS ---
    # We define the order in which we expect sections to appear.
    # The 'type' is the label yielded to the frontend/system.
    SECTION_MAP = {
        "**AUDIO SCRIPT**": "speech",
        "**SUMMARY**": "summary"
    }
    
    # Create an ordered list of markers for detection
    MARKERS = list(SECTION_MAP.keys())

    buffer = ""
    full_answer_accumulator = ""
    
    # State tracking
    current_marker = None  # Start undefined
    found_summary_title = False
    inside_think_tag = False 
    chunks_received = 0
    raw_answer_full = ""

    # Helper: Process a block of text based on the current active section
    def process_section_content(content, section_marker, is_final_flush=False):
        nonlocal found_summary_title
        if not content.strip():
            return [], "" 

        section_type = SECTION_MAP.get(section_marker)
        yielded_items = []

        # 1. AUDIO SCRIPT or 3D PRESENTATION (Speech Logic)
        if section_type == "speech":
            sentences = re.split(r'([.!?])', content)
            limit = len(sentences) if is_final_flush else len(sentences) - 1
            
            for i in range(0, limit, 2):
                if i+1 < len(sentences):
                    s = sentences[i].strip() + sentences[i+1].strip()
                    if s: yielded_items.append((s, section_type))
                elif is_final_flush and sentences[i].strip():
                    yielded_items.append((sentences[i].strip(), section_type))
            
            leftover = ""
            if not is_final_flush and len(sentences) % 2 == 1:
                leftover = sentences[-1]
            return yielded_items, leftover

        # 2. SUMMARY (Title/Bullet Logic)
        elif section_type == "summary":
            lines = content.split('\n')
            limit = len(lines) if is_final_flush else len(lines) - 1
            
            for i in range(limit):
                line = lines[i].strip()
                if not line: continue
                
                if not found_summary_title:
                    clean_title = line.replace('*', '').strip("# ").strip()
                    yielded_items.append((clean_title, "title"))
                    found_summary_title = True
                else:
                    clean_bullet = line.replace('*', '').strip()
                    if clean_bullet:
                        yielded_items.append((clean_bullet, "bullet"))
            
            leftover = ""
            if not is_final_flush:
                leftover = lines[-1]
            return yielded_items, leftover

    # --- STREAMING LOOP ---
    try:
        for chunk in question_answerer_model.stream(messages):
            text_chunk = chunk.content
            chunks_received += 1
            raw_answer_full += text_chunk
            
            # 1. Handle Thinking Tags (DeepSeek R1 / Generic)
            if "<think>" in text_chunk:
                inside_think_tag = True
                text_chunk = text_chunk.replace("<think>", "")
            
            if "</think>" in text_chunk:
                inside_think_tag = False
                parts = text_chunk.split("</think>")
                text_chunk = parts[1] if len(parts) > 1 else ""

            if inside_think_tag or not text_chunk:
                continue

            buffer += text_chunk
            full_answer_accumulator += text_chunk

            # 2. Detect Section Transitions
            # Check if ANY marker appears in the buffer
            found_marker = None
            found_marker_index = -1

            for marker in MARKERS:
                if marker in buffer:
                    # We need to find the earliest marker in the buffer to split correctly
                    idx = buffer.find(marker)
                    if found_marker is None or idx < found_marker_index:
                        found_marker = marker
                        found_marker_index = idx

            # 3. Handle Transition
            if found_marker:
                # content_before belongs to the OLD section
                content_before = buffer[:found_marker_index]
                
                # Flush the OLD section
                if current_marker:
                    items, _ = process_section_content(content_before, current_marker, is_final_flush=True)
                    for item in items: yield item
                
                # Start the NEW section
                current_marker = found_marker
                
                # Remove the marker from buffer and continue processing the rest
                # We add a generic strip to remove newlines immediately after markers
                buffer = buffer[found_marker_index + len(found_marker):].lstrip()

            # 4. Incremental Processing (within current section)
            if current_marker:
                items, new_buffer = process_section_content(buffer, current_marker, is_final_flush=False)
                for item in items: yield item
                buffer = new_buffer
    
        # --- FINAL FLUSH ---
        # Process whatever is left in the buffer for the last active section
        if current_marker and buffer.strip():
            items, _ = process_section_content(buffer, current_marker, is_final_flush=True)
            for item in items: yield item
        
        # --- LOGGING & HISTORY ---
        print("\n" + "="*50)
        print("FULL LLM RESPONSE (RAW):")
        print(raw_answer_full)
        print("="*50 + "\n")

        update_chat_history(session_id, question, full_answer_accumulator)
        duration = time.perf_counter() - start_time
        logger.info("text_answer_generation_finished", duration=duration, chunks_received=chunks_received)
    
    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.error("text_answer_generation_failed", error=str(e), duration=duration)
        raise e

#@log_event("audio_generation")
def synthesize_wav(sentence, language=xtts_config["default_language"], context_data=None):
    if context_data:
        structlog.contextvars.bind_contextvars(**context_data)
    structlog.contextvars.bind_contextvars(sentence=sentence, language=language)
    if sentence.endswith("."):
        sentence = sentence[:-1]
    sentence = sentence + " "
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        output_path = tmp.name
    tts_model.tts_to_file(
        text=sentence,
        speaker=xtts_config["speaker"],
        language=language,
        file_path=output_path
    )
    with open(output_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()
    os.remove(output_path)
    return audio_base64