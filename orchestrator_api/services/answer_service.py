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
import re
from shared.utils import log_event
from services.rag_singleton import rag_manager

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

# Sequence of characters which separe the real and proper answer to the user's question from the summary
SEPARATOR = "#####"
logger = structlog.get_logger()

@log_event("question_transcription", result_mapper=lambda x: {"transcription": x[0], "language": x[1]})
def transcribe_audio(audio_path):
    result = stt_model.transcribe(audio_path)
    language = result.get("language", "en")
    os.remove(audio_path)
    return result["text"], language


@log_event("text_answer_generation")
def stream_text_answer_by_sentence(question: str, pdf_name: str | None):
    structlog.contextvars.bind_contextvars(question=question)

    context, sources = "", []
    if pdf_name:
        try:
            # Recupera contesto (questo può richiedere tempo se c'è swap)
            context, sources = rag_manager.retrieve_context(pdf_name, question, k=5)
            structlog.contextvars.bind_contextvars(rag_sources=sources, rag_pdf=pdf_name)
        except Exception as e:
            logger.error(f"RAG Retrieval failed: {e}")
            structlog.contextvars.bind_contextvars(rag_pdf=pdf_name, rag_error="retrieve_failed")
            context, sources = "", []

    prompt = tutor_config["prompt"].format(question=question, context=context)
    
    # [DEBUG] Verifica che il prompt parta
    print(f"\n[DEBUG] Prompt sent to LLM. Context len: {len(context)} chars. Waiting for stream...\n")

    buffer = ""
    is_summary_mode = False
    found_title = False
    
    # Flag per gestire il DeepSeek Thinking
    inside_think_tag = False 
    chunks_received = 0

    try:
        for chunk in question_answerer_model.stream(prompt):
            text_chunk = chunk.content
            
            if not text_chunk:
                continue

            chunks_received += 1
            # --- GESTIONE DEEPSEEK THINKING (<think>...</think>) ---
            
            if inside_think_tag:
                if "</think>" in text_chunk:
                    parts = text_chunk.split("</think>", 1)
                    inside_think_tag = False
                    text_chunk = parts[1]
                else:
                    continue
            
            
            if "<think>" in text_chunk:
                parts = text_chunk.split("<think>", 1)
                pre_think = parts[0] 
                
                
                rest = parts[1]
                if "</think>" in rest:
                    after_think_parts = rest.split("</think>", 1)
                    text_chunk = pre_think + after_think_parts[1]
                else:
                    inside_think_tag = True
                    text_chunk = pre_think

            if not text_chunk:
                continue


            buffer += text_chunk

            # --- MODE 1: SPEECH ---
            if not is_summary_mode:
                if SEPARATOR in buffer:
                    speech_part, summary_part = buffer.split(SEPARATOR, 1)
                    
                    if speech_part.strip():
                        sentences = re.split(r'([.!?])', speech_part)
                        for i in range(0, len(sentences) - 1, 2):
                            sentence = sentences[i].strip() + sentences[i+1].strip()
                            if sentence:
                                print(f"[DEBUG] Yielding speech: {sentence[:30]}...")
                                yield (sentence, "speech")
                        
                        if len(sentences) % 2 == 1 and sentences[-1].strip():
                            yield (sentences[-1].strip(), "speech")

                    is_summary_mode = True
                    buffer = summary_part 
                else:
                    sentences = re.split(r'([.!?])', buffer)
                    for i in range(0, len(sentences) - 1, 2):
                        s = sentences[i].strip() + sentences[i+1].strip()
                        if s: 
                            print(f"[DEBUG] Yielding speech: {s[:30]}...")
                            yield (s, "speech")
                    buffer = sentences[-1] if len(sentences) % 2 == 1 else ""

            # --- MODE 2: SUMMARY ---
            if is_summary_mode:
                lines = buffer.split('\n')
                for i in range(len(lines) - 1):
                    line = lines[i].strip()
                    if not line: continue 

                    if not found_title:
                        yield (line, "title")
                        found_title = True
                    else:
                        clean_line = line.lstrip("*").strip()
                        yield (clean_line, "bullet")
                buffer = lines[-1]

        # --- FINAL FLUSH ---
        if buffer.strip() and not inside_think_tag:
            if is_summary_mode:
                line = buffer.strip()
                if not found_title:
                    yield (line, "title")
                else:
                    clean_line = line.lstrip("*").strip()
                    yield (clean_line, "bullet")
            else:
                yield (buffer.strip(), "speech")
        
        # --- FALLBACK PER BUFFER VUOTO ---
        if chunks_received == 0:
            print("[DEBUG] No chunks received from LLM!")
            yield ("Mi dispiace, c'è stato un problema tecnico e non riesco a formulare una risposta.", "speech")
        elif not buffer.strip() and chunks_received > 0:
             print("[DEBUG] Chunks received but buffer empty (maybe only thinking?)")
                
    except Exception as e:
        logger.error(f"LLM Stream Error: {e}")
        yield ("Mi dispiace, ho avuto un problema tecnico e non riesco a formulare una risposta.", "speech")


@log_event("audio_generation")
def synthesize_wav(sentence, language=xtts_config["default_language"]):
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