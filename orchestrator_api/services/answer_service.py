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

@log_event("question_transcription", result_mapper=lambda x: {"transcription": x[0], "language": x[1]})
def transcribe_audio(audio_path):
    # Transcribe audio using Whisper
    result = stt_model.transcribe(audio_path)
    # Detect language
    language = result.get("language", "en")

    # Cleanup temporary audio file
    os.remove(audio_path)

    return result["text"], language


@log_event("text_answer_generation")
def stream_text_answer_by_sentence(question: str, pdf_name: str | None):
    structlog.contextvars.bind_contextvars(question=question)

    context, sources = "", []
    if pdf_name:
        try:
            context, sources = rag_manager.retrieve_context(pdf_name, question, k=5)
            structlog.contextvars.bind_contextvars(rag_sources=sources, rag_pdf=pdf_name)
        except Exception:
            structlog.contextvars.bind_contextvars(rag_pdf=pdf_name, rag_error="retrieve_failed")
            context, sources = "", []

    prompt = tutor_config["prompt"].format(question=question, context=context)

    buffer = ""
    
    # State flags
    is_summary_mode = False
    found_title = False


    for chunk in question_answerer_model.stream(prompt):
        # ChatOllama restituisce il testo direttamente in chunk.content
        text_chunk = chunk.content
        
        if text_chunk:
            buffer += text_chunk

            # --- MODE 1: SPEECH ---
            if not is_summary_mode:
                if SEPARATOR in buffer:
                    # 1. Split speech from summary
                    speech_part, summary_part = buffer.split(SEPARATOR, 1)
                    
                    # Flush speech sentences
                    if speech_part.strip():
                        sentences = re.split(r'([.!?])', speech_part)
                        for i in range(0, len(sentences) - 1, 2):
                            sentence = sentences[i].strip() + sentences[i+1].strip()
                            if sentence:
                                yield (sentence, "speech")
                        
                        # Flush tail of speech
                        if len(sentences) % 2 == 1 and sentences[-1].strip():
                            yield (sentences[-1].strip(), "speech")

                    # Switch to Summary Mode
                    is_summary_mode = True
                    buffer = summary_part 
                else:
                    # Standard Speech buffering logic
                    sentences = re.split(r'([.!?])', buffer)
                    for i in range(0, len(sentences) - 1, 2):
                        s = sentences[i].strip() + sentences[i+1].strip()
                        if s: yield (s, "speech")
                    buffer = sentences[-1] if len(sentences) % 2 == 1 else ""

            # --- MODE 2: SUMMARY (Title + Bullets) ---
            if is_summary_mode:
                # Split by newlines to process line-by-line
                lines = buffer.split('\n')

                # Process all fully formed lines (leave the last one in buffer)
                for i in range(len(lines) - 1):
                    line = lines[i].strip()
                    if not line: 
                        continue # Skip empty lines

                    if not found_title:
                        # The first non-empty line after ##### is the Title
                        yield (line, "title")
                        found_title = True
                    else:
                        # Subsequent lines are bullets
                        # Remove the '*' if present
                        clean_line = line.lstrip("*").strip()
                        yield (clean_line, "bullet")
                
                # Keep the last incomplete line in buffer
                buffer = lines[-1]

    # --- FINAL FLUSH ---
    if buffer.strip():
        if is_summary_mode:
            # Handle the very last line
            line = buffer.strip()
            if not found_title:
                yield (line, "title")
            else:
                clean_line = line.lstrip("*").strip()
                yield (clean_line, "bullet")
        else:
            yield (buffer.strip(), "speech")

@log_event("audio_generation")
def synthesize_wav(sentence, language=xtts_config["default_language"]):
    structlog.contextvars.bind_contextvars(sentence=sentence, language=language)

    # remove trailing dot to avoid "dot" or "punto"
    if sentence.endswith("."):
        sentence = sentence[:-1]

    # add a safety space
    sentence = sentence + " "

    # Temporary WAV output
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        output_path = tmp.name

    # Full-sentence TTS
    tts_model.tts_to_file(
        text=sentence,
        speaker=xtts_config["speaker"],
        language=language,
        file_path=output_path
    )

    # Encode to base64
    with open(output_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()

    os.remove(output_path)
    return audio_base64