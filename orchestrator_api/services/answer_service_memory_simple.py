from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import time
import structlog
from dotenv import load_dotenv, find_dotenv
import json
import yaml
import base64
import whisper
from config.model_config_loader import ModelConfig
from TTS.api import TTS
import torch
import os
import tempfile
import re
from services.llm_service import get_llm_model, get_chat_agent
from langchain_core.messages import HumanMessage, SystemMessage

# Gets the logger instance
logger = structlog.get_logger()

# Load environment variables
load_dotenv(find_dotenv())

# Load model configuration params
model_config = ModelConfig()
xtts_config = model_config.get("xtts")
tutor_config = model_config.get("tutor")
whisper_config = model_config.get("whisper")

# Get the models
stt_model = whisper.load_model("turbo")
tts_model = TTS(xtts_config["model"]).to("cuda")

# INITIALIZE THE AGENT INSTEAD OF RAW LLM
# We pass the system prompt here to ensure it's embedded in the agent's logic if needed,
# or we can pass it dynamically.
agent_app = get_chat_agent(
    model_name=tutor_config["model"], 
    temperature=tutor_config["temperature"],
    system_prompt=tutor_config["prompt"] 
)

# Sequence of characters which separe the real and proper answer to the user's question from the summary
SEPARATOR = "#####"

def get_text_answer(question: str, session_id = "1") -> str:

    logger.info(f"Starting text answer generation for session {session_id}")
    # This matches the simplicity you wanted:
    config = {"configurable": {"thread_id": session_id}}
    
    # We pass the message in the "messages" key
    result = chat_agent.invoke(
        {"messages": [("user", question)]}, 
        config
    )

    logger.info(f"Text answer generated successfully", duration=(time.time() - start_text_answer_generation))
    
    # The last message in the state is the AI's response
    return result["messages"][-1].content


def transcribe_audio(audio_path):
    # Transcribe audio using Whisper and log processing time
    logger.info(f"Transcribing audio")
    start_audio_transcription = time.time()
    result = stt_model.transcribe(audio_path)
    
    # Log transcription and detected language
    language = result.get("language", "en")
    logger.info(f"Question transcribed succesfully.", transcription=result['text'], duration=(time.time() - start_audio_transcription), language=language)

    # Cleanup temporary audio file
    os.remove(audio_path)

    return result["text"], language


def stream_text_answer_by_sentence(question: str):
    prompt = tutor_config["prompt"].format(question=question)
    buffer = ""
    
    # State flags
    is_summary_mode = False
    found_title = False

    for chunk in question_answerer_model.stream(prompt):
        for block in chunk.content_blocks:
            if block["type"] == "reasoning":
                continue

            if block["type"] == "text":
                text = block.get("text", "")
                buffer += text

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


def synthesize_wav(sentence, language=xtts_config["default_language"]):

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