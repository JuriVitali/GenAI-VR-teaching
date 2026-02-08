from flask_socketio import SocketIO, emit
from flask import request
from eventlet import tpool
import requests
import tempfile
import os
import base64
import uuid
import json
from services.frontend_log_service import save_log_batch
from app import socketio
from services.answer_service import (
    detect_language,
    stream_text_answer_by_sentence,
    synthesize_wav,
    transcribe_audio,
    retrieve_context
)
import structlog
from dotenv import load_dotenv, find_dotenv
from services.object_extraction_service import extract_prompts, improve_prompt
from queue import Queue
from pathlib import Path
from shared.log_merger import merge_and_cleanup_session_logs
from services.memory_service import update_chat_history
import time

SID_TO_SESSION_ID = {}

def _get_session_id_from_ws(sid: str, data: dict | None = None) -> str | None:
    if isinstance(data, dict):
        s = data.get("session_id")
        if s:
            SID_TO_SESSION_ID[sid] = s
            return s
    return SID_TO_SESSION_ID.get(sid)
from services.rag_singleton import rag_manager

# Gets the logger instance
logger = structlog.get_logger()
active_pdf_by_sid = {}

load_dotenv(find_dotenv())
TEXT_TO_IMAGE_URL = os.getenv("ZIMAGE_TURBO_API_URL")
IMAGE_TO_3D_MODEL = os.getenv("IMAGE_TO_3D_MODEL")
if IMAGE_TO_3D_MODEL == "Trellis":
    IMAGE_TO_3D_URL = os.getenv("TRELLIS_API_URL")
else:
    IMAGE_TO_3D_URL = os.getenv("HUNYUAN_API_URL")

p = os.getenv("FRONTEND_LOG_FILE_PATH")
FRONTEND_LOG_FILE_PATH = Path(p).absolute()
FRONTEND_LOG_FILE_PATH.mkdir(parents=True, exist_ok=True)
a = os.getenv("LOG_FILE_PATH")
BACKEND_LOG_FILE_PATH = Path(a).absolute()

# --- JSON PRE-GENERATION SETUP ---
PRE_GEN_QUESTIONS_DIR = os.getenv("PRE_GEN_QUESTIONS_DIR")
STARTING_OBJECTS_DIR = os.getenv("STARTING_OBJECTS_DIR")

def _get_json_path(pdf_name, base_dir):
    if not base_dir:
        return None
    safe_name = Path(pdf_name).stem
    return Path(base_dir).absolute() / f"{safe_name}.json"

def _get_existing_questions(pdf_name):
    """Recupera le domande pre-generate."""
    file_path = _get_json_path(pdf_name, PRE_GEN_QUESTIONS_DIR)
    if file_path and file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [{"id": item.get("question_id"), "text": item.get("text_question")} for item in data]
        except Exception as e:
            logger.error(f"Failed to load questions for {pdf_name}", error=str(e))
    return []

def _get_starting_objects(pdf_name):
    """Recupera gli oggetti iniziali."""
    file_path = _get_json_path(pdf_name, STARTING_OBJECTS_DIR)
    if file_path and file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [{"object_id": item.get("object"), "question_id": item.get("question_id")} for item in data]
        except Exception as e:
            logger.error(f"Failed to load starting objects for {pdf_name}", error=str(e))
    return []

@socketio.on('connect')
def on_connect():
    sid = request.sid
    logger.info("[WS] CONNECT", sid=sid) #session_id will be bound on first message


@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    session_id = SID_TO_SESSION_ID.pop(sid, None)
    logger.info("[WS] DISCONNECT", sid=sid, session_id=session_id)


@socketio.on("ask")
def handle_ask(data):
    sid = request.sid
    session_id = _get_session_id_from_ws(sid, data)

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        connection="websocket",
        session_id=session_id,
        ws_event="ask",
    )
    worker_ctx = {
        "session_id": session_id,
        "request_id": data.get("request_id", -1)
    }
    context = structlog.contextvars.get_contextvars()
    session_id_ctx = context.get("session_id")
    headers = {}
    if session_id_ctx:
        headers["X-Session-ID"] = session_id_ctx

    try:
        sid = request.sid
        pdf_name = active_pdf_by_sid.get(sid)
        if not pdf_name:
            emit("error", {"message": "No PDF selected. Please select a PDF first."}, to=sid)
            return

        st, err = rag_manager.get_status(pdf_name)
        if st != "ready":
            emit("error", {"message": f"PDF not ready: {st}. Please wait."}, to=sid)
            return
        
        request_id = data.get("request_id", -1)
        question_format = data.get("question_format", "audio")
        audio_response = data.get("audio_response", True)
        object_gen = int(data.get("objects", 0))
        max_objects = data.get("max_objects", 1)
        
        if question_format == "audio":        
            audio_b64 = data.get("audio_question")
            audio_bytes = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            text_question, language = transcribe_audio(tmp_path)
        else:
            text_question = data.get("text_question")
            language = detect_language(text_question)
        
        emit("language_detected", {"language": language})

        audio_queue = Queue()

        def audio_worker(sid_local, ctx):
            structlog.contextvars.bind_contextvars(**ctx)
            while True:
                text_to_speak = audio_queue.get()
                if text_to_speak is None:
                    socketio.emit("audio_done", {"status": "completed", "request_id": request_id}, to=sid_local)
                    break

                try:
                    logger.info("audio_generation_started", sentence=text_to_speak)
                    start_t=time.perf_counter()
                    wav_b64 = tpool.execute(
                        synthesize_wav,
                        text_to_speak,
                        language=language,
                        context_data=ctx
                    )
                    duration = time.perf_counter() - start_t
                    logger.info("audio_generation_finished", duration=duration)
                    socketio.emit(
                        "audio_sentence",
                        {"base64": wav_b64, "request_id": ctx.get("request_id")},
                        to=sid_local
                    )

                except Exception as e:
                    logger.error(
                        "audio_synthesis_failed",
                        error=str(e),
                        session_id=ctx.get("session_id"),
                        ws_event="audio_worker",
                    )

                socketio.sleep(0)
                audio_queue.task_done()

        socketio.start_background_task(audio_worker, sid, worker_ctx)

        try:
            text_answer = []
            summary_title = ""
            summary_bullets = []

            is_negative_answer = False

            answer_context, sources = retrieve_context(text_question, pdf_name)

            # --- STREAM LOOP ---
            for text_content, message_type in stream_text_answer_by_sentence(text_question, language, pdf_name, sid, answer_context):

                # 1. MAIN SPEECH (Immediate TTS)
                if message_type == "speech":
                    emit("text_chunk", {"text": text_content, "request_id": request_id})
                    text_answer.append(text_content)
                    
                    if audio_response:
                        audio_queue.put(text_content)

                # 2. SUMMARY CONTENT (Collect)
                elif message_type == "title":
                    summary_title = text_content
                elif message_type == "bullet":
                    summary_bullets.append(text_content)

                # Keep connection alive
                socketio.sleep(0)

                # --- NEGATIVE ANSWER CHECK ---
                # Check only if we haven't flagged it yet
                if not is_negative_answer:
                    current_full_text = " ".join(text_answer).lower()
                    negative_triggers = [
                        "mi dispiace", "non ho trovato", "non è possibile trovare", 
                        "non dispongo di informazioni", "i cannot find the answer",
                        "non c'è risposta", "non sono in grado di", "non so la risposta",
                        "non posso rispondere", "non ho informazioni", "non ho dati", 
                        "non riesco a trovare"
                    ]
                    
                    if any(trigger in current_full_text for trigger in negative_triggers):
                        logger.info("[FLAG] Negative answer detected. Will skip assets at the end.")
                        is_negative_answer = True
                    
            # Final Logging
            full_text_answer = " ".join(text_answer)
            logger.info("tutor_answer", full_text_answer=full_text_answer)
            emit("text_done", {"status": "completed", "request_id": request_id})

            # --- POST-STREAM PROCESSING ---
            if is_negative_answer:
                logger.info("[SKIP] Asset generation skipped: Negative answer detected.")
                # Avvisiamo il frontend che non arriveranno oggetti
                socketio.emit("objects_done", {"status": "skipped", "request_id": request_id}, to=sid)
            else:
                # Send Summary to UI
                if summary_title or summary_bullets:
                    emit("summary", {
                        "title": summary_title, 
                        "body": summary_bullets,
                        "request_id": request_id
                    })

                # img and obj extraction
                obj_extracted, summary_img_extracted = extract_prompts(text_question, full_text_answer, answer_context, language)
                # img prompt improvement
                summary_img_prompt = improve_prompt(summary_img_extracted.prompt)
                    
                try:
                    txt2img_response = requests.post(
                        TEXT_TO_IMAGE_URL,
                        json={"prompt": summary_img_prompt, "summary": "true"},
                        headers=headers,
                        timeout=20
                    )
                    if txt2img_response.status_code == 200:
                        emit("summary_image", {
                            "image_id": txt2img_response.json().get("image_id"),
                            "caption": summary_img_extracted.caption,
                            "request_id": request_id
                        })
                    else:
                        logger.error("txt2img_error", status=txt2img_response.status_code)

                except requests.exceptions.RequestException as e:
                    logger.error("txt2img_connection_failed", error=str(e))

                # 3. Trigger 3D Object Generation (Step 6)
                if (object_gen):
                    socketio.start_background_task(
                        run_object_generation, sid, obj_extracted.prompt, obj_extracted.presentation_speech, language, headers, worker_ctx
                    )

        finally:
            # Clean up: Tell the worker to stop after the queue is empty
            audio_queue.put(None) 

    except Exception as e:
        logger.exception("error", error=str(e))
        emit("error", {"message": str(e)})


@socketio.on("default_ask")
def handle_default_ask(data):
    sid = request.sid
    
    # 1. Retrieve inputs
    question_id = data.get("question_id")
    audio_response = data.get("audio_response", True)
    objects_enabled = data.get("objects", False)
    request_id = data.get("request_id", -1)
    session_id = data.get("session_id")
    is_object_explanation = data.get("is_object", False)

    # 2. Validation
    pdf_name = active_pdf_by_sid.get(sid)
    if not pdf_name:
        emit("error", {"message": "No PDF selected. Please select a PDF first."}, to=sid)
        return

    if question_id is None:
        emit("error", {"message": "Missing question_id."}, to=sid)
        return
    
    if session_id is None:
        emit("error", {"message": "Missing session_id."}, to=sid)
        return

    # 3. Retrieve Data from JSON
    if is_object_explanation:
        base_dir = STARTING_OBJECTS_DIR
    else:
        base_dir = PRE_GEN_QUESTIONS_DIR

    file_path = _get_json_path(pdf_name, base_dir)
    target_data = None
    
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                saved_list = json.load(f)
                # Find the entry with the matching ID
                for entry in saved_list:
                    if entry.get("question_id") == question_id:
                        target_data = entry
                        break
        except Exception as e:
            logger.error(f"Failed to load json for default_ask: {pdf_name}", error=str(e))
    
    if not target_data:
        emit("error", {"message": f"Question ID {question_id} not found for this PDF."}, to=sid)
        return

    # 4. Emit Events in Sequence
    
    # A. Language Detected (fixed to 'it' per instructions)
    emit("language_detected", {"language": "it"})
    
    # B. Text Chunks and Audio Sentences (Interleaved)
    text_chunks = target_data.get("text_chunks", [])
    audio_sentences = target_data.get("audio_sentences", [])
    
    # We iterate based on text chunks. 
    # Safety: Use length of text_chunks. If audio is missing for a chunk, we skip audio emission.
    count = len(text_chunks)
    
    for i in range(count):
        socketio.sleep(0.3)
        # Emit Text Chunk
        txt = text_chunks[i]
        emit("text_chunk", {"text": txt, "request_id": request_id})
        
        # Emit Audio Sentence (if enabled and exists)
        if audio_response and i < len(audio_sentences):
            b64_audio = audio_sentences[i]
            if b64_audio: # Ensure it's not empty
                emit("audio_sentence", {"base64": b64_audio, "request_id": request_id})
        
        
    socketio.sleep(0.2)

    # C. Text Done
    emit("text_done", {"status": "completed", "request_id": request_id})

    # D. Audio Done (if enabled)
    if audio_response:
        emit("audio_done", {"status": "completed", "request_id": request_id})

    # E. Summary
    summ = target_data.get("summary", {})
    if summ.get("title") or summ.get("body"):
        emit("summary", {"title": summ.get("title", ""), "body": summ.get("body", []), "request_id": request_id})
    socketio.sleep(0.3)
    # F. Summary Image
    summ_img = target_data.get("summary_image", {})
    if summ_img.get("image_id"):
        emit("summary_image", {
            "image_id": summ_img.get("image_id"),
            "caption": summ_img.get("caption", ""),
            "request_id": request_id
        })

    socketio.sleep(10)
    # G. Objects (if enabled)
    if objects_enabled:
        obj_data = target_data.get("object", {})
        # Only emit 'object' event if there is actual object data
        if obj_data.get("object"):
             emit("object", {
                "object": obj_data.get("object"),
                "text": obj_data.get("text", ""),
                "speech": obj_data.get("speech", ""),
                "request_id": request_id
            })
        
        emit("objects_done", {"status": "completed", "request_id": request_id})

    # H. Update chat history
    user_question = target_data.get("text_question", "")
    avatar_response = target_data.get("text_response", "")
    update_chat_history(session_id, user_question, avatar_response)


@socketio.on("log")
def handle_log_batch(data):
    sid = request.sid

    if not isinstance(data, dict) or data.get("type") != "log_batch":
        emit("log_batch_ack", {"type": "log_batch_ack", "session_id": None, "acked_to_seq": 0, "ok": False}, to=sid)
        return

    session_id = data.get("session_id")
    SID_TO_SESSION_ID[sid] = session_id

    if not session_id:
        emit("log_batch_ack", {"type": "log_batch_ack", "session_id": None, "acked_to_seq": 0, "ok": False}, to=sid)
        return

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        connection="websocket",
        session_id=session_id,
        user_id=data.get("user_id"),
        condition=data.get("condition"),
        ws_event="log_batch",
    )

    try:
        
        acked_to_seq, rows_written = save_log_batch(FRONTEND_LOG_FILE_PATH, data)
        logger.info("log_batch_saved", rows_written=rows_written, acked_to_seq=acked_to_seq)

        # CONTROLLO SE C'È L'EVENTO "session_end" NEL BATCH APPENA SALVATO
        events = data.get("events", [])
        session_ended = any(evt.get("event_name") == "session_end" for evt in events)

        if session_ended:
            logger.info("session_end_detected_in_logs", session_id=session_id)
            
            # Definizione del task di merge
            def run_merge_task(sess_id):
                # Aspettiamo qualche secondo per assicurarci che anche questo log ("session_end_detected...")
                # sia stato scritto su app.log dal backend.
                time.sleep(2.5) 
                
                fe_base_path = str(FRONTEND_LOG_FILE_PATH)
                be_path = str(BACKEND_LOG_FILE_PATH)
                out_path = FRONTEND_LOG_FILE_PATH.parent / "sessions" / f"full_session_{sess_id}.jsonl"

                merge_and_cleanup_session_logs(sess_id, fe_base_path, be_path, str(out_path))

            # Avvio del task in background
            socketio.start_background_task(run_merge_task, session_id)

        # ACK AL FRONTEND
        emit("log_batch_ack", {
            "type": "log_batch_ack",
            "session_id": session_id,
            "acked_to_seq": acked_to_seq,
            "ok": True
        }, to=sid)

    except Exception as e:
        logger.error("log_batch_failed", error=str(e))
        emit("log_batch_ack", {
            "type": "log_batch_ack",
            "session_id": session_id,
            "acked_to_seq": 0,
            "ok": False
        }, to=sid)


def run_object_generation(sid, obj_blueprint_text, obj_presentation_text, language, headers, ctx):
    structlog.contextvars.bind_contextvars(**ctx)
    try:
        socketio.sleep(0)
        obj_presentation_audio = synthesize_wav(obj_presentation_text, language, context_data=ctx)

        try:
            txt2img_response = requests.post(
                TEXT_TO_IMAGE_URL,
                json={"prompt": obj_blueprint_text, "summary": False},
                headers=headers,
                timeout=20
            )
        except requests.exceptions.RequestException as e:
            logger.error("txt2img_connection_failed", error=str(e))
            socketio.emit("objects_done", {"status": "completed", "request_id": ctx.get("request_id")}, to=sid)
            return

        if txt2img_response.status_code == 200:
            img_id = txt2img_response.json().get("image_id")

            try:
                img2obj_response = requests.post(
                    IMAGE_TO_3D_URL,
                    json={"img_id": img_id},
                    headers=headers,
                    timeout=120
                )
            except requests.exceptions.RequestException as e:
                logger.error("img2obj_connection_failed", error=str(e))
                socketio.emit("objects_done", {"status": "completed", "request_id": ctx.get("request_id")}, to=sid)
                return

            if img2obj_response.status_code == 200:
                obj_filename = img2obj_response.json().get("object_id")
                socketio.emit("object", {
                    "object": obj_filename,
                    "text": obj_presentation_text,
                    "speech": obj_presentation_audio,
                    "request_id": ctx.get("request_id")
                }, to=sid)
            else:
                logger.error("img2obj_error_status", status=img2obj_response.status_code)
        else:
            logger.error("txt2img_error_status", status=txt2img_response.status_code)

    except Exception as e:
        logger.error("generation_pipeline_crashed", error=str(e))

    socketio.emit("objects_done", {"status": "completed", "request_id": ctx.get("request_id")}, to=sid)


@socketio.on("pdf_selected")
def handle_pdf_selected(data):
    sid = request.sid
    pdf_name = (data or {}).get("pdf_name", "").strip().lower()

    if not pdf_name:
        emit("pdf_selected_ack", {"status": "error", "message": "missing_pdf_name"}, to=sid)
        return

    active_pdf_by_sid[sid] = pdf_name

    st, err = rag_manager.get_status(pdf_name)
    
    # Recupero dei dati pre-generati
    existing_questions = _get_existing_questions(pdf_name)
    starting_objects = _get_starting_objects(pdf_name)

    if st == "ready":
        emit("pdf_selected_ack", {
            "pdf_name": pdf_name, 
            "status": "ok",
            "questions": existing_questions,
            "objects": starting_objects
        }, to=sid)
        return

    if st == "building":
        # opzionale
        emit("pdf_selected_ack", {
            "pdf_name": pdf_name, 
            "status": "building",
            "questions": existing_questions,
            "objects": starting_objects
        }, to=sid)
        return

    def build_and_ack():
        ok, msg = tpool.execute(rag_manager.ensure_ready, pdf_name)

        # Re-fetch both just in case something changed during build or simply to allow consistency
        questions_after_build = _get_existing_questions(pdf_name)
        objects_after_build = _get_starting_objects(pdf_name)

        if ok:
            socketio.emit("pdf_selected_ack", {
                "pdf_name": pdf_name, 
                "status": "ok",
                "questions": questions_after_build,
                "objects": objects_after_build
            }, to=sid)
        else:
            st2, err2 = rag_manager.get_status(pdf_name)
            socketio.emit(
                "pdf_selected_ack",
                {"pdf_name": pdf_name, "status": "error", "message": err2 or msg},
                to=sid
            )

    socketio.start_background_task(build_and_ack)
