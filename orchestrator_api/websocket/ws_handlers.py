# ws_handlers.py
from flask_socketio import SocketIO, emit
from flask import request
from eventlet import tpool
import requests
import tempfile
import os
import base64
import uuid
from services.frontend_log_service import save_log_batch
from app import socketio
from services.answer_service import (
    stream_text_answer_by_sentence,
    synthesize_wav,
    transcribe_audio
)
import structlog
from dotenv import load_dotenv, find_dotenv
from services.object_extraction_service import extract_prompts
from queue import Queue
from pathlib import Path

SID_TO_SESSION_ID = {}

def _get_session_id_from_ws(sid: str, data: dict | None = None) -> str | None:
    if isinstance(data, dict):
        s = data.get("session_id")
        if s:
            SID_TO_SESSION_ID[sid] = s
            return s
    return SID_TO_SESSION_ID.get(sid)

logger = structlog.get_logger()

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
    rid = str(uuid.uuid4())
    sid = request.sid
    session_id = _get_session_id_from_ws(sid, data)

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        connection="websocket",
        session_id=session_id,
        rid=rid,
        ws_event="ask",
    )
    worker_ctx = {
        "session_id": session_id,
        "rid": rid,
    }


    try:
        audio_response = data.get("audio_response", True)
        object_gen = int(data.get("objects", 0))
        audio_b64 = data.get("audio_question")
        audio_bytes = base64.b64decode(audio_b64)
        max_objects = data.get("max_objects", 1)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        transcription, language = transcribe_audio(tmp_path)

        emit("language_detected", {"language": language})

        audio_queue = Queue()

        def audio_worker(sid_local, ctx):
            while True:
                text_to_speak = audio_queue.get()
                if text_to_speak is None:
                    socketio.emit("audio_done", {"status": "completed"}, to=sid_local)
                    break

                try:
                    wav_b64 = tpool.execute(
                        synthesize_wav,
                        text_to_speak,
                        language=language
                    )

                    socketio.emit(
                        "audio_sentence",
                        {"base64": wav_b64},
                        to=sid_local
                    )

                except Exception as e:
                    logger.error(
                        "audio_synthesis_failed",
                        error=str(e),
                        session_id=ctx.get("session_id"),
                        rid=ctx.get("rid"),
                        ws_event="audio_worker",
                    )

                socketio.sleep(0)
                audio_queue.task_done()


        socketio.start_background_task(audio_worker, sid, worker_ctx)

        try:
            full_answer = []
            summary_title = ""
            summary_bullets = []

            for text_content, message_type in stream_text_answer_by_sentence(transcription):
                if message_type == "speech":
                    emit("text_chunk", {"text": text_content})
                    full_answer.append(text_content)

                    if audio_response:
                        audio_queue.put(text_content)

                elif message_type == "title":
                    summary_title = text_content

                elif message_type == "bullet":
                    summary_bullets.append(text_content)

                socketio.sleep(0)

            if summary_title or summary_bullets:
                emit("summary", {"title": summary_title, "body": summary_bullets})

            answer_text = " ".join(full_answer)
            logger.info("tutor_response", answer_text=answer_text)
            emit("text_done", {"status": "completed"})

        finally:
            audio_queue.put(None)

        obj_extracted, summary_img_extracted = extract_prompts(transcription, answer_text)

        context = structlog.contextvars.get_contextvars()
        rid_ctx = context.get("rid")
        session_id_ctx = context.get("session_id")
        headers = {}
        if rid_ctx:
            headers["X-Request-ID"] = rid_ctx
        if session_id_ctx:
            headers["X-Session-ID"] = session_id_ctx
        try:
            txt2img_response = requests.post(
                TEXT_TO_IMAGE_URL,
                json={"prompt": summary_img_extracted.prompt, "summary": "true"},
                headers=headers,
                timeout=20
            )
            if txt2img_response.status_code == 200:
                emit("summary_image", {
                    "image_id": txt2img_response.json().get("image_id"),
                    "caption": summary_img_extracted.caption
                })
            else:
                logger.error("txt2img_error", status=txt2img_response.status_code)

        except requests.exceptions.RequestException as e:
            logger.error("txt2img_connection_failed", error=str(e))

        if object_gen:
            socketio.start_background_task(
                run_object_generation, sid, obj_extracted, language, headers
            )

    except Exception as e:
        emit("error", {"message": str(e)})


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

    rid = str(uuid.uuid4())

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        connection="websocket",
        session_id=session_id,
        rid=rid,
        user_id=data.get("user_id"),
        condition=data.get("condition"),
        ws_event="log_batch",
    )


    try:
        acked_to_seq, rows_written = save_log_batch(FRONTEND_LOG_FILE_PATH, data)
        logger.info("log_batch_saved", rows_written=rows_written, acked_to_seq=acked_to_seq)

        emit("log_batch_ack", {
            "type": "log_batch_ack",
            "session_id": session_id,
            "acked_to_seq": acked_to_seq,
            "rid": rid,
            "ok": True
        }, to=sid)

    except Exception as e:
        logger.error("log_batch_failed", error=str(e))
        emit("log_batch_ack", {
            "type": "log_batch_ack",
            "session_id": session_id,
            "acked_to_seq": 0,
            "rid": rid,
            "ok": False
        }, to=sid)


def run_object_generation(sid, obj_extracted, language, headers):
    try:
        socketio.sleep(0)
        obj_presentation_audio = synthesize_wav(obj_extracted.speech, language)

        try:
            txt2img_response = requests.post(
                TEXT_TO_IMAGE_URL,
                json={"prompt": obj_extracted.prompt, "summary": False},
                headers=headers,
                timeout=20
            )
        except requests.exceptions.RequestException as e:
            logger.error("txt2img_connection_failed", error=str(e))
            socketio.emit("objects_done", {"status": "completed"}, to=sid)
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
                socketio.emit("objects_done", {"status": "completed"}, to=sid)
                return

            if img2obj_response.status_code == 200:
                obj_filename = img2obj_response.json().get("object_id")
                socketio.emit("object", {
                    "object": obj_filename,
                    "text": obj_extracted.speech,
                    "speech": obj_presentation_audio
                }, to=sid)
            else:
                logger.error("img2obj_error_status", status=img2obj_response.status_code)
        else:
            logger.error("txt2img_error_status", status=txt2img_response.status_code)

    except Exception as e:
        logger.error("generation_pipeline_crashed", error=str(e))

    socketio.emit("objects_done", {"status": "completed"}, to=sid)
