import eventlet
eventlet.monkey_patch()

import sys, os
import uuid
import structlog
import logging
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request

root_dir = "/home/vrai/Copy"
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from shared.logger import setup_logging

logger = setup_logging("orchestrator_api")

from controllers.controller import chat_bp
from websocket.socketio_instance import socketio
import websocket.ws_handlers

load_dotenv(find_dotenv())
PORT = os.getenv("ORCHESTRATOR_PORT")

def create_app():
    app = Flask(__name__)

    @app.before_request
    def start_trace():
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    session_id = request.headers.get("X-Session-ID")

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        rid=rid,
        session_id=session_id,
    )

    app.register_blueprint(chat_bp, url_prefix="/api")
    socketio.init_app(app)
    return app

if __name__ == "__main__":
    app = create_app()
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.INFO)
    logger.info("orchestrator_starting", host="0.0.0.0", port=PORT)
    socketio.run(app, host="0.0.0.0", port=PORT, log_output=False)