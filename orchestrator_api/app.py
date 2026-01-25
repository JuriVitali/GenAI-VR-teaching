import eventlet
eventlet.monkey_patch()

import sys, os
import uuid
import structlog
import logging
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request

load_dotenv(find_dotenv())

ROOT_DIR = os.getenv("ROOT_DIR")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared.logger import setup_logging

logger = setup_logging("orchestrator_api")

from orchestrator_api.controllers.controller import chat_bp
from orchestrator_api.websocket.socketio_instance import socketio
import orchestrator_api.websocket.ws_handlers

load_dotenv(find_dotenv())
PORT = os.getenv("ORCHESTRATOR_PORT")

def create_app():
    app = Flask(__name__)


    @app.before_request
    def start_trace():
        rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(rid=rid)

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