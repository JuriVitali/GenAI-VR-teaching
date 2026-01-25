import sys
import os
import uuid
import structlog
import logging
from flask import Flask, request
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

ROOT_DIR = os.getenv("ROOT_DIR")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared.logger import setup_logging

# Initialize the logger
logger = setup_logging("trellis-3d-api")

from controllers.controller import gen_bp

# Load environment variables
load_dotenv(find_dotenv())

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Middleware: Correlation ID handler
    @app.before_request
    def bind_request_details():
        # Get RID from headers (sent by Orchestrator) or generate new one
        rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        # Bind rid so all logs in this request context have it
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(rid=rid)

    app.register_blueprint(gen_bp, url_prefix="/api")
    return app

if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("TRELLIS_PORT", 5020))
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.INFO)
    logger.info("server_starting", port=port, host="0.0.0.0")
    app.run(host="0.0.0.0", port=port, debug=False)