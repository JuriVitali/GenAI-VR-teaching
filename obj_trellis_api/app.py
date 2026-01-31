import sys
import os
import uuid

from requests import session
import structlog
import logging
from flask import Flask, request
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

root_dir = os.getenv("ROOT_DIR")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from shared.logger import setup_logging

# Initialize the logger
logger = setup_logging("trellis-3d-api")

from controllers.controller import gen_bp

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Middleware: Correlation ID handler
    @app.before_request
    def bind_request_details():
        session_id = request.headers.get("X-Session-ID")
        # Bind session_id so all logs in this request context have it
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(session_id=session_id)

    app.register_blueprint(gen_bp, url_prefix="/api")
    return app

if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("TRELLIS_PORT", 5020))
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.INFO)
    logger.info("server_starting", port=port, host="0.0.0.0")
    app.run(host="0.0.0.0", port=port, debug=False)