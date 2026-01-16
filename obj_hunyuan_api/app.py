import sys
import os
import uuid
import structlog
from flask import Flask, request
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv

root_dir = "/home/vrai/Copy"
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from shared.logger import setup_logging

# Initialize the logger
logger = setup_logging("hunyuan-3d-api")

# Get the directory where app.py is located
project_root = "/home/vrai/Copy/obj_hunyuan_api"
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
        session_id = request.headers.get("X-Session-ID")
        # Bind rid so all logs in this request context have it
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(rid=rid, session_id=session_id)

    app.register_blueprint(gen_bp, url_prefix="/api")
    return app

if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("HUNYUAN_PORT", 5030))
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logger.info("server_starting", port=port, host="0.0.0.0")
    app.run(host="0.0.0.0", port=port, debug=False)