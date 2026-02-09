from flask import Blueprint, request, jsonify, send_file
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import structlog
import os
from shared.utils import log_event
from flask import make_response

load_dotenv(find_dotenv())
GEN_OBJECTS_DIR = os.getenv("OBJECTS_DIR")
GEN_IMAGES_DIR = os.getenv("SUMMARY_IMAGES_DIR")
PRE_GEN_OBJECTS_DIR = os.getenv("PRE_GEN_OBJECTS_DIR")
PRE_GEN_IMAGES_DIR = os.getenv("PRE_GEN_IMAGES_DIR")

# Gets the logger instance
logger = structlog.get_logger()

chat_bp = Blueprint("chat_bp", __name__)

@chat_bp.route("/objects", methods=["GET"])
@log_event("object_download", result_mapper=lambda r: {"status_code": r.status_code})
def get_object():
    """
    Returns a generated .glb object file.
    Expects a 'filename' query parameter without extension.
    """
    is_pregen = request.args.get("pre_generated", "").lower() == "true"    
    obj_id = request.args.get("filename")
    if not obj_id:
        logger.warn("missing_parameter", parameter="filename")
        return jsonify({"error": "Missing 'filename' parameter"}), 400

    structlog.contextvars.bind_contextvars(obj_id=obj_id)

    # Select directory
    obj_dir = PRE_GEN_OBJECTS_DIR if is_pregen else GEN_OBJECTS_DIR

    try:    
        # Ensure directory is set
        if not obj_dir:
            logger.error("Environment variable not set")
            return make_response(jsonify({"error": "Server misconfiguration"}), 500)

        # Construct file path safely
        file_path = Path(obj_dir) / f"{obj_id}.glb"

        # Check file existence
        if not file_path.exists() or not file_path.is_file():
            return make_response(jsonify({"error": "File not found"}), 404)

        # Serve the .glb file
        return send_file(
            file_path,
            as_attachment=False,
            mimetype="model/gltf-binary"
        )

    except Exception as e:
        return make_response(jsonify({"error": "Server error", "details": str(e)}), 500)

@chat_bp.route("/images", methods=["GET"])
@log_event("image_download", result_mapper=lambda r: {"status_code": r.status_code})
def get_image():
    """
    Returns a generated .png image file.
    Expects a 'filename' query parameter without extension.
    """
    is_pregen = request.args.get("pre_generated", "").lower() == "true" 
    img_id = request.args.get("filename")
    if not img_id:
        logger.warn("missing_parameter", parameter="filename")
        return jsonify({"error": "Missing 'filename' parameter"}), 400

    structlog.contextvars.bind_contextvars(img_id=img_id)


    # Select directory
    img_dir = PRE_GEN_IMAGES_DIR if is_pregen else GEN_IMAGES_DIR

    try:
        # Ensure directory is set
        if not img_dir:
            logger.error("Environment variable not set")
            return make_response(jsonify({"error": "Server misconfiguration"}), 500)

        # Construct file path safely
        file_path = Path(img_dir) / f"{img_id}.png"

        # Check file existence
        if not file_path.exists() or not file_path.is_file():
            logger.warning(f"Requested file not found: {file_path}")
            return make_response(jsonify({"error": "File not found"}), 404)

        # Serve the .png file
        return send_file(
            file_path,
            as_attachment=False,
            mimetype="image/png"
        )

    except Exception as e:
        logger.error("Error while fetching object", error=str(e))
        return make_response(jsonify({"error": "Server error", "details": str(e)}), 500)