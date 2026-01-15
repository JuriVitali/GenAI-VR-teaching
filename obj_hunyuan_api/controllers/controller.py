import structlog
from flask import Blueprint, request, jsonify
from services.object_generation_service import generate_object

# Gets the logger instance
logger = structlog.get_logger()

gen_bp = Blueprint("gen_bp", __name__)

@gen_bp.route("/generate", methods=["POST"])
def handle_generation():
    data = request.json
    prompt_img_id = data.get("img_id")
    
    if not prompt_img_id:
        logger.warn("missing_parameter", parameter="img_id")
        return jsonify({"error": "Prompt is required"}), 400

    try:
        # Trigger the generation service
        obj_id = generate_object(prompt_img_id)
        
        return jsonify({
            "status": "success",
            "object_id": obj_id,
        }), 200
    except Exception as e:
        logger.error("generation_failed", error=str(e))
        return jsonify({"error": "Generation failed"}), 500