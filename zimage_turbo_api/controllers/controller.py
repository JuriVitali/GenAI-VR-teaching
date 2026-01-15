import structlog
from flask import Blueprint, request, jsonify
from services.image_generation_service import generate_image

# Gets the logger instance
logger = structlog.get_logger()

gen_bp = Blueprint("gen_bp", __name__)

@gen_bp.route("/generate", methods=["POST"])
def handle_generation():
    data = request.json
    prompt = data.get("prompt")
    summary = data.get("summary", False)
    
    if not prompt:
        logger.warn("missing_parameter", parameter="prompt")
        return jsonify({"error": "Prompt is required"}), 400

    try:
        # Trigger the generation service
        image_id = generate_image(prompt, summary)
        
        return jsonify({
            "status": "success",
            "image_id": image_id,
        }), 200
    except Exception as e:
        logger.error("generation_failed", error=str(e))
        return jsonify({"error": "Generation failed"}), 500