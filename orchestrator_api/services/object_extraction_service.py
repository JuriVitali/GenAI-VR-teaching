from config.model_config_loader import ModelConfig
from services.llm_service import get_llm_model
from pydantic import BaseModel, Field
import structlog
from typing import List
from shared.utils import log_event

# Load model configuration params
model_config = ModelConfig()
prompt_extraction_config = model_config.get("prompt_extraction")
prompt_enhancement_config = model_config.get("prompt_enhancement")
prompt_enhancement_model = get_llm_model(prompt_extraction_config["model"], prompt_extraction_config["temperature"])
prompt_extraction_model = get_llm_model(prompt_extraction_config["model"], prompt_extraction_config["temperature"])

# --- OUTPUT DATA STRUCTURES ---

# --- 2D IMAGE ASSET ---
class VisualAsset2D(BaseModel):
    prompt: str = Field(
        ..., 
        description=(
            "The prompt for a text-to-image model. "
            "Describe a realistic, immersive educational scene or close-up. "
            "CRITICAL RULES: 1. NO SPLIT SCREENS (old vs new). 2. NO CHARTS/GRAPHS/TEXT. "
            "3. If vague, add historical/material details (e.g., '1920s wooden radio')."
        )
    )
    caption: str = Field(
        ..., 
        description="A concise figure legend explaining the image (max 15 words). Language: Target Language."
    )

# --- 3D OBJECT ASSET ---
class VisualAsset3D(BaseModel):
    prompt: str = Field(
        ..., 
        description=(
            "The prompt for Trellis (Image-to-3D). "
            "Describe a SINGLE, SOLID, OPAQUE object. "
            "MANDATORY FORMAT: Start exactly with 'Professional studio photography of [Subject]...'. "
            "CRITICAL RULES: 1. REALISM (No toys/models). 2. NO TRANSPARENCY (No glass/water/fire). "
            "3. NO SCENES (Must be isolated on white). 4. VIEW: High-angle three-quarter view (45 degrees)."
        )
    )
    presentation_speech: str = Field(
        ..., 
        description="A single spoken sentence for the Avatar to introduce this object. Language: Target Language."
    )

# --- MAIN CONTAINER ---
class SceneGeneration(BaseModel):
    summary_image: VisualAsset2D = Field(
        ..., 
        description="The 2D illustration for the summary panel."
    )
    obj: VisualAsset3D = Field(
        ..., 
        description="The 3D object asset to be projected in VR."
    )

# ---------------------------

@log_event("prompt_generation", result_mapper=lambda x: {"obj_prompt": x[0].prompt, "obj_description": x[0].presentation_speech, "summary_img_prompt": x[1].prompt, "summary_img_caption": x[1].caption})
def extract_prompts(question: str, answer: str, context, language) -> SceneGeneration:
    """
    Analyzes the Q&A to generate visual assets using Pydantic structured output.
    """
    # Bind context for logging
    structlog.contextvars.bind_contextvars(question=question, answer=answer)
    
    # format the prompt template defined above
    formatted_prompt = prompt_extraction_config["prompt"].format(
        question=question, 
        answer=answer, 
        context=context,
        language=language
    )
    
    # Configure the LLM with the new Pydantic schema
    structured_llm = prompt_extraction_model.with_structured_output(
        SceneGeneration, 
        method="json_schema" # or "function_calling" depending on backend
    )
    
    # Invoke
    response = structured_llm.invoke(formatted_prompt)

    return response.obj, response.summary_image

@log_event("prompt_improvement")
def improve_prompt(raw_prompt: str):
    structlog.contextvars.bind_contextvars(raw_prompt=raw_prompt)
    full_prompt = prompt_enhancement_config["prompt"].format(raw_prompt=raw_prompt)
    improved_prompt = prompt_enhancement_model.invoke(full_prompt)

    return improved_prompt.content