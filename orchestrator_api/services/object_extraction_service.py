from config.model_config_loader import ModelConfig
from services.llm_service import get_llm_model
from pydantic import BaseModel, Field
import structlog
from typing import List
from shared.utils import log_event

# Load model configuration params
model_config = ModelConfig()
prompt_extraction_config = model_config.get("prompt_extraction")
prompt_extraction_model = get_llm_model(prompt_extraction_config["model"], prompt_extraction_config["temperature"])

# --- OUTPUT DATA STRUCTURES ---

class Summary_img(BaseModel):
    prompt: str = Field(
        ..., 
        description="The detailed textual description used to generate image."
    )
    caption: str = Field(
        ..., 
        description="A brief description of the image."
    )
    
class Obj(BaseModel):
    prompt: str = Field(
        ..., 
        description="The detailed textual description used to generate the 3D object."
    )
    speech: str = Field(
        ..., 
        description="The sentence the avatar will say to the user when presenting this specific object."
    )

class SceneItem(BaseModel):
    obj: Obj = Field(
        ..., 
        description="A 3D object to generate, including its visual prompt and avatar speech."
    )
    img: Summary_img = Field(
        ..., 
        description="An image to generate, including its visual prompt and its caption."
    )

# ---------------------------

@log_event("prompt_generation", result_mapper=lambda x: {"obj_prompt": x[0].prompt, "obj_description": x[0].speech, "summary_img_prompt": x[1].prompt, "summary_img_caption": x[1].caption})
def extract_prompts(question: str, answer: str) -> List[SceneItem]:
    """
    Analyze the question and answer to identify ...
    """
    structlog.contextvars.bind_contextvars(question=question, answer=answer)
    # Format the prompt using the updated YAML config
    prompt = prompt_extraction_config["prompt"].format(question=question, answer=answer)
    
    structured_llm_json = prompt_extraction_model.with_structured_output(
        SceneItem, 
        method="json_schema"
    )
    
    response = structured_llm_json.invoke(prompt)

    return response.obj, response.img