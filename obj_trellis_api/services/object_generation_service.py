import sys
sys.path.append("/trellis/TRELLIS")
import structlog
from trellis.utils import render_utils, postprocessing_utils
from trellis.pipelines import TrellisImageTo3DPipeline
from config.model_config_loader import ModelConfig
import torch
import random
import imageio
from PIL import Image
from dotenv import load_dotenv, find_dotenv
import os
import uuid
from shared.utils import log_event
import yaml

# Load environment variables    
load_dotenv(find_dotenv())

# Load model configuration params
model_config = ModelConfig()
image_to_3d_config = model_config.get("image_to_3d_model")
obj_postprocessing_config = model_config.get("objects_postprocessing")

# Object generation pipeline setup
pipeline_from_image = TrellisImageTo3DPipeline.from_pretrained(image_to_3d_config["model_id"]) #IMAGE_WORKFLOW
pipeline_from_image.cuda()

# Folder in which generated contents will be saved
gen_objects_dir = os.getenv("OBJECTS_DIR")
gen_images_dir = os.getenv("IMAGES_FOR_OBJ_DIR")
renders_dir = os.getenv("RENDERS_DIR")
os.makedirs(gen_objects_dir, exist_ok=True)
os.makedirs(renders_dir, exist_ok=True)

@log_event("object_generation", result_mapper=lambda x: {"obj_id": x})
def generate_object(prompt_img_id: str):

    structlog.contextvars.bind_contextvars(prompt_img_id=prompt_img_id)

    img_path = f"{gen_images_dir}/{prompt_img_id}.png"
    prompt_img = Image.open(img_path)

    outputs = pipeline_from_image.run(
        prompt_img, 
        seed=random.randint(0, 1000),
        sparse_structure_sampler_params={
            "steps": image_to_3d_config["model_params"]["sparse_structure_sampler"]["steps"],  
            "cfg_strength": image_to_3d_config["model_params"]["sparse_structure_sampler"]["cfg_strength"],   
        },
        slat_sampler_params={
            "steps": image_to_3d_config["model_params"]["slat_sampler"]["steps"], 
            "cfg_strength": image_to_3d_config["model_params"]["slat_sampler"]["cfg_strength"], 
        },
    )

    obj_filename = f"{uuid.uuid4().hex}"

    # DEBUG: Render the outputs
    # video = render_utils.render_video(outputs['gaussian'][0])['color'] 
    # imageio.mimsave(f"{renders_dir}/{obj_filename}.mp4", video, fps=30)
    
    # Postprocessing
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0], 
        outputs['mesh'][0],
        simplify = obj_postprocessing_config["simplify"], 
        texture_size = obj_postprocessing_config["texture_size"]
    )
    del outputs #Delete the generated objects explicitly
    
    # Saving object
    out_path = f"{gen_objects_dir}/{obj_filename}.glb"
    glb.export(out_path)
    
    return obj_filename
