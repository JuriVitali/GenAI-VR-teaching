import torch
import trimesh
import random
import imageio
from PIL import Image
from dotenv import load_dotenv, find_dotenv
import os
import uuid
import structlog
import yaml
from diffusers import ZImagePipeline
import sys
from config.model_config_loader import ModelConfig
from shared.utils import log_event

# Load environment variables    
load_dotenv(find_dotenv())

# Load model configuration params
model_config = ModelConfig()
image_generator_config = model_config.get("image_generator")

# --------- Z-Image Turbo -----------------------------------------------
# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# [Optional] Attention Backend
# Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
# pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
# pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3

# [Optional] Model Compilation
# Compiling the DiT model accelerates inference, but the first run will take longer to compile.
# pipe.transformer.compile()

# [Optional] CPU Offloading
# Enable CPU offloading for memory-constrained devices.
# pipe.enable_model_cpu_offload()
# -------------------------------------------------------------------

# Folder where generated images will be saved
images_for_obj_dir = os.getenv("IMAGES_FOR_OBJ_DIR")
summary_images_dir = os.getenv("SUMMARY_IMAGES_DIR")
os.makedirs(images_for_obj_dir, exist_ok=True)
os.makedirs(summary_images_dir, exist_ok=True)

@log_event("image_generation", result_mapper=lambda x: {"img_id": x})
def generate_image(prompt: str, summary: bool):
    structlog.contextvars.bind_contextvars(prompt=prompt)

    #Generate Image
    generated_image = pipe(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=9,  # This actually results in 8 DiT forwards
        guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
        generator=torch.Generator("cuda").manual_seed(20),
    ).images[0]

    # Saving image
    img_filename = f"{uuid.uuid4().hex}"
    if summary:
        img_path = f"{summary_images_dir}/{img_filename}.png"
    else:
        img_path = f"{images_for_obj_dir}/{img_filename}.png"

    generated_image.save(img_path)
    return img_filename
