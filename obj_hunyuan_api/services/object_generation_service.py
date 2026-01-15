import structlog
import torch
import trimesh
import random
import imageio
from PIL import Image
from dotenv import load_dotenv, find_dotenv
import os
import uuid
import time
import yaml
import sys
from config.model_config_loader import ModelConfig

# Define the base path to where you cloned the repo
repo_base_path = "/hunyuan3d_20/Hunyuan3D-2.1"

# Add the specific sub-folders to the path
sys.path.insert(0, os.path.join(repo_base_path, 'hy3dshape'))
sys.path.insert(0, os.path.join(repo_base_path, 'hy3dpaint'))
from textureGenPipeline import Hunyuan3DPaintPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.rembg import BackgroundRemover

# Gets the bound logger instance
logger = structlog.get_logger()

# Load environment variables    
load_dotenv(find_dotenv())

# Load model configuration params
model_config = ModelConfig()
image_to_3d_config = model_config.get("image_to_3d_model")

# Folder in which generated contents will be saved
gen_objects_dir = os.getenv("GEN_OBJECTS_DIR")
gen_images_dir = os.getenv("GEN_IMAGES_DIR")
renders_dir = os.getenv("RENDERS_DIR")
os.makedirs(gen_objects_dir, exist_ok=True)
os.makedirs(renders_dir, exist_ok=True)

shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')

rembg = BackgroundRemover()

# paint
conf = Hunyuan3DPaintConfig(
    max_num_view=6,      # Speed: 6 | Quality: 9
    resolution=512       # Speed: 512 | Quality: 1024
)
# Force PBR (Physically Based Rendering) for more realistic materials
conf.multiview_cfg_path = f"{repo_base_path}/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = f"{repo_base_path}/hy3dpaint/hunyuanpaintpbr"
paint_pipeline = Hunyuan3DPaintPipeline(conf)


def generate_object(prompt_img_id: str):

    img_path = f"{gen_images_dir}/{prompt_img_id}.png"
    image = Image.open(img_path).convert("RGBA")

    # Remove background to get clean boundaries
    image = rembg(image)

    logger.info("Generating the object ...")
    start_obj_generation= time.time()

    # let's generate a mesh first
    mesh_untextured = shape_pipeline(
        image=image, 
        num_inference_steps=25,       # Default is 50. Reduced to 25-30 to speed up the process.
        octree_resolution=256,        # Lowering this speeds up "Volume Decoding".
        guidance_scale=7.5            # Keep around 5.0 - 7.5
    )[0]

    # Simplify mesh before texturing
    mesh_untextured = mesh_untextured.simplify_quadratic_decimation(40000)

    filename = f"{uuid.uuid4().hex}"

    temp_mesh_path = f"{gen_objects_dir}/{filename}_raw.obj"
    mesh_untextured.export(temp_mesh_path)
    logger.info(f"Untextured mesh saved to {temp_mesh_path}")

    mesh_textured = paint_pipeline(temp_mesh_path, image_path=img_path)

    logger.info("Object generated succesfully", duration=(time.time() - start_obj_generation))

    # Load the OBJ (and its associated MTL/Textures)
    mesh_to_export = trimesh.load(mesh_textured)

    out_path = f"{gen_objects_dir}/{filename}.glb"
    mesh_to_export.export(out_path)
    
    os.remove(temp_mesh_path)
    return filename