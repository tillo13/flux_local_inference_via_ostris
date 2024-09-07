import os
import logging
from dotenv import load_dotenv
import torch
from diffusers import FluxPipeline
from safetensors.torch import load_file as load_safetensors
from datetime import datetime
from PIL import Image

# Global variables for easy tweaking
NUMBER_OF_FILES = 5
OUTPUT_DIR = "F:/output/flux_lora_training/generated_images"
PROMPTS = [
    "andytillo on a mountain top, shouting to the wind",
    "andytillo playing a guitar at a concert, crowd cheering",
    "andytillo doing a handstand in a yoga class",
    "andytillo surfing a giant wave",
    "andytillo cooking in a busy restaurant kitchen"
]
SEED = 42
IMAGE_SIZE = (1024, 1024)  # Image size based on example
GUIDANCE_SCALE = 3.5  # Adjust for more or less adherence to the prompt
NUM_INFERENCE_STEPS = 28  # Number of denoising steps

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Set environment variables to use external SSD for Hugging Face cache
logging.info("Setting up environment variables...")
cache_dir = 'F:/lora/huggingface_cache'
os.environ['HF_HOME'] = cache_dir
os.environ['HF_HUB_CACHE'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = 'F:/lora/huggingface_datasets'
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ['DISABLE_TELEMETRY'] = 'YES'

# Your Hugging Face Token
HUGGING_FACE_TOKEN = os.getenv("HF_TOKEN")
if not HUGGING_FACE_TOKEN:
    raise ValueError("Hugging Face token not found in environment variables.")

# Set the Huggingface token to use for the session
os.environ["HUGGINGFACE_TOKEN"] = HUGGING_FACE_TOKEN

# Your LoRA weights path
lora_weights_path = "F:/output/flux_lora_training/flux_lora_training.safetensors"

# Ensure required directores exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and set up FLUX.1 [dev] with LoRA
def setup_pipeline(lora_weights_path):
    pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, cache_dir=cache_dir)
    pipeline.enable_model_cpu_offload()  # Offload to save VRAM, adjust as needed

    # Load LoRA weights
    lora_state_dict = load_safetensors(lora_weights_path)

    # Apply LoRA weights
    pipeline.text_encoder.load_state_dict({**pipeline.text_encoder.state_dict(), **lora_state_dict}, strict=False)
    return pipeline

# Function to generate an image based on prompt
def generate_image(pipeline, prompt: str, steps: int = NUM_INFERENCE_STEPS, guidance_scale: float = GUIDANCE_SCALE):
    generator = torch.Generator("cuda").manual_seed(SEED)
    result = pipeline(prompt=prompt, height=IMAGE_SIZE[0], width=IMAGE_SIZE[1], guidance_scale=guidance_scale, num_inference_steps=steps, generator=generator)
    image = result.images[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(OUTPUT_DIR, f"{timestamp}_{prompt.replace(' ', '_')}.png")
    image.save(file_name)
    logging.info(f"Image saved as {file_name}")

def main():
    pipe = setup_pipeline(lora_weights_path)
    for prompt in PROMPTS[:NUMBER_OF_FILES]:
        generate_image(pipe, prompt)

if __name__ == "__main__":
    main()