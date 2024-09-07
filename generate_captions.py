import os
import shutil  # Ensure shutil is imported
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Set environment variables to use external SSD for Hugging Face cache
os.environ['HF_HOME'] = 'F:/lora/huggingface_cache'
os.environ['HF_HUB_CACHE'] = 'F:/lora/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = 'F:/lora/huggingface_datasets'

# Create necessary directories if they don't exist
Path('F:/lora/huggingface_cache').mkdir(parents=True, exist_ok=True)
Path('F:/lora/huggingface_datasets').mkdir(parents=True, exist_ok=True)

# Function to delete the existing cache in C:\
def clear_existing_cache():
    old_hf_hub_cache = Path.home() / '.cache/huggingface/hub'
    old_hf_datasets_cache = Path.home() / '.cache/huggingface/datasets'
    
    if old_hf_hub_cache.exists():
        print(f'Removing existing Hugging Face hub cache at {old_hf_hub_cache}')
        shutil.rmtree(old_hf_hub_cache)
        print('Old hub cache removed.')
        
    if old_hf_datasets_cache.exists():
        print(f'Removing existing Hugging Face datasets cache at {old_hf_datasets_cache}')
        shutil.rmtree(old_hf_datasets_cache)
        print('Old datasets cache removed.')

# Clear existing cache
clear_existing_cache()

# Function to verify the cache setup
def verify_cache_setup():
    hf_home = os.environ.get('HF_HOME')
    hf_cache = os.environ.get('HF_HUB_CACHE')
    datasets_cache = os.environ.get('HF_DATASETS_CACHE')
    print(f'HF_HOME: {hf_home}')
    print(f'HF_HUB_CACHE: {hf_cache}')
    print(f'HF_DATASETS_CACHE: {datasets_cache}')

# Verify cache setup
verify_cache_setup()

# Load BLIP model and processor
print('Loading BLIP model and processor...')
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir='F:/lora/huggingface_cache')
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir='F:/lora/huggingface_cache')
print('BLIP model and processor loaded.')

# Directory containing images
image_dir = 'input_images2'
image_exts = ['.jpg', '.jpeg', '.png']
trigger_word = "amae"  # Replace with your chosen trigger word

# Function to generate captions and save them with verbose logging
def generate_captions(image_dir, image_exts):
    print(f'Starting to generate captions for images in {image_dir}...')
    for filename in os.listdir(image_dir):
        if any(filename.lower().endswith(ext) for ext in image_exts):
            image_path = os.path.join(image_dir, filename)
            print(f'Processing {image_path}...')
            image = Image.open(image_path).convert("RGB")
            
            # Generate caption
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                caption_ids = model.generate(**inputs)
            caption = processor.decode(caption_ids[0], skip_special_tokens=True)
            
            # Append the trigger word to the caption
            caption_with_trigger = f"{trigger_word} {caption}"
            
            # Save caption to a txt file with the same name as the image
            txt_path = os.path.splitext(image_path)[0] + '.txt'
            with open(txt_path, 'w') as txt_file:
                txt_file.write(caption_with_trigger)
            print(f'Captioned {filename}: {caption_with_trigger}')
    print('Caption generation completed.')

# Run the caption generation
generate_captions(image_dir, image_exts)