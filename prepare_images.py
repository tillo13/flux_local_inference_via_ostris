import os
import shutil
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import re

image_dir = 'input_images2'
output_dir = 'prepared_images'
trigger_word = "andytillo"  


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
image_exts = ['.jpg', '.jpeg', '.png']
target_resolutions = [512, 768, 1024]

# Create output directory if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to normalize filenames
def normalize_filename(filename):
    name, ext = os.path.splitext(filename)
    normalized = re.sub(r'[^a-zA-Z0-9]', '_', name).lower()
    return f"{normalized}{ext}"

# Normalizes the filenames in the input directory
def normalize_filenames(input_dir):
    print(f'Normalizing filenames in {input_dir}...')
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in image_exts):
            normalized_filename = normalize_filename(filename)
            if filename != normalized_filename:
                os.rename(
                    os.path.join(input_dir, filename),
                    os.path.join(input_dir, normalized_filename)
                )
                print(f'Renamed {filename} to {normalized_filename}')
    print('Filename normalization completed.')

# Function to resize image maintaining aspect ratio
def resize_image_to_target(image, target_size):
    aspect_ratio = image.width / image.height
    if aspect_ratio > 1:  # Horizontal image
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:  # Vertical or square image
        new_width = int(target_size * aspect_ratio)
        new_height = target_size
    return image.resize((new_width, new_height), Image.LANCZOS)

# Function to process images: resize and save them
def process_images(image_dir, output_dir, target_resolutions):
    print(f'Resizing images in {image_dir} and saving to {output_dir}...')
    for filename in os.listdir(image_dir):
        if any(filename.lower().endswith(ext) for ext in image_exts):
            image_path = os.path.join(image_dir, filename)
            with Image.open(image_path).convert("RGB") as image:
                for target_size in target_resolutions:
                    resized_image = resize_image_to_target(image, target_size)
                    new_filename = f"{os.path.splitext(filename)[0]}_{target_size}.jpg"
                    output_path = os.path.join(output_dir, new_filename)
                    resized_image.save(output_path, quality=95)
                    print(f'Resized and saved {filename} to {new_filename}')
    print('Image resizing completed.')

# Function to generate captions and save them with verbose logging
def generate_captions(output_dir, image_exts):
    print(f'Starting to generate captions for images in {output_dir}...')
    for filename in os.listdir(output_dir):
        if any(filename.lower().endswith(ext) for ext in image_exts):
            image_path = os.path.join(output_dir, filename)
            print(f'Processing {image_path}...')
            with Image.open(image_path).convert("RGB") as image:
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

# Normalize filenames
normalize_filenames(image_dir)
# Process images and generate captions
process_images(image_dir, output_dir, target_resolutions)
generate_captions(output_dir, image_exts)