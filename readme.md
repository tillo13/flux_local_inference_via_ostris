# LoRA Creation via AI-Toolkit Offloading to External Drive

## Overview
This project builds upon the original AI-Toolkit repository by [Ostris](https://github.com/ostris/ai-toolkit) to add functionality for external drive offloading and introduces several new utility scripts. The enhancements aim to simplify the process of generating captions, preparing images, and running inference efficiently, particularly for systems with limited VRAM and storage.

## Enhancements

### 1. External Drive Offloading
Environment variables have been set to enable the use of an external SSD for caching, ensuring the main drive isn't overburdened. The modified `run.py` reflects these changes.

### 2. New Utility Scripts
- **flux_gpu.py**: Script for running the FLUX model on a GPU.
- **flux_mac_cpu.py**: Script for running the FLUX model on a Mac CPU.
- **generate_captions.py**: Script for generating captions for images.
- **prepare_images.py**: Script for normalizing, resizing, and captioning images.
- **test_new_lora.py**: Script to generate images using the newly trained LoRA model.

## Installation

### Requirements
- Python >3.10
- Nvidia GPU with at least 12GB VRAM (e.g., RTX 3060)
- External SSD for caching (optional but recommended)
- Python venv
- Git

### Clone the Repository
```sh
git clone https://github.com/tillo13/lora-creation-with-ostris.git
cd lora-creation-with-ostris
```

### Setup Virtual Environment
For Linux:
```sh
python3 -m venv venv
source venv/bin/activate
```

For Windows:
```sh
python -m venv venv
.\venv\Scripts\activate
```

### Install Dependencies
```sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Step-by-Step Process

### 1. Generate Captions for Images
Use the `generate_captions.py` script to generate captions for your images stored in the `input_images2` directory.

Usage:
```sh
python generate_captions.py
```
This script will create captions for your images and save them in the same directory as the images.

### 2. Prepare Images
Next, use the `prepare_images.py` script to normalize filenames, resize images to multiple resolutions, and generate captions.

Usage:
```sh
python prepare_images.py
```
This script processes the images in `input_images2`, resizes them to 512, 768, and 1024 pixels, and generates captions with a trigger word.

### 3. Run the AI-Toolkit with Custom Config
Modify the `run.py` script to use an external SSD for caching and run your custom configurations.

Usage:
```sh
python run.py path/to/config.yml
```
Make sure your `.env` file has the Hugging Face token and necessary configurations.

### 4. Test the Trained LoRA Model
After running the training, use the `test_new_lora.py` script to generate images using the newly trained LoRA model.

Usage:
```sh
python test_new_lora.py
```
This script applies LoRA weights to the FLUX model and generates images based on various prompts, saving them to an output directory.

### 5. Generate Images without LoRA
If you want to generate images using the FLUX model without LoRA, you can use the `flux_gpu.py` or `flux_mac_cpu.py` scripts depending on your platform.

#### Using GPU
Usage:
```sh
python flux_gpu.py
```
This script will utilize your GPU to generate images based on a pre-defined prompt.

#### Using Mac CPU
Usage:
```sh
python flux_mac_cpu.py
```
This script is optimized to run on a Mac CPU, generating images based on a pre-defined prompt.

## Scripts and Their Usage

### flux_gpu.py
This script is designed to run inference using the FLUX model on a GPU. It generates images based on a pre-defined prompt while logging runtime metrics.

#### Features:
- Utilizes GPU for faster image generation.
- Logs runtime and system information to a CSV file.
- Generates multiple images with a configurable prompt and settings.

### flux_mac_cpu.py
This script is aimed at Mac users who want to run the FLUX model on their CPU. It generates images by leveraging the computational power of a Mac's CPU.

#### Features:
- Utilizes CPU for image generation.
- Configurable prompts and image settings.
- Logs runtime and system information.

### generate_captions.py
This script generates captions for images stored in the `input_images2` directory. It utilizes the BLIP model from Salesforce for image captioning.

#### Features:
- Generates captions for each image.
- Saves captions to text files with the same name as the images.
- Uses an external SSD for Hugging Face cache to enhance performance.

### prepare_images.py
This script normalizes filenames, resizes images to multiple resolutions, and generates captions. It is designed to preprocess images for further use.

#### Features:
- Normalizes filenames to ensure compatibility.
- Resizes images while maintaining aspect ratio.
- Generates and appends captions to each image.
- Uses an external SSD for Hugging Face cache to enhance performance.

### test_new_lora.py
This script generates images using a newly trained LoRA (Low-Rank Adaptation) model. It applies LoRA weights to the FLUX.1 [dev] model and generates images based on different prompts.

#### Features:
- Loads and applies LoRA weights to the FLUX model.
- Generates images with multiple prompts.
- Saves generated images to a specified output directory.
- Uses an external SSD for Hugging Face cache to enhance performance.

## Modified run.py
This script serves as the main entry point for running jobs with the AI-Toolkit. It has been modified to support caching on an external SSD.

#### Features:
- Sets environment variables to use an external SSD for Hugging Face cache.
- Runs jobs based on provided config files.
- Supports continuation in case of failure with the `--recover` flag.

## Directory Structure
```sh
my-ai-toolkit/
├── docker/
├── extensions/
├── venv/
├── toolkit/
│   ├── job.py
│   ├── ...
│   └── ...
├── .env
├── requirements.txt
├── flux_gpu.py
├── flux_mac_cpu.py
├── generate_captions.py
├── prepare_images.py
├── test_new_lora.py
├── run.py
└── README.md
```

## Credits
This project is based on the Ostris AI Toolkit. Special thanks to Ostris for the original repository and their continued development efforts.

## License
This project is licensed under the MIT License.