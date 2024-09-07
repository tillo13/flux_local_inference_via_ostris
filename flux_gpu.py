import os  
import time  
import platform  
import psutil  
import csv  
import random  
from datetime import datetime  
from tqdm import tqdm  
import torch  
from diffusers import DiffusionPipeline  
from huggingface_hub import login  
from dotenv import load_dotenv  
  
# Load environment variables from .env file  
load_dotenv()  
  
# Retrieve the token from the .env file  
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  
  
# Ensure HF_TOKEN is set  
if HF_TOKEN is None:  
    raise ValueError("HUGGINGFACE_TOKEN is not set in the .env file")  
  
# Login to Hugging Face (this will handle global authentication)  
login(token=HF_TOKEN)  
  
# GLOBAL VARIABLES  
PROMPT = "a cat dressed as deadpool holding a sign saying: Ain't Flux neat, Andy?"  
MODEL_ID = "black-forest-labs/FLUX.1-dev"  # Use the dev version of the model  
NUMBER_OF_INFERENCE_STEPS = 1
CSV_FILENAME = "runtime_logs.csv"  
SEED = 42  
RANDOMIZE_SEED = True  
MAX_SEQUENCE_LENGTH = 512  
GUIDANCE_SCALE = 3.5  
HEIGHT = 1024  
WIDTH = 1024  
NUMBER_OF_LOOPS = 50  # Default number of images to generate  
  
# Ensure necessary packages are installed  
def ensure_packages():  
    required_packages = ["torch", "diffusers", "transformers", "accelerate", "requests", "tqdm", "psutil"]  
    for pkg in required_packages:  
        try:  
            __import__(pkg)  
        except ImportError:  
            print(f"Installing {pkg}...")  
            os.system(f"pip install {pkg}")  
  
ensure_packages()  
  
# Set TOKENIZERS_PARALLELISM to false to suppress warnings  
os.environ["TOKENIZERS_PARALLELISM"] = "false"  
  
# Utility function to time other functions  
def time_function(function, *args, **kwargs):  
    start_time = time.time()  
    result = function(*args, **kwargs)  
    end_time = time.time()  
    runtime = end_time - start_time  
    print(f"{function.__name__} runtime: {runtime:.2f} seconds")  
    return result, runtime  
  
# Create required directories  
def create_directories():  
    print("Creating directories...")  
    directories = ["generated_images"]  
    for directory in directories:  
        if not os.path.exists(directory):  
            os.makedirs(directory)  
            print(f"Created directory: {directory}")  
        else:  
            print(f"Directory already exists: {directory}")  
    print("Directories created")  
  
# Get system information  
def get_system_info():  
    info = {  
        "os": platform.system(),  
        "os_version": platform.version(),  
        "cpu": platform.processor(),  
        "cpu_cores": psutil.cpu_count(logical=False),  
        "cpu_threads": psutil.cpu_count(logical=True),  
        "ram": round(psutil.virtual_memory().total / (1024 ** 3), 2)  # Convert bytes to GB  
    }  
    return info  
  
# Log runtime and system info to CSV  
def log_to_csv(data):  
    file_exists = os.path.isfile(CSV_FILENAME)  
    with open(CSV_FILENAME, mode='a', newline='') as file:  
        writer = csv.DictWriter(file, fieldnames=data.keys())  
        if not file_exists:  
            writer.writeheader()  
        writer.writerow(data)  
  
# Main function to set up and run the model  
def main():  
    # Use GPU if available  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    # Print out the device being used  
    print(f"Using device: {device}")  
    create_dirs_result, create_dirs_time = time_function(create_directories)  
    # Set initial seed value  
    seed = SEED  
  
    try:  
        print("Loading model...")  
        start_time = time.time()  
        with tqdm(total=100, desc="Loading model") as pbar:  
            # Load the DiffusionPipeline directly  
            pipe = DiffusionPipeline.from_pretrained(  
                MODEL_ID,  
                torch_dtype=torch.float32,  
                low_cpu_mem_usage=True  
            ).to(device)  # Move the model to GPU
            pbar.update(100)  
        load_pipeline_time = time.time() - start_time  
        print(f"Load pipeline runtime: {load_pipeline_time:.2f} seconds")  
  
        total_generate_image_time = 0  
        total_save_image_time = 0  
  
        for i in range(NUMBER_OF_LOOPS):  
            # Randomize seed if necessary  
            if RANDOMIZE_SEED:  
                seed = random.randint(0, 1000000)  
            generator = torch.manual_seed(seed).to(device)  
            print(f"Generating image {i+1}/{NUMBER_OF_LOOPS} with seed {seed}...")  
            start_time = time.time()  
            # Generate image  
            image = pipe(  
                prompt=PROMPT,  
                height=HEIGHT,  
                width=WIDTH,  
                guidance_scale=GUIDANCE_SCALE,  
                output_type="pil",  
                num_inference_steps=NUMBER_OF_INFERENCE_STEPS,  
                max_sequence_length=MAX_SEQUENCE_LENGTH,  
                generator=generator  
            )["images"][0]  
            generate_image_time = time.time() - start_time  
            print(f"Generate image {i+1}/{NUMBER_OF_LOOPS} runtime: {generate_image_time:.2f} seconds")  
            total_generate_image_time += generate_image_time  
  
            start_time = time.time()  
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
            image_path = os.path.join("generated_images", f"{timestamp}_flux_creation_{i+1}.png")  
            image.save(image_path)  
            save_image_time = time.time() - start_time  
            print(f"Image {i+1} saved as '{image_path}'")  
            print(f"Save image {i+1}/{NUMBER_OF_LOOPS} runtime: {save_image_time:.2f} seconds")  
            total_save_image_time += save_image_time  
  
    except Exception as e:  
        print(f"An error occurred: {e}")  
        return  
  
    total_time = create_dirs_time + load_pipeline_time + total_generate_image_time + total_save_image_time  
    print("\n=== SUMMARY ===")  
    print(f"create_directories runtime: {create_dirs_time:.2f} seconds")  
    print(f"Load pipeline runtime: {load_pipeline_time:.2f} seconds")  
    print(f"Total generate image runtime: {total_generate_image_time:.2f} seconds")  
    print(f"Total save image runtime: {total_save_image_time:.2f} seconds")  
    print(f"Total runtime: {total_time:.2f} seconds")  
    # Log the runtime and system info  
    system_info = get_system_info()  
    log_data = {  
        "timestamp": timestamp,  
        "create_directories_runtime": create_dirs_time,  
        "load_pipeline_runtime": load_pipeline_time,  
        "total_generate_image_runtime": total_generate_image_time,  
        "total_save_image_runtime": total_save_image_time,  
        "total_runtime": total_time,  
        "prompt": PROMPT,  
        "num_inference_steps": NUMBER_OF_INFERENCE_STEPS,  
        "model_id": MODEL_ID,  
        "seed": seed,  
        "number_of_loops": NUMBER_OF_LOOPS,  
        **system_info  
    }  
    log_to_csv(log_data)  
  
if __name__ == "__main__":  
    main()