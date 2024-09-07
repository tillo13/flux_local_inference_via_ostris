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
  
# GLOBAL VARIABLES  
PROMPT = ("a cat dressed as deadpool holding a sign saying: Yo, Andy!  Flux?")
 
MODEL_ID = "black-forest-labs/FLUX.1-schnell"  
NUMBER_OF_INFERENCE_STEPS = 2
CSV_FILENAME = "runtime_logs.csv"  
SEED = 42  
RANDOMIZE_SEED = True  
NUMBER_OF_LOOPS = 50  # Default number of image creations  
  
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
    directories = [  
        "generated_images"  
    ]  
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
    # Use CPU only  
    device = torch.device("cpu")  
    # Print out the device being used  
    print(f"Using device: {device}")  
    create_dirs_result, create_dirs_time = time_function(create_directories)  
    # Determine seed value  
    base_seed = random.randint(0, 1000000) if RANDOMIZE_SEED else SEED
  
    try:  
        print("Loading model...")  
        start_time = time.time()  
        with tqdm(total=100, desc="Loading model") as pbar:  
            # Load the DiffusionPipeline directly  
            pipe = DiffusionPipeline.from_pretrained(  
                MODEL_ID,  
                torch_dtype=torch.float32,  
                low_cpu_mem_usage=True  
            )  
            pbar.update(100)  
        load_pipeline_time = time.time() - start_time  
        print(f"Load pipeline runtime: {load_pipeline_time:.2f} seconds")  
          
        for i in range(NUMBER_OF_LOOPS):  
            seed = base_seed + i  
            generator = torch.manual_seed(seed)  
            print(f"Generating image {i+1} with seed {seed}...")  
            start_time = time.time()  
            # Generate image  
            image = pipe(prompt=PROMPT, generator=generator, num_inference_steps=NUMBER_OF_INFERENCE_STEPS)["images"][0]  
            generate_image_time = time.time() - start_time  
            print(f"Generate image runtime: {generate_image_time:.2f} seconds")  
            start_time = time.time()  
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
            image_path = os.path.join("generated_images", f"{timestamp}_flux_creation_{i+1}.png")  
            image.save(image_path)  
            save_image_time = time.time() - start_time  
            print(f"Image {i+1} saved as '{image_path}'")  
            print(f"Save image runtime: {save_image_time:.2f} seconds")  
  
            total_time = create_dirs_time + load_pipeline_time + generate_image_time + save_image_time  
            print("\n=== SUMMARY ===")  
            print(f"create_directories runtime: {create_dirs_time:.2f} seconds")  
            print(f"Load pipeline runtime: {load_pipeline_time:.2f} seconds")  
            print(f"Generate image runtime: {generate_image_time:.2f} seconds")  
            print(f"Save image runtime: {save_image_time:.2f} seconds")  
            print(f"Total runtime: {total_time:.2f} seconds")  
  
            # Log the runtime and system info  
            system_info = get_system_info()  
            log_data = {  
                "timestamp": timestamp,  
                "create_directories_runtime": create_dirs_time,  
                "load_pipeline_runtime": load_pipeline_time,  
                "generate_image_runtime": generate_image_time,  
                "save_image_runtime": save_image_time,  
                "total_runtime": total_time,  
                "prompt": PROMPT,  
                "num_inference_steps": NUMBER_OF_INFERENCE_STEPS,  
                "model_id": MODEL_ID,  
                "seed": seed,  
                **system_info  
            }  
            log_to_csv(log_data)  
  
    except Exception as e:  
        print(f"An error occurred: {e}")  
        return  
  
if __name__ == "__main__":  
    main()