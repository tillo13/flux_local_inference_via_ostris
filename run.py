import os
import sys
import time
import argparse
import logging
from typing import Union, OrderedDict
from dotenv import load_dotenv
from toolkit.job import get_job

# Load the .env file if it exists
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set environment variables to use external SSD for Hugging Face cache
logging.info("Setting up environment variables...")
os.environ['HF_HOME'] = 'F:/lora/huggingface_cache'
os.environ['HF_HUB_CACHE'] = 'F:/lora/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = 'F:/lora/huggingface_datasets'
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ['DISABLE_TELEMETRY'] = 'YES'

sys.path.insert(0, os.getcwd())

# Check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    import torch
    torch.autograd.set_detect_anomaly(True)

def print_end_message(jobs_completed, jobs_failed):
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    logging.info("")
    logging.info("========================================")
    logging.info("Result:")
    if len(completed_string) > 0:
        logging.info(f" - {completed_string}")
    if len(failure_string) > 0:
        logging.info(f" - {failure_string}")
    logging.info("========================================")

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()

    # Require at least one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # Flag to continue if failed job
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # Flag to continue if failed job
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    args = parser.parse_args()
    config_file_list = args.config_file_list

    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    jobs_completed = 0
    jobs_failed = 0

    logging.info(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        try:
            job_start_time = time.time()
            logging.info(f"Starting job with config file: {config_file}")
            job = get_job(config_file, args.name)
            job.run()
            job.cleanup()
            jobs_completed += 1
            job_end_time = time.time()
            logging.info(f"Completed job with config file: {config_file} in {job_end_time - job_start_time:.2f} seconds.")
        except Exception as e:
            logging.error(f"Error running job: {e}")
            jobs_failed += 1
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e

    end_time = time.time()
    logging.info(f"Total time for all jobs: {end_time - start_time:.2f} seconds.")
    print_end_message(jobs_completed, jobs_failed)

if __name__ == '__main__':
    main()