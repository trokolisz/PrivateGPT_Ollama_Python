import json
import pathlib
import time

import httpx
import ollama
import os


OLLAMA_CONNECTION_STRING = os.environ.get(
    "OLLAMA_CONNECTION_STRING", "http://localhost:11434"
    )
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

PROMPT_TEMPLATE_PATH = os.environ.get("PROMPT_TEMPLATE_PATH", "prompt.txt")


SAMPLE_DESCRIPTION = "Generate a description based on logs"

SAMPLE_LOGS = [
    "2024-01-15 08:30:12 INFO: Application startup complete",
    "2024-01-15 08:32:45 WARNING: High memory usage detected (85%)",
    "2024-01-15 08:35:23 ERROR: Database connection timeout",
    "2024-01-15 08:36:01 INFO: User authentication successful: user_id=12345",
    "2024-01-15 08:40:55 DEBUG: Cache refresh completed in 1.2s",
    "2024-01-15 08:41:03 ERROR: Failed to process payment transaction: TX_ID_789",
    "2024-01-15 08:45:30 INFO: Scheduled backup started",
    "2024-01-15 08:46:15 WARNING: API rate limit approaching threshold"
]

def wait_for_ollama(ollama_client: ollama.Client):
    print("Connecting to Ollama server...")
    tries = 10
    while tries > 0:
        try:
            ollama_client.ps()
            print("Successfully connected to Ollama server")
            break
        except httpx.RequestError:
            print(f"Waiting for Ollama server... {tries} attempts remaining")
            time.sleep(1)
            tries -= 1
    if tries == 0:
        raise RuntimeError("Could not connect to OLLAMA")
    
def download_model(ollama_client: ollama.Client, model: str):
    print(f"\nChecking for model {model}...")
    models_list = ollama_client.list()
    print("Available models: ", models_list)
    existing_models = []
    if model in models_list:
        existing_models = [model_info.get("name", "") for model_info in models_list["models"]]
    if model not in existing_models:
        print(f"Downloading model {model}... This might take a while...")
        ollama_client.pull(model)
        print(f"Model {model} successfully downloaded")
    else:
        print(f"Model {model} already exists, skipping download")
    

def read_prompt_template(template_path: str) -> str:
    try:
        with open(template_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise RuntimeError(f"Prompt template file not found: {template_path}")

def generate_log_description(
    ollama_client: ollama.Client,
    model: str,
    logs: list[str],
    prompt_template: str
) -> str:
    try:
        print("\nFormatting logs and preparing prompt...")
        formatted_logs = "\n".join(logs)
        prompt = prompt_template.format(logs=formatted_logs)
        
        print("Generating description using Ollama... Please wait...")
        response = ollama_client.generate(
            model=model,
            prompt=prompt,
            stream=False
        )
        print("Description generated successfully")
        
        return response['response']
    except KeyError as e:
        raise RuntimeError(f"Invalid prompt template: missing placeholder {e}")

def main():
    print("\n=== Log Analysis System Starting ===\n")
    
    ollama_client = ollama.Client(host=OLLAMA_CONNECTION_STRING)
    wait_for_ollama(ollama_client)
    download_model(ollama_client, OLLAMA_MODEL)
    
    print(f"\nReading prompt template from {PROMPT_TEMPLATE_PATH}...")
    prompt_template = read_prompt_template(PROMPT_TEMPLATE_PATH)
    print("Prompt template loaded successfully")
    
    description = generate_log_description(
        ollama_client,
        OLLAMA_MODEL,
        SAMPLE_LOGS,
        prompt_template
    )
    
    print("\n=== Generated Log Analysis ===")
    print(description)
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()