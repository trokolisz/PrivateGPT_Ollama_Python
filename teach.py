import pandas as pd
import requests
import json
import os
import logging
from typing import Dict, Any
from rich.console import Console
from rich.progress import Progress

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

console = Console()

def create_modelfile(base_model="llama2", tag="custom"):
    """Create a Modelfile for training"""
    try:
        modelfile = f"""
FROM {base_model}
PARAMETER temperature 0.7
PARAMETER top_p 0.7
PARAMETER top_k 50
"""
        with open("Modelfile", "w") as f:
            f.write(modelfile)
        logging.debug(f"Created Modelfile with base model {base_model}")
    except IOError as e:
        logging.error(f"Failed to create Modelfile: {e}")
        raise

def check_api_response(response: requests.Response, operation: str) -> Dict[str, Any]:
    """Validate API response"""
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"{operation} failed: {e}")
        raise Exception(f"API {operation} failed: {str(e)}")

def train_model(content: str, tag="custom") -> Dict[str, Any]:
    """Train the model with new content"""
    logging.info(f"Starting model training with tag: {tag}")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Training model...", total=3)
        
        # Create new model
        create_modelfile(tag=tag)
        progress.advance(task)
        
        # Create the model
        create_url = "http://localhost:11434/api/create"
        create_data = {
            "name": f"llama2:{tag}",
            "path": "Modelfile"
        }
        logging.debug(f"Sending create request: {json.dumps(create_data)}")
        response = requests.post(create_url, json=create_data)
        result = check_api_response(response, "model creation")
        progress.advance(task)
        
        # Train with the content
        train_url = "http://localhost:11434/api/train"
        train_data = {
            "name": f"llama2:{tag}",
            "training_data": content
        }
        logging.debug(f"Sending training request for model llama2:{tag}")
        response = requests.post(train_url, json=train_data)
        result = check_api_response(response, "model training")
        progress.advance(task)
        
        return result

def read_file(file_path: str) -> str:
    """Read content from txt or csv files"""
    logging.info(f"Reading file: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            return df.to_string()
        else:
            raise ValueError("Unsupported file format. Please use .txt or .csv files")
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        raise

def query_llama(content: str, tag="custom") -> str:
    """Send content to Llama model"""
    logging.info(f"Querying model llama2:{tag}")
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": f"llama2:{tag}",
        "prompt": content,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        result = check_api_response(response, "query")
        return result['response']
    except Exception as e:
        logging.error(f"Query failed: {e}")
        raise

def main():
    try:
        console.print("[bold green]Welcome to the Llama2 Training Interface[/bold green]")
        file_path = console.input("[yellow]Enter the path to your txt or csv file: [/yellow]")
        
        with console.status("[bold blue]Processing...[/bold blue]") as status:
            content = read_file(file_path)
            console.print("[green]File read successfully![/green]")
            
            console.print("\n[bold blue]Training model with new content...[/bold blue]")
            train_response = train_model(content)
            console.print("[green]Training complete![/green]")
            
            test_prompt = console.input("\n[yellow]Enter a test prompt to verify learning: [/yellow]")
            response = query_llama(test_prompt)
            
            console.print("\n[bold cyan]Llama's response:[/bold cyan]")
            console.print(f"[white]{response}[/white]")
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        logging.exception("An error occurred during execution")
    finally:
        logging.info("Session ended")

if __name__ == "__main__":
    main()
