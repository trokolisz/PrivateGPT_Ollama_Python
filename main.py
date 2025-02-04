import json
import logging
import pathlib
import time
from dataclasses import dataclass
from typing import List, Optional

import httpx
import ollama
import os

# Sample logs for testing
SAMPLE_LOGS = [
    "2024-01-20 10:15:23 INFO Server started successfully",
    "2024-01-20 10:15:24 ERROR Database connection failed",
    "2024-01-20 10:15:25 WARNING High memory usage detected"
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration for the log analysis system."""
    connection_string: str
    model_name: str
    prompt_template_path: str
    timeout: int = 30
    max_retries: int = 3
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls(
            connection_string=os.environ.get("OLLAMA_CONNECTION_STRING", "http://localhost:11434"),
            model_name=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
            prompt_template_path=os.environ.get("PROMPT_TEMPLATE_PATH", "prompt.txt"),
            timeout=int(os.environ.get("OLLAMA_TIMEOUT", "30")),
            max_retries=int(os.environ.get("OLLAMA_MAX_RETRIES", "3"))
        )

class LogAnalyzer:
    """Handles log analysis using Ollama models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = ollama.Client(host=config.connection_string)
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Load and validate the prompt template."""
        try:
            with open(self.config.prompt_template_path, "r") as f:
                template = f.read()
            # Validate template has required placeholder
            template.format(logs="test")
            return template
        except FileNotFoundError:
            raise RuntimeError(f"Prompt template not found: {self.config.prompt_template_path}")
        except KeyError:
            raise RuntimeError("Invalid prompt template: missing {logs} placeholder")
    
    def wait_for_server(self) -> None:
        """Wait for Ollama server to become available."""
        logger.info("Connecting to Ollama server...")
        for attempt in range(self.config.max_retries):
            try:
                self.client.ps()
                logger.info("Successfully connected to Ollama server")
                return
            except httpx.RequestError:
                logger.warning(f"Connection attempt {attempt + 1}/{self.config.max_retries} failed")
                time.sleep(2 ** attempt)  # Exponential backoff
        raise RuntimeError("Could not connect to Ollama server")
    
    def ensure_model_available(self) -> None:
        """Ensure the required model is downloaded."""
        logger.info(f"Checking for model {self.config.model_name}")
        try:
            models = self.client.list()
            existing_models = [m.get("name", "") for m in models.get("models", [])]
            
            if self.config.model_name not in existing_models:
                logger.info(f"Downloading model {self.config.model_name}")
                self.client.pull(self.config.model_name)
                logger.info("Model downloaded successfully")
            else:
                logger.info("Model already available")
        except Exception as e:
            raise RuntimeError(f"Failed to ensure model availability: {e}")
    
    def analyze_logs(self, logs: List[str]) -> str:
        """Analyze logs using the Ollama model."""
        try:
            logger.info("Preparing log analysis")
            formatted_logs = "\n".join(logs)
            prompt = self.prompt_template.format(logs=formatted_logs)
            
            logger.info("Generating analysis")
            response = self.client.generate(
                model=self.config.model_name,
                prompt=prompt,
                stream=False
            )
            
            return response['response']
        except Exception as e:
            raise RuntimeError(f"Failed to analyze logs: {e}")

def main():
    """Main entry point for the log analysis system."""
    try:
        logger.info("=== Log Analysis System Starting ===")
        
        # Initialize configuration and analyzer
        config = Config.from_env()
        analyzer = LogAnalyzer(config)
        
        # Prepare the system
        analyzer.wait_for_server()
        analyzer.ensure_model_available()
        
        # Analyze sample logs
        description = analyzer.analyze_logs(SAMPLE_LOGS)
        
        logger.info("=== Analysis Results ===")
        print(description)
        logger.info("=== Analysis Complete ===")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
