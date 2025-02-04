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
    "2024-01-20 10:15:25 WARNING High memory usage detected",
    "2024-01-20 10:15:26 INFO User login successful: user123",
    "2024-01-20 10:15:27 INFO Database connection restored",
    "2024-01-20 10:15:28 INFO Cache cleared successfully",
    "2024-01-20 10:15:29 WARNING High CPU usage: 85%",
    "2024-01-20 10:15:30 INFO User logout: user123",
    "2024-01-20 10:15:31 ERROR Failed to process payment",
    "2024-01-20 10:15:32 INFO Backup started",
    "2024-01-20 10:15:33 INFO Backup completed",
    "2024-01-20 10:15:34 WARNING High memory usage detected",
    "2024-01-20 10:15:35 INFO User login successful: user456",
    "2024-01-20 10:15:36 ERROR Database connection failed",
    "2024-01-20 10:15:37 INFO Database connection restored",
    "2024-01-20 10:15:38 INFO File upload successful: doc.pdf",
    "2024-01-20 10:15:39 WARNING System running low on disk space",
    "2024-01-20 10:15:40 INFO Scheduled maintenance started",
    "2024-01-20 10:15:41 INFO Cache update completed",
    "2024-01-20 10:15:42 ERROR Failed to send email",
    "2024-01-20 10:15:43 INFO User logout: user456",
    "2024-01-20 10:15:44 INFO Daily health check passed",
    "2024-01-20 10:15:45 WARNING High CPU usage: 90%"
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

@dataclass
class LogStats:
    """Statistics about analyzed logs."""
    total_logs: int = 0
    errors: int = 0
    warnings: int = 0
    infos: int = 0
    unique_components: set = None
    
    def __post_init__(self):
        self.unique_components = set()
    
    def to_dict(self):
        return {
            "total_logs": self.total_logs,
            "errors": self.errors,
            "warnings": self.warnings,
            "infos": self.infos,
            "unique_components": list(self.unique_components)
        }

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
    
    def _analyze_statistics(self, logs: List[str]) -> LogStats:
        """Gather statistics about the logs."""
        stats = LogStats()
        stats.total_logs = len(logs)
        
        for log in logs:
            stats.total_logs += 1
            if "ERROR" in log:
                stats.errors += 1
            elif "WARNING" in log:
                stats.warnings += 1
            elif "INFO" in log:
                stats.infos += 1
            
            # Try to extract component name from bracketed text
            if "[" in log and "]" in log:
                component = log[log.find("[")+1:log.find("]")]
                stats.unique_components.add(component)
        
        return stats
    
    def analyze_logs(self, logs: List[str]) -> str:
        """Analyze logs using the Ollama model."""
        try:
            logger.info("Analyzing log statistics...")
            stats = self._analyze_statistics(logs)
            
            logger.info("Preparing log analysis with statistics...")
            formatted_logs = "\n".join(logs)
            
            # Add statistics to the prompt
            stats_summary = f"\nLog Statistics:\n" \
                          f"Total Logs: {stats.total_logs}\n" \
                          f"Errors: {stats.errors}\n" \
                          f"Warnings: {stats.warnings}\n" \
                          f"Info: {stats.infos}\n"
            
            prompt = self.prompt_template.format(
                logs=formatted_logs + "\n" + stats_summary
            )
            
            logger.info("Generating analysis summary...")
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
