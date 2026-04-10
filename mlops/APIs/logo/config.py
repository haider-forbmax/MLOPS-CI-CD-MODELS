import os
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "192.168.18.48:8000")
    TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "onnx")
    # Model input/output configuration
    MODEL_INPUT_NAME = os.getenv("MODEL_INPUT_NAME", "images")
    MODEL_OUTPUT_NAME = os.getenv("MODEL_OUTPUT_NAME", "output0")
    
    # Default processing parameters
    DEFAULT_INPUT_SIZE = json.loads(os.getenv("DEFAULT_INPUT_SIZE", "[640, 640]"))
    DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.25"))
    DEFAULT_NMS_THRESHOLD = float(os.getenv("DEFAULT_NMS_THRESHOLD", "0.45"))
    
    # Class mapping
    CLASSES = json.loads(os.getenv("CLASSES", '{"0": "object"}'))
    
    # Legacy support
    INPUT_WIDTH = DEFAULT_INPUT_SIZE[0]
    INPUT_HEIGHT = DEFAULT_INPUT_SIZE[1]
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10MB
    SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS", "jpeg,png,jpg,webp").split(",")
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "9002"))
    
    # Timeouts
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    TRITON_TIMEOUT = int(os.getenv("TRITON_TIMEOUT", "10"))
    
    # YOLO11 Specific
    TASK_TYPE = os.getenv("TASK_TYPE", "detect")
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # API Security
    API_KEY = os.getenv("API_KEY", "123")
    
    @classmethod
    def setup_logging(cls):
        """Setup logging configuration based on environment variables"""
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
            ]
        )
        return logging.getLogger(__name__)