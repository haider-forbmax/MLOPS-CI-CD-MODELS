import os
import json
import logging
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
class Config:
    # Triton Inference Server Configuration
    TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "192.168.18.80:30483")
    TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "resnet100")

    # Model Input/Output Names
    MODEL_INPUT_NAME = os.getenv("MODEL_INPUT_NAME", "input")
    MODEL_OUTPUT_NAME = os.getenv("MODEL_OUTPUT_NAME", "output")

    # Model Dimensions
    MODEL_INPUT_DIMS = [int(x) for x in os.getenv("MODEL_INPUT_DIMS", "3,122,122").split(",")]
    MODEL_OUTPUT_DIMS = [int(x) for x in os.getenv("MODEL_OUTPUT_DIMS", "512").split(",")]

    # Image Processing Limits
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10MB in bytes
    SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS", "jpeg,png,jpg,webp").split(",")

    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "9002"))

    # Timeouts (in seconds)
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    TRITON_TIMEOUT = int(os.getenv("TRITON_TIMEOUT", "10"))

    # Task Configuration
    TASK_TYPE = os.getenv("TASK_TYPE", "embedding")

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")

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