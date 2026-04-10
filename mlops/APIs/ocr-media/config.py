# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Configuration
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "680"))
    TRITON_TIMEOUT = int(os.getenv("TRITON_TIMEOUT", "1200"))  # Increased for long audio files

    # Model inference URLs and API
    # OlmOCR model endpoint
    MODEL_URL = os.getenv("MODEL_URL", "https://ocr-ft-dgx.nimar.gov.pk")
    API_KEY = os.getenv("API_KEY", "123")
    
    # Model configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "OlmOCR")

    # Upload folder
    SUPPORTED_FORMATS = ["base64"]
    TASK_TYPE = "OCR"

    API_PORT = int(os.getenv("API_PORT", "8000"))

    # User API Keys for authentication
    USER_API_KEYS = dict(
        item.split(":", 1)
        for item in os.getenv("USER_API_KEYS", "").split(",")
        if ":" in item
    )

    # Logging configuration
    LOG_FILE = os.getenv("LOG_FILE", "upload.log")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
