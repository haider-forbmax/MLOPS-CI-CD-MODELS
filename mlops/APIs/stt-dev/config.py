# config.py
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, os.getenv("UPLOADS_DIR", "uploads"))
os.makedirs(UPLOADS_DIR, exist_ok=True)

class Config:
    # API Configuration
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    TRITON_TIMEOUT = int(os.getenv("TRITON_TIMEOUT", "300"))  # Increased for long audio files

    # Model inference URLs and API
    # Updated to use new REST API endpoint
    # MODEL_URL = os.getenv("MODEL_URL", "https://stt-dgx.nimar.gov.pk/")
    MODEL_URL = os.getenv("MODEL_URL", "vllm-whisper-service.models-inference.svc.cluster.local:8000")
    API_KEY = os.getenv("API_KEY", "123")
    
    # Model configuration for new Whisper API
    MODEL_NAME = os.getenv("MODEL_NAME", "whisper-large-v3")
    RESPONSE_FORMAT = os.getenv("RESPONSE_FORMAT", "json")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

    # Upload folder
    UPLOADS_DIR = os.getenv("UPLOADS_DIR", "./uploads")

    # Supported video formats (from .env)
    VIDEO_EXTENSIONS = os.getenv("VIDEO_EXTENSIONS", ".mp4,.avi,.mov,.wmv,.mkv,.flv").split(",")

    # Supported audio formats (keep as is or also from .env if needed)
    AUDIO_EXTENSIONS = [".wav", ".mp3", ".aac", ".ogg", ".flac"]

    # User API Keys for authentication
    USER_API_KEYS = dict(
        item.split(":", 1)
        for item in os.getenv("USER_API_KEYS", "").split(",")
        if ":" in item
    )

    # Legacy model I/O names (kept for backward compatibility, but not used with new API)
    MODEL_INPUT_NAME = os.getenv("MODEL_INPUT_NAME", "input__0")
    MODEL_OUTPUT_NAME = os.getenv("MODEL_OUTPUT_NAME", "output__0")

    # Logging configuration
    LOG_FILE = os.getenv("LOG_FILE", "upload.log")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    # Language and news type support
    SUPPORTED_LANGUAGES = os.getenv("SUPPORTED_LANGUAGES", "english").split(",")

    SUPPORTED_NEWS_TYPES = os.getenv("SUPPORTED_NEWS_TYPES", "live,on_demand").split(",")