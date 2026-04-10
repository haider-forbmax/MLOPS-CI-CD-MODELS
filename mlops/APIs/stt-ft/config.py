# config.py
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    # API Configuration
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
    Supported_Language = ["en","ur","hi"]
    # vLLM Whisper API (OpenAI-compatible)
    MODEL_URL = os.getenv("MODEL_URL", "https://docling-dgx.nimar.gov.pk/v1/audio/transcriptions")
    API_KEY = os.getenv("API_KEY", "123")
    ENGLISH_MODEL_URL = os.getenv("ENGLISH_MODEL_URL", "http://192.168.18.30:8071/v1/audio/transcriptions")
    URDU_MODEL_URL = os.getenv("URDU_MODEL_URL", "http://192.168.18.30:8070/v1/audio/transcriptions")
    LANGUAGE_DETECTOR_URL = os.getenv("LANGUAGE_DETECTOR_URL", "http://192.168.18.164:2021/detect")
    # Model configuration
    ENGLISH_MODEL_NAME = os.getenv("MODEL_NAME", "whisper-distil-large-v3.5")
    URDU_MODEL_NAME = os.getenv("MODEL_NAME", "whisper-large-v3-ft-urdu")
    MODEL_NAME = os.getenv("MODEL_NAME", "whisper-large-v3")
    RESPONSE_FORMAT = os.getenv("RESPONSE_FORMAT", "verbose_json")  # json, text, srt, verbose_json, vtt
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
    LANGUAGE = os.getenv("LANGUAGE", "en")  # Urdu

    # Upload folder - FIX: Use absolute path
    UPLOADS_DIR = os.path.join(BASE_DIR, os.getenv("UPLOADS_DIR", "uploads"))

    SSL_VERIFY = os.getenv("SSL_VERIFY", "true").lower() == "true"


    # Supported formats
    VIDEO_EXTENSIONS = os.getenv("VIDEO_EXTENSIONS", ".mp4,.avi,.mov,.wmv,.mkv,.flv").split(",")
    AUDIO_EXTENSIONS = [".wav", ".mp3", ".aac", ".ogg", ".flac", ".m4a",".webm"]

    # User API Keys for authentication
    USER_API_KEYS = dict(
        item.split(":", 1)
        for item in os.getenv("USER_API_KEYS", "").split(",")
        if ":" in item
    )

    # Logging
    LOG_FILE = os.getenv("LOG_FILE", "upload.log")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    # Supported options
    SUPPORTED_LANGUAGES = os.getenv("SUPPORTED_LANGUAGES", "urdu,english").split(",")
    SUPPORTED_NEWS_TYPES = os.getenv("SUPPORTED_NEWS_TYPES", "live,on_demand").split(",")


# Create uploads directory if it doesn't exist
os.makedirs(Config.UPLOADS_DIR, exist_ok=True)