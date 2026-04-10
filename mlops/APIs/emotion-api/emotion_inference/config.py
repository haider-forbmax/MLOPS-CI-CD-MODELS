import os

from dotenv import load_dotenv
from pydantic import BaseModel


load_dotenv()


def _get_list_env(key: str, default: list[str]) -> list[str]:
    value = os.getenv(key)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


class Config:
    AUDIO_FORMATS = _get_list_env(
        "AUDIO_FORMATS",
        [".wav", ".mp3", ".flac", ".aac", ".ogg", ".opus", ".m4a", ".aiff", ".amr", ".wma"],
    )
    VIDEO_FORMATS = _get_list_env(
        "VIDEO_FORMATS",
        [".mp4", ".mkv", ".webm", ".avi", ".mov", ".mpg", ".mpeg", ".flv", ".3gp", ".ts", ".f4v"],
    )
    SUPPORTED_FORMATS = _get_list_env("SUPPORTED_FORMATS", AUDIO_FORMATS + VIDEO_FORMATS)

    TRITON_URL = os.getenv(
        "TRITON_URL",
        "triton-speech-emotion-service.models-inference.svc.cluster.local:8000",
    )
    MODEL_NAME = os.getenv("MODEL_NAME", "speech_emotion_recognition")
    MODEL_ID = os.getenv("MODEL_ID", "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")
    API_KEY = os.getenv("API_KEY", "123")
    VERSION = os.getenv("APP_VERSION", "1.0.0")


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = Config.VERSION
