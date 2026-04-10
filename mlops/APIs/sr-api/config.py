import os
## from typing import Dict, Any  # unused
from pyannote.audio import Audio
from dotenv import load_dotenv
from pydantic import BaseModel
# Load environment variables from .env file
load_dotenv()

class Config:
    # Service endpoints
    # API Configuration
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    TRITON_TIMEOUT = int(os.getenv("TRITON_TIMEOUT", "10"))
    MILVUS_ALIAS = os.getenv("MILVUS_ALIAS", "default")
    MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "SR_Database")
    # Model inference URLs
    # Model names
    # MODEL_URL = os.getenv("URDU_MODEL_URL", "dev-sr-dgx.nimar.gov.pk")
    MODEL_URL = os.getenv("SR_MODEL_URL", "dev-sr-dgx.nimar.gov.pk") # 192.168.18.22:31635 
    # MODEL_URL = os.getenv("SR_MODEL_URL", "192.168.18.22:31635") #  
    MODEL_NAME = os.getenv("SR_MODEL_NAME", "ecapa_speaker_verification")
    # API Keys
    VIDEO_EXTENSION = [".mp4", ".avi", ".mov", ".wmv", ".mkv", ".flv", ".ts"]
    AUDIO_EXTENSIONS = [".wav", ".mp3", ".aac", ".ogg", ".flac"]
    # Milvus configuration
    # MILVUS_HOST = os.getenv("MILVUS_HOST", "192.168.18.22")
    # MILVUS_PORT = int(os.getenv("MILVUS_PORT", "31036"))
    MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus-standalone.fr-milvus-database.svc.cluster.local")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "speaker_recognition")
    MILVUS_VECTOR_DIM = int(os.getenv("MILVUS_VECTOR_DIM", "192"))

    AUDIO = Audio(sample_rate=16000, mono="downmix")

    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

    API_KEY = "123"
    # LOG_FILE = os.getenv("LOG_FILE", "upload.log")
    # LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()






class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"
