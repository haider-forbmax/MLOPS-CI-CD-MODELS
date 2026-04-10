import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Always load the .env that lives next to this file.
# load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()

class Config:
    # Service endpoints
    
    FR_SERVICE_URL = os.getenv("FR_SERVICE_URL", "192.168.18.30:3080")
    MODEL_NAME = os.getenv("MODEL_NAME", "face_pipeline")

    # Milvus configuration
    DATABASE_NAME = os.getenv("DATABASE_NAME", "NIMAR_Face_Embeddings")
    MILVUS_ALIAS = os.getenv("MILVUS_ALIAS", "default")
    MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus-standalone")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "face_embeddings")
    MILVUS_VECTOR_DIM = int(os.getenv("MILVUS_VECTOR_DIM", "512"))

    # Face detection defaults
    FACE_DETECTION_CONFIDENCE = float(os.getenv("FACE_DETECTION_CONFIDENCE", "0.5"))
    FACE_DETECTION_NMS_THRESHOLD = float(os.getenv("FACE_DETECTION_NMS_THRESHOLD", "0.4"))


    # changing Prediction defaults (new)
    PRED_CONFIDENCE = float(os.getenv("PRED_CONFIDENCE", "0.75"))
    PRED_NMS_THRESHOLD = float(os.getenv("PRED_NMS_THRESHOLD", "0.6"))
    PRED_SIMILARITY_THRESHOLD = float(os.getenv("PRED_SIMILARITY_THRESHOLD", "0.52"))

    # Face recognition defaults
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    MAX_FACES_PER_IMAGE = int(os.getenv("MAX_FACES_PER_IMAGE", "10"))

    # API settings
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
    API_KEY = os.getenv("API_KEY", "face-recognition-api-key")

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        return {
            "face_detection": {
                "confidence_threshold": cls.PRED_CONFIDENCE,
                "nms_threshold": cls.PRED_NMS_THRESHOLD
            },
            "face_recognition": {
                "similarity_threshold": cls.PRED_SIMILARITY_THRESHOLD
            },
            "database": {
                "collection_name": cls.MILVUS_COLLECTION_NAME,
                "vector_dimension": cls.MILVUS_VECTOR_DIM
            },
            "services": {
                "fr_endpoint": cls.FR_SERVICE_URL,
                "fr_model_name": cls.MODEL_NAME,
                "milvus_endpoint": f"{cls.MILVUS_HOST}:{cls.MILVUS_PORT}"
            },
            #changing
            "parameters": {
                "confidence": cls.PRED_CONFIDENCE,
                "nms_threshold": cls.PRED_NMS_THRESHOLD,
                "similarity_threshold": cls.PRED_SIMILARITY_THRESHOLD
            }
        }

    @classmethod
    def update_config(cls, updates: Dict[str, Any]) -> None:
        if "face_detection" in updates:
            fd = updates["face_detection"]
            if "confidence_threshold" in fd:
                cls.PRED_CONFIDENCE = fd["confidence_threshold"]
            if "nms_threshold" in fd:
                cls.PRED_NMS_THRESHOLD = fd["nms_threshold"]

        if "face_recognition" in updates:
            fr = updates["face_recognition"]
            if "similarity_threshold" in fr:
                cls.PRED_SIMILARITY_THRESHOLD = fr["similarity_threshold"]

config = Config()
