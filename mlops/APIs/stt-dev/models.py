from pydantic import BaseModel
from typing import List, Optional, Dict
from enum import Enum

class LanguageEnum(str, Enum):
    urdu = "urdu"

class NewsTypeEnum(str, Enum):
    live = "live"

class Segment(BaseModel):
    start: float
    end: float
    text: str

class TranscriptionResponse(BaseModel):
    full_text: Optional[str] = None
    timestamp: List[Segment]

class ErrorResponse(BaseModel):
    detail: str

class HealthResponse(BaseModel):
    status: str           # "healthy" / "unhealthy"
    service: str          # e.g., "Speech-to-Text API"
    triton_server: str    # "healthy" / "unhealthy" / "unreachable"
    model_status: str