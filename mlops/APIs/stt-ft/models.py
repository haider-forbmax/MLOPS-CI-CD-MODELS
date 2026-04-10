from pydantic import BaseModel
from typing import List, Optional, Dict
from enum import Enum

class LanguageEnum(str, Enum):
    urdu = "urdu"

class NewsTypeEnum(str, Enum):
    live = "live"

from typing import Optional

class Segment(BaseModel):
    start_time: float
    end_time: float
    text: Optional[str] = None

class TranscriptionResponse(BaseModel):
    full_text: Optional[str] = None
    timestamp: list[Segment]



class ErrorResponse(BaseModel):
    detail: str

class HealthResponse(BaseModel):
    status: str           # "healthy" / "unhealthy"
    service: str          # e.g., "Speech-to-Text API"
    model_status: dict