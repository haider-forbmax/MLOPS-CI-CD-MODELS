# models.py
from pydantic import BaseModel, Field
from typing import List
from config import Config

class SentimentRequest(BaseModel):
    text: str = Field(..., description="The input text for sentiment analysis.")


class SentimentResponse(BaseModel):
    sentiment: str = Field(description=f"{Config.CLASSES}")
    confidence: str = Field(description="High, Medium, or Low")
    reasoning: str = Field(description="Explanation of why this sentiment was chosen")


class HealthResponse(BaseModel):
    status: str
    available_models: List[str]
    checked_url: str




