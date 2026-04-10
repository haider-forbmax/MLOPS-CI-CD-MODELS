from pydantic import BaseModel, Field
from typing import Optional
from config import Config


# Translation Models
class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate (English or Urdu)", min_length=1)

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, how are you?"
            }
        }


class TranslationResponse(BaseModel):
    original_text: str = Field(..., description="Original input text")
    translated_text: str = Field(..., description="Translated text")
    detected_language: str = Field(..., description="Detected source language (English or Urdu)")
    target_language: str = Field(..., description="Target language")
    
    class Config:
        json_schema_extra = {
            "example": {
                "original_text": "Hello, how are you?",
                "translated_text": "ہیلو، آپ کیسے ہیں؟",
                "detected_language": "English",
                "target_language": "Urdu"
            }
        }


# Summarization Models
class SummarizationRequest(BaseModel):
    text: str = Field(..., description="Text to summarize (English or Urdu)", min_length=20)
    #max_words: Optional[int] = Field(default=150, description="Maximum words in summary", ge=50, le=500)

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Pakistan's exports grew in 2025 due to stronger textile demand and improved logistics. However, inflation and energy prices remained major concerns for households and small businesses. Analysts expect growth to continue if fiscal discipline and investment reforms are sustained."
                # "max_words": 150
            }
        }


class SummarizationResponse(BaseModel):
    original_text: str = Field(..., description="Original input text")
    summary: str = Field(..., description="Generated summary in English")
    # detected_language: str = Field(..., description="Detected language of input")
    # word_count: int = Field(..., description="Word count of the summary")

    class Config:
        json_schema_extra = {
            "example": {
                "original_text": "Long article text...",
                "summary": "Brief summary of the article..."
                # "detected_language": "English",
                # "word_count": 145
            }
        }


# Error Response Model
class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


# Health Check Model
class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    available_models: list = Field(..., description="List of available models")
    checked_url: str = Field(..., description="URL that was checked")
    version: str = Field(default=Config().API_VERSION, description="API version")  # Make version optional with default


class TokenValidationRequest(BaseModel):
    text: str = Field(..., description="Text to validate for token count")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a sample text you want to check token count for."
            }
        }


class TokenValidationResponse(BaseModel):
    token_count: int = Field(..., description="Total token count of the input text")
    INPUT_TOKENS_LIMIT: int = Field(..., description="Maximum allowed token limit")
    within_limit: bool = Field(..., description="True if token count is within limit")
    truncated_preview: Optional[str] = Field(None, description="Truncated preview text (if over limit)")
