# sentiment_service.py
from openai import OpenAI
from pydantic import BaseModel, Field
from fastapi import HTTPException
from config import Config
from prompts import DYNAMIC_SENTIMENT_PROMPT
from models import SentimentResponse

config = Config()

# Initialize OpenAI client with GPT-OSS endpoint and key
client = OpenAI(
    base_url=config.BASE_URL,
    api_key=config.LLM_API_KEY
)


def analyze_sentiment(content: str) -> SentimentResponse:
    """
    Analyzes sentiment using GPT-OSS model.
    Returns a structured SentimentResponse response.
    """
    if not content or not content.strip():
        raise HTTPException(
            status_code=400,
            detail="Input text is empty. Please provide content for sentiment analysis."
        )

    try:
        response = client.responses.parse(
            model=config.MODEL_NAME,
            input=[
                {"role": "system", "content": DYNAMIC_SENTIMENT_PROMPT},
                {"role": "user", "content": f"Analyze the sentiment of this text:\n\n{content}"}
            ],
            text_format=SentimentResponse,
        )

        return response.output_parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")
