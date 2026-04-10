# main.py
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from config import Config
from sentiment_service import analyze_sentiment
import requests
import logging
# from models import TrainSentimentRequest, TrainSentimentResponse
from models import SentimentRequest, SentimentResponse, HealthResponse

config = Config()

app = FastAPI(
    title=config.SERVICE_NAME,
    description=f"{config.SERVICE_NAME} using {config.MODEL_NAME} model",
    version=config.SERVICE_VERSION,
)



log_level = getattr(logging, Config.LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# API Key Dependency
# ----------------------------
def verify_api_key(x_api_key: str = Header(...)):
    """
    Simple header-based API key verification.
    """
    if x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True


# ----------------------------
# Root Endpoint
# ----------------------------
@app.get("/")
def root():
    return {
        "message": f"Welcome to {config.SERVICE_NAME}",
        "docs": "/docs",
        "health": "/health",
        "sentiment": "/v1/sentiment",
        "model": config.MODEL_NAME,
        "base_url": config.BASE_URL,
    }


# ----------------------------
# Health Endpoint
# ----------------------------
@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Pings the GPT-OSS model list endpoint to verify health.
    """
    try:
        resp = requests.get(config.HEALTH_ENDPOINT, timeout=5)
        resp.raise_for_status()
        models = [m["id"] for m in resp.json().get("data", [])]
        return HealthResponse(
            status="healthy",
            available_models=models,
            checked_url=config.HEALTH_ENDPOINT,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")


# ----------------------------
# Sentiment Analysis Endpoint
# ----------------------------
@app.post("/v1/sentiment", response_model=SentimentResponse)
def analyze_sentiment_api(payload: SentimentRequest,  authorization: str = Header(None)):
    """
    Accepts text input and returns sentiment classification.
    """
        # Validate API key
    if not authorization:
        logger.warning("Missing Authorization header")
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )
    
    # Extract API key from Authorization header (expecting "Bearer <api_key>")
    try:
        auth_parts = authorization.split()
        if len(auth_parts) != 2 or auth_parts[0].lower() != "bearer":
            logger.warning("Invalid Authorization header format")
            raise HTTPException(
                status_code=401,
                detail="Invalid Authorization header format. Expected: Bearer <api_key>"
            )
        
        api_key = auth_parts[1]
        if not verify_api_key(api_key):
            logger.warning("Invalid API key provided")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header"
        )
    

    text = payload.text.strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Input text is empty. Please provide a non-empty 'text' field."
        )

    try:
        result = analyze_sentiment(text)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

# @app.post("/v1/train_sentiment", response_model=TrainSentimentResponse)
# def train_sentiment(payload: TrainSentimentRequest,
#                     authorization: str = Header(None)):

#     if not authorization:
#         raise HTTPException(status_code=401, detail="Missing Authorization header")

#     try:
#         auth_parts = authorization.split()
#         if len(auth_parts) != 2 or auth_parts[0].lower() != "bearer":
#             raise HTTPException(status_code=401, detail="Invalid Authorization header format")

#         api_key = auth_parts[1]
#         if api_key != config.API_KEY:
#             raise HTTPException(status_code=401, detail="Invalid API key")

#     except HTTPException:
#         raise

#     if not payload.Classes:
#         raise HTTPException(status_code=400, detail="Classes list cannot be empty")

#     if not payload.Examples:
#         raise HTTPException(status_code=400, detail="Examples cannot be empty")

#     try:
#         train_configuration(payload)

#         return TrainSentimentResponse(
#             status="success"
#         )

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Training failed: {e}")
# # ----------------------------
# # Run App
# # ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8300, reload=False)
