import time
import uvicorn
import logging
from config import Config
from fastapi import FastAPI, UploadFile, Header, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from models import TranscriptionResponse, HealthResponse
from utils import process_video, validate_api_key
from inference import get_client

security = HTTPBearer()

# FastAPI app
app = FastAPI(
    title="Speech-to-Text API",
    description="STT API supporting Urdu transcription using vLLM Whisper",
    version="2.0.0"
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


@app.post("/v1/transcribe", response_model=TranscriptionResponse, response_model_exclude_none=True)
def transcribe_video_endpoint(
    video_file: UploadFile,  # Changed from video_file to file
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Transcribe video or audio file to text
    
    Args:
        file: Video or audio file to transcribe
        authorization: Bearer token
    
    Returns:
        TranscriptionResponse with full text and timestamps
    """
    authorization = credentials.credentials
    if not authorization:
        logger.warning("Missing Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    # Validate API key
    try:
        
        api_key = authorization
        if not validate_api_key(api_key):
            logger.warning("Invalid API key provided")
            raise HTTPException(status_code=401, detail="Invalid API key")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    logger.info(f"Received transcription request for: {video_file.filename}")

    start_time = time.time()
    result = process_video(video_file)
    elapsed = time.time() - start_time
    
    logger.info(f"Processed {video_file.filename} in {elapsed:.2f}s")
    return result


@app.get("/health", response_model=HealthResponse)
async def health_check():
    logger.debug("Health check requested")

    try:
        client = get_client()
        model_status = client.health_check()
        overall_status = (
            "healthy" if all(model_status.values()) else "unhealthy"
        )
        

        response = HealthResponse(
            status=overall_status,
            service="Speech-to-Text API",
            model_status=model_status
        )

        if overall_status == "unhealthy":
            logger.warning("Health check failed")
            raise HTTPException(status_code=503, detail=response.model_dump())

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(
            status_code=503,
            detail=HealthResponse(
                status="unhealthy",
                service="Speech-to-Text API",
                model_status="unreachable"
            ).model_dump()
        )


@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "api_url": Config.MODEL_URL,
        "response_format": Config.RESPONSE_FORMAT,
        "temperature": Config.TEMPERATURE,
        "language": Config.LANGUAGE,
        "uploads_dir": Config.UPLOADS_DIR,
        "video_extensions": Config.VIDEO_EXTENSIONS,
        "audio_extensions": Config.AUDIO_EXTENSIONS,
        "supported_languages": Config.SUPPORTED_LANGUAGES
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Speech-to-Text API",
        "version": "2.0.0",
        "description": "Speech-to-Text API using vLLM Whisper",
        "endpoints": {
            "transcribe": "/v1/transcribe",
            "health": "/health",
            "config": "/config",
            "docs": "/docs"
        },
        "supported_formats": {
            "video": Config.VIDEO_EXTENSIONS,
            "audio": Config.AUDIO_EXTENSIONS
        },
        "api": {
            "url": Config.MODEL_URL,
            "response_format": Config.RESPONSE_FORMAT
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2022)