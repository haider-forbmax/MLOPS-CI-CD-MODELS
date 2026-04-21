
import logging
import time
import uvicorn
from config import Config
from models import TranscriptionResponse, HealthResponse
from utils import process_video, validate_api_key
from inference import get_client
from fastapi import FastAPI, UploadFile, Header, HTTPException

# FastAPI app with better documentation
app = FastAPI(
    title="Speech-to-Text API",
    description="STT API supporting Urdu transcription using Whisper",
    version="1.0.0"
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


@app.post("/v1/transcribe", response_model=TranscriptionResponse)
def transcribe_video_endpoint(
    video_file: UploadFile,
    authorization: str = Header(None)):

    """
    Transcribe video or audio file to text
    
    Args:
        video_file: Video or audio file to transcribe
        news_type: Type of news (live, on_demand)
        language: Language code (ur for Urdu)
    
    Returns:
        TranscriptionResponse with full text and timestamps
    """

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
        if not validate_api_key(api_key):
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




    logger.info(f"Received request for transcribe_video with file: {video_file.filename}")

    start_time = time.time()
    # result = process_video(video_file, news_type, language)
    result = process_video(video_file)
    end_time = time.time()
    
    logger.info(f"Processed file: {video_file.filename} in {end_time - start_time:.2f} seconds")
    return result


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with Whisper API server status"""
    logger.debug("Health check requested")
    try:
        # Get client and check health
        client = get_client()
        api_healthy = client.health_check()
        
        api_status = "healthy" if api_healthy else "unhealthy"
        model_status = "healthy" if api_healthy else "unhealthy"
        overall_status = "healthy" if api_healthy else "unhealthy"

        response = HealthResponse(
            status=overall_status,
            service="Speech-to-Text API",
            triton_server=api_status,  # Keeping field name for backward compatibility
            model_status=model_status
        )

        if overall_status == "unhealthy":
            logger.warning("Health check failed - service unhealthy")
            raise HTTPException(status_code=503, detail=response.model_dump())

        logger.debug("Health check passed - service healthy")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check error: {e}")
        response = HealthResponse(
            status="unhealthy",
            service="Speech-to-Text API",
            triton_server="unreachable",
            model_status="unreachable"
        )
        raise HTTPException(status_code=503, detail=response.model_dump())


@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "api_url": Config.MODEL_URL,
        "model_name": Config.MODEL_NAME,
        "temperature": Config.TEMPERATURE,
        "response_format": Config.RESPONSE_FORMAT,
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
        "version": "1.0.0",
        "description": "Speech-to-Text API using Whisper via REST API",
        "endpoints": {
            "transcribe": "/v1/transcribe",
            "health": "/health",
            "config": "/config",
            "docs": "/docs"
        },
        "uploads_dir": Config.UPLOADS_DIR,
        "supported_formats": {
            "video": Config.VIDEO_EXTENSIONS,
            "audio": Config.AUDIO_EXTENSIONS
        },
        "model": {
            "name": Config.MODEL_NAME,
            "api_url": Config.MODEL_URL
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2022)