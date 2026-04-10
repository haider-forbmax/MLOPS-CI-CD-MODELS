import os
import logging
from fastapi import FastAPI, UploadFile, HTTPException, Header, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from emotion_inference.utils import classify_emotion_triton, triton_client, feature_extractor
from emotion_inference.config import Config, HealthResponse
import tritonclient.http as httpclient
from datetime import datetime
import tempfile
import traceback
import shutil
import subprocess


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Emotion Inference API", version="1.0")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Standardize HTTPException responses to always return 
    {"message": "user-friendly message", "error": "technical error"}
    """
    # If detail is already a dict with message and error, use it
    if isinstance(exc.detail, dict) and "message" in exc.detail and "error" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    
    # Otherwise, create standardized format
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": "An error occurred while processing your request",
            "error": str(exc.detail)
        }
    )
# Global exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with standardized format"""
    return JSONResponse(
        status_code=422,
        content={
            "message": "Invalid request data provided",
            "error": str(exc.errors())
        }
    )
# Global exception handler for unexpected errors
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "message": "An internal server error occurred. Please try again later",
            "error": str(exc)
        }
    )

def verify_api_key(authorization: str = Header(None)):
    """Verify Bearer token authorization"""
    if not authorization:
        logger.warning("Missing Authorization header")
        raise HTTPException(
            status_code=401,
            detail={"message":"Authorization header is required to access this endpoint","error":"Missing Authorization header"}
            )

    try:
        auth_parts = authorization.split()
        if len(auth_parts) != 2 or auth_parts[0].lower() != "bearer":
            logger.warning("Invalid Authorization header format")
            raise HTTPException(
                status_code=401,
                detail={"message":"Authorization must be in format: Bearer <api_key>","error":"Invalid Authorization header format"}
            )

        api_key = auth_parts[1]
        if api_key != Config.API_KEY:
            logger.warning("Invalid API key provided")
            raise HTTPException(
                status_code=401,
                detail={"message":"The API key provided is not valid","error":"Invalid API key"}
            )

        return api_key
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error("Error validating API key: %s", e)
        raise HTTPException(
            status_code=401,
            detail={"message":"Unable to validate authorization","error":f"{str(e)}"}
        )


def extract_audio(video_path: str, audio_path: str):
    """
    Extract audio from any video file (including .ts) to WAV using ffmpeg
    """
    command = [
        "ffmpeg",
        "-y",  # overwrite output
        "-i", video_path,
        "-vn",  # no video
        "-acodec", "pcm_s16le",  # WAV format
        "-ar", "16000",  # 16 kHz
        "-ac", "1",  # mono
        audio_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return audio_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr.decode()}")
        raise HTTPException(
            status_code=400,
            detail={
                "message": "The provided file does not contain valid audio",
                "error": f"FFmpeg extraction failed: {e.stderr.decode()}"
            }
        )

@app.post("/v1/voice_compassion")
async def voice_compassion_endpoint(video_file: UploadFile,_: str = Depends(verify_api_key)):
    """
    Process uploaded audio or video file and detect emotions in the audio.
    For video files, audio will be extracted automatically.
    """
    logger.info(f"Received request for Voice Compassion with file: {video_file.filename}")
    temp_filename = None
    temp_dir = tempfile.mkdtemp(prefix="voice_compassion_")
    try:
        # Check if Triton client is available
        if triton_client is None or feature_extractor is None:
            logger.warning(f"Triton inference is down")
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Emotion recognition service is temporarily unavailable. Please try again later",
                    "error": "Triton inference server or feature extractor not initialized"
                }
            )
        
        # Read the uploaded video file into memory
        file_content = video_file.file.read()
        file_name, file_extension = os.path.splitext(video_file.filename)
        
        if not any(file_extension.lower() == ext for ext in Config.SUPPORTED_FORMATS):
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Unsupported file format. Supported formats: {', '.join(Config.SUPPORTED_FORMATS)}",
                    "error": f"File extension {file_extension} not supported"
                }
            )
        # Save the uploaded file in the temporary directory
        file_path = os.path.join(temp_dir, video_file.filename)
        with open(file_path, "wb") as temp_file:
            temp_file.write(file_content)

        # Check if the file has a video extension
        
        if any(file_extension.lower() == ext for ext in Config.VIDEO_FORMATS):
            # Save the audio clip in the temporary directory
            audio_file_path = os.path.join(temp_dir, f"{file_name}.wav")
            
            audio_file_path = extract_audio(file_path,audio_file_path)
            
        else:
            audio_file_path = file_path
        
        try:
            label, confidence = classify_emotion_triton(audio_file_path)
        except ValueError as e:
            # Handle the ValueError from utils.py
            logger.error(f"Emotion classification failed: {e}")
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Unable to detect emotion in the provided audio",
                    "error": str(e)
                }
            )
        except Exception as e:
            # Handle any other unexpected errors from classification
            logger.error(f"Unexpected error during emotion classification: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "An error occurred while processing the audio",
                    "error": str(e)
                }
            )
        return {"emotion": label}
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch any unexpected errors not already handled
        logger.error(f"Unexpected error in voice_compassion_endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "An internal server error occurred. Please try again later",
                "error": str(e)
            }
        )
    finally:
        # Always clean up the temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with Triton server and model status"""
    logger.debug("Health check requested")
    try:
        with httpclient.InferenceServerClient(url=Config.TRITON_URL) as triton_client:
            server_live = triton_client.is_server_live()
            server_ready = triton_client.is_server_ready()
            model_ready = triton_client.is_model_ready(Config.MODEL_NAME)

        triton_status = "healthy" if server_live and server_ready else "unhealthy"
        model_status_str = "healthy" if model_ready else "unhealthy"
        overall_status = "healthy" if triton_status == "healthy" and model_ready else "unhealthy"

        response = HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
        )

        if overall_status == "unhealthy":
            logger.warning("Health check failed - service unhealthy")
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Service is unhealthy",
                    "error": f"Triton status: {triton_status}, Model ready: {model_ready}"
                }
            )

        logger.debug("Health check passed - service healthy")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check error: {e}")
        traceback.print_exc() 
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Unable to perform health check",
                "error": str(e)
            }
        )

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "triton_server_url": Config.TRITON_URL,
        "model_name": Config.MODEL_NAME,
        "supported_extensions": Config.SUPPORTED_FORMATS
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Voice Compassion API",
        "version": "1.0.0",
        "description": "Voice Compassion API using Triton Inference Server",
        "endpoints": {
            "voice_compassion": "/v1/voice-compassion/",
            "health": "/health",
            "config": "/config",
            "docs": "/docs"
        },
        "supported_formats": {
            "video": Config.VIDEO_FORMATS,
            "audio": Config.AUDIO_FORMATS,
        }
    }
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("emotion_inference.app:app", host="0.0.0.0", port=2000)
