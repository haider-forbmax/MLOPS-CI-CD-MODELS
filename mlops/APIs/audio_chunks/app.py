from fastapi import FastAPI, UploadFile, Depends, HTTPException, Header, File, Request
from fastapi.exceptions import RequestValidationError
from utils import *
import traceback
import logging
from config import Config 
import tempfile
from fastapi.responses import JSONResponse
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Audio Chunking Service",
    version="1.0.0",
    description="Extract and chunk audio from video/audio files based on silence detection")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed logging"""
    logger.error(f"Validation Error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "message": "Invalid request format. Ensure file is uploaded as form data.",
            "error": "Validation failed"
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancers"""
    return {
        "status": "healthy",
        "service": "Audio Chunking Service",
        "version": "1.0.0"
    }


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
    

@app.post("/get_audio_chunks")
async def get_audio_chunks(video_file: UploadFile = File(...), _: str = Depends(verify_api_key)):
    logger.info(f"Processing: {video_file.filename} (Content-Type: {video_file.content_type})")
    
    tmp_path = None
    audio_path = None
    
    try:
        # Validate file was actually uploaded
        if not video_file or not video_file.filename:
            return JSONResponse(
                status_code=400,
                content={
                    "message": "No file uploaded",
                    "error": "File parameter is required"
                }
            )
        
        suffix = os.path.splitext(video_file.filename)[1].lower()
        
        # Validate file extension
        if suffix not in Config.VIDEO_EXTENSION and suffix not in ['.mp3', '.wav', '.flac', '.aac', '.m4a']:
            return JSONResponse(
                status_code=400,
                content={
                    "message": "Unsupported file format",
                    "error": f"Supported formats: video (mp4, mov, avi) or audio (mp3, wav, flac, aac, m4a)"
                }
            )
        
        is_video = suffix in Config.VIDEO_EXTENSION
        
        # Write uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await video_file.read()
            if not content:
                return JSONResponse(
                    status_code=400,
                    content={
                        "message": "Empty file uploaded",
                        "error": "File cannot be empty"
                    }
                )
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info(f"File saved to {tmp_path} ({os.path.getsize(tmp_path)} bytes)")

        # Extract audio if needed
        if is_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_tmp:
                audio_path = audio_tmp.name
            try:
                extract_audio(tmp_path, audio_path)
            except RuntimeError as e:
                logger.error(f"Audio extraction failed: {str(e)}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "message": "The uploaded video does not contain an audio stream",
                        "error": str(e)
                    }  
                )
            file_to_process = audio_path
        else:
            file_to_process = tmp_path
        
        # Validate audio file
        if not is_audio_file(file_to_process):
            logger.error("No valid audio stream found")
            return JSONResponse(
                status_code=400,
                content={
                    "message": "The uploaded file does not contain a valid audio stream",
                    "error": "Invalid or missing audio stream"
                }
            )
        
        # Process audio and detect chunks
        chunk_times = get_large_audio_chunks_on_silence(file_to_process)
        if not chunk_times:
            logger.error("No significant audio chunks detected")
            return JSONResponse(
                status_code=400,
                content={
                    "message": "No significant audio chunks detected",
                    "error": "Unable to extract meaningful audio segments"
                }
            )

        # Convert ms to seconds
        chunk_times_sec = [(round(start / 1000, 2), round(end / 1000, 2)) for start, end in chunk_times]
        logger.info(f"Extracted {len(chunk_times_sec)} chunks from {video_file.filename}")
        return chunk_times_sec

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "message": "An error occurred while processing the file",
                "error": str(e)
            }
        )
    finally:
        # Cleanup temp files
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_path}: {e}")
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete audio file {audio_path}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8092)