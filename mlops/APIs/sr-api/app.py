from fastapi import FastAPI, UploadFile, Depends, HTTPException, Header,File, Request
## import moviepy.editor as mp  # unused
import os
import tempfile
import librosa
from fastapi.responses import JSONResponse
from datetime import datetime
from inference_pipeline import *
from utils import *
import logging
import tritonclient.http as httpclient
from config import Config, HealthResponse
import traceback
## from moviepy.editor import VideoFileClip  # unused
import uuid
# import subprocess

app = FastAPI(
    title="Speaker Recognition API",
    version="1.0.0")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request for tracking"""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        logger.info(f"[{request_id}] Response: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"[{request_id}] Unhandled exception: {e}")
        raise

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
    

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with Triton server and model status"""
    logger.debug("Health check requested")
    try:
        with httpclient.InferenceServerClient(url=Config.MODEL_URL) as triton_client:
            server_live = triton_client.is_server_live()
            server_ready = triton_client.is_server_ready()
            model_ready = triton_client.is_model_ready(Config.MODEL_NAME)

        triton_status = "healthy" if server_live and server_ready else "unhealthy"
        model_status_str = "healthy" if model_ready else "unhealthy"
        overall_status = "healthy" if triton_status == "healthy" and model_status_str else "unhealthy"
        response = HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
        ) 
        if overall_status == "unhealthy":
            logger.warning("Health check failed - service unhealthy")
            raise HTTPException(status_code=503, detail={
                    "message": "Service is currently unavailable",
                    "error": response.model_dump()
                })

        logger.debug("Health check passed - service healthy")
        return response

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Health check error: {e}")
        response = HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
        )
        raise HTTPException(status_code=503, detail={
                "message": "Unable to perform health check",
                "error": str(e)
            })


@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "triton_server_url": Config.MODEL_URL,
        "model_name": Config.MODEL_NAME,
        "video_extensions": Config.VIDEO_EXTENSION
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Speaker Recognition API",
        "version": "1.0.0",
        "description": "Speaker Recognition API using Triton Inference Server",
        "endpoints": {
            "speaker_recognize": "/speaker_recognize",
            "health": "/health",
            "config": "/config",
            "docs": "/docs"
        },
        "supported_formats": {
            "video": Config.VIDEO_EXTENSION,
        }
    }

@app.post("/v1/speaker_recognition")
async def speaker_recognition_endpoint(request: Request,
    video_file: UploadFile,
    _: str = Depends(verify_api_key)
    ):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(f"[{request_id}] Received request for Speaker Recognition with file: {video_file.filename}")
    # Create a new thread for processing each user's video
    try:
        result = process_video(video_file)
        logger.info(f"[{request_id}] Successfully processed file: {video_file.filename}")
        return result

    except UserAudioError as ue:
        logger.error(f"[{request_id}] UserAudioError: {ue}")
        return JSONResponse(
            status_code=400,
            content={
                "message": "There was an issue with the uploaded audio/video file",
                "error": str(ue)
            }
        )
    except TritonServerError as te:
        # Triton server errors (503)
        logger.error(f"[{request_id}] TritonServerError: {te}")
        return JSONResponse(
            status_code=503,
            content={
                "message": "Speaker recognition service is temporarily unavailable. Please try again later",
                "error": str(te)
            }
        )
    except RuntimeError as re:
        # Processing errors (500)
        logger.error(f"[{request_id}] RuntimeError: {re}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "message": "An internal error occurred during processing",
                "error": str(re)
            }
        )
    except Exception as e:
        logger.error(f"[{request_id}] Error processing video file: {video_file.filename}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "message": "An unexpected error occurred. Please try again later",
                "error": str(e)
            }
        )

@app.post("/v1/add_speaker")
async def add_speaker_endpoint(
    person_label: str,
    audio_file: UploadFile = None,
    _: str = Depends(verify_api_key)
):
    """
    Endpoint to add a new speaker using an audio or video file.
    Validates that the file has a non-silent audio track.
    """
    try:
        logger.info(f"Received add_speaker request for label: {person_label}, file: {audio_file.filename}")       
        file_name, file_extension = os.path.splitext(audio_file.filename)

        # if any(file_extension.lower() == ext for ext in Config.VIDEO_EXTENSION):
        # Step 1: Save uploaded file temporarily
        file_ext = os.path.splitext(audio_file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(await audio_file.read())
            tmp_path = tmp_file.name
        # Step 2: Validate audio content
        try:
            waveform, sr = librosa.load(tmp_path, sr=16000, mono=True)
        except Exception:
            traceback.print_exc()
            return JSONResponse(
                status_code=400,
                content={
                    "message": "The uploaded file is not a valid audio file or contains no audio",
                    "error": str(e)
                }
            )
        if waveform is None or len(waveform) == 0:
            return JSONResponse(
                status_code=400,
                content={
                    "message": "The uploaded file does not contain any audio data",
                    "error": "Empty audio file"
                }
            )
        # Check for silence (very low amplitude)
        if np.abs(waveform).mean() < 1e-4:
            return JSONResponse(
                status_code=400,
                content={
                    "message": "The audio file is silent or has no meaningful sound",
                    "error": "Silent audio detected"
                }
            )
        # Step 3: Train the model with this audio
        try:
            logger.info(f"Training speaker embedding for: {person_label}")
            training(person_label, tmp_path)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error during training: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "message": "An error occurred while training the speaker model",
                    "error": str(e)
                }
            )
        # Step 4: Clean up temporary file
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        logger.info(f"Successfully added speaker '{person_label}' to Milvus database")
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Speaker '{person_label}' added successfully"
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in add_speaker: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "message": "An unexpected error occurred while adding the speaker",
                "error": str(e)
            }
        )
    
    finally:
        # Step 4: Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.post("/v1/delete_speaker")
async def delete_speaker_endpoint(
    person_label: str,
    _: str = Depends(verify_api_key)
):

    try:
        delete_embeddings(person_label)
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Speaker '{person_label}' deleted successfully"
            }
        )
    except Exception as e:
        logger.error(f"Error deleting speaker '{person_label}': {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "message": f"An error occurred while deleting speaker '{person_label}'",
                "error": str(e)
            }
        )
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8091)
