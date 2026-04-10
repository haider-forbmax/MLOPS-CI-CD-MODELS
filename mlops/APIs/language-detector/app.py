import os
import tempfile
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Response Models
class DetectionResponse(BaseModel):
    filename: str
    language: str
    confidence: float
    duration: float

class HealthResponse(BaseModel):
    status: str
    device: str | None = None

# Global model variable
model = None

def load_model():
    """
    Load the Whisper model.
    Attempts to load on CUDA first, falls back to CPU.
    """
    global model
    model_size = os.getenv("MODEL_SIZE", "medium")
    device = os.getenv("DEVICE", "cuda")
    compute_type = os.getenv("COMPUTE_TYPE", "float16" if device == "cuda" else "int8")

    try:
        logger.info(f"Loading {model_size} model on {device} with {compute_type}...")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        logger.warning(f"Failed to load on {device}: {e}. Falling back to CPU...")
        try:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        except Exception as cpu_e:
            logger.error(f"Failed to load model on CPU: {cpu_e}")
            raise RuntimeError("Could not load WhisperModel") from cpu_e
    logger.info("Model loaded successfully.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model in a thread to avoid blocking
    await asyncio.to_thread(load_model)
    yield
    # Shutdown: Clean up resources if needed
    global model
    del model

app = FastAPI(title="Language Detector Service", lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    if model is None:
        return JSONResponse(status_code=503, content={"status": "initializing"})
    return HealthResponse(status="healthy", device=model.model.device)

def process_detection(file_path: str):
    """
    Synchronous function to run inference using faster-whisper.
    Returns: dict with language info
    """
    # beam_size=1 is faster for detection
    segments, info = model.transcribe(file_path, beam_size=1)
    return {
        "detected_language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_language(
    file: UploadFile = File(...),
):
    """
    High-availability, async endpoint to detect language from audio/video.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    # Create a temporary file to store the upload
    fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename or "")[1])
    try:
        # Async writing to file
        with os.fdopen(fd, 'wb') as tmp:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                tmp.write(chunk)

        # Run inference in a thread pool executor
        if model is None:
             raise HTTPException(status_code=503, detail="Model is loading")

        result = await asyncio.to_thread(process_detection, tmp_path)

        return DetectionResponse(
            filename=file.filename or "unknown",
            language=result["detected_language"],
            confidence=result["language_probability"],
            duration=result["duration"]
        )

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {tmp_path}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2021)