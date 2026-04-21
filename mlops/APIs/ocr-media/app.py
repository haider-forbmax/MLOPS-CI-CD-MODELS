# test the workflow for ocr media API
"""
Optimized Media OCR API
- Fully async processing
- No disk I/O (in-memory image handling)
- Connection pooling with lifecycle management
- Proper concurrency control
"""

import os
import base64
import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body, Header
from olmocr import OCRClient
from schema import OCRRequest, OCRResponse
from config import Config

config = Config()

# Configure logging
log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global client (initialized in lifespan)
ocr_client: OCRClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - initialize and cleanup resources."""
    global ocr_client
    
    # Startup: Initialize async OCR client with connection pooling
    ocr_client = OCRClient(
        base_url=config.MODEL_URL,
        max_connections=100,
        max_concurrent_requests=50,  # Adjust based on backend capacity
        timeout=120.0,
        enable_cache=True,
        cache_size=256
    )
    logger.info("OCR client initialized with connection pooling")
    
    yield
    
    # Shutdown: Cleanup resources
    if ocr_client:
        await ocr_client.close()
        logger.info("OCR client closed")


app = FastAPI(
    title="Media OCR API",
    version="1.0",
    lifespan=lifespan
)


# ---------------- HELPER FUNCTIONS ----------------
def safe_base64_decode(data: str) -> bytes:
    """Safely decode base64, fixing padding and removing URI prefix if needed."""
    # Remove data URI prefix if present
    if data.startswith("data:image"):
        data = data.split(",", 1)[1]
    
    # Fix missing padding
    missing_padding = len(data) % 4
    if missing_padding:
        data += "=" * (4 - missing_padding)
    
    try:
        return base64.b64decode(data)
    except Exception as e:
        raise ValueError(f"Invalid base64 string: {e}")


def validate_api_key(api_key: str) -> bool:
    """Validate the provided API key against the configured key."""
    expected_key = config.API_KEY
    if not expected_key or expected_key == "your_api_key_here":
        logger.warning("API_KEY not properly configured in environment")
        return False
    return api_key == expected_key


# ---------------- ENDPOINTS ----------------
@app.get("/")
async def root():
    """API root endpoint with basic info."""
    return {
        "name": "OCR API",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "ocr": "/v1/ocr",
            "config": "/config"
        }
    }


@app.get("/health")
async def health_check():
    """Check if the OCR model is ready."""
    try:
        if await ocr_client.health():
            return {
                "status": "ok",
                "message": "Model is ready",
                "model": "OlmOCR"
            }
        else:
            logger.warning("OlmOCR health check returned False")
            raise HTTPException(status_code=503, detail="Model not ready")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@app.get("/config")
async def get_config():
    """Get current configuration."""
    return {
        "model_name": "OlmOCR",
        "supported_formats": config.SUPPORTED_FORMATS,
        "task_type": config.TASK_TYPE,
    }


@app.post("/v1/ocr", response_model=OCRResponse)
async def ocr_endpoint(
    payload: OCRRequest = Body(...),
    authorization: str = Header(None)
):
    """
    Process OCR from base64-encoded image.
    
    Expected JSON payload:
    {
        "image_base64": "<base64_string>"
    }
    
    Returns:
    {
        "raw_text": "<extracted_text>",
        "markdown_text": "<extracted_text>"
    }
    """
    # Validate authorization
    if not authorization:
        logger.warning("Missing Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    try:
        auth_parts = authorization.split()
        if len(auth_parts) != 2 or auth_parts[0].lower() != "bearer":
            raise HTTPException(
                status_code=401,
                detail="Invalid Authorization header format. Expected: Bearer <api_key>"
            )
        
        if not validate_api_key(auth_parts[1]):
            logger.warning("Invalid API key provided")
            raise HTTPException(status_code=401, detail="Invalid API key")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating API key: {e}")
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    
    # Validate input
    if not payload.image_base64:
        raise HTTPException(status_code=400, detail="Missing 'image_base64' in request body")
    
    try:
        # Decode base64 image (no disk I/O)
        image_data = safe_base64_decode(payload.image_base64)
        
        # Log image size
        size_mb = len(image_data) / (1024 * 1024)
        logger.info(f"Processing image: {size_mb:.2f} MB")
        
        # Process OCR asynchronously (in-memory)
        result = await ocr_client.ocr_with_layout(
            image_data=image_data,
            language="auto",
            resize=True
        )
        
        # Validate result
        if not isinstance(result, dict):
            logger.error(f"Invalid result type: {type(result)}")
            raise HTTPException(status_code=500, detail="OCR returned invalid format")
        
        if "raw_text" not in result or "markdown_text" not in result:
            logger.error(f"Missing fields in result: {result.keys()}")
            raise HTTPException(status_code=500, detail="OCR returned incomplete result")
        
        return OCRResponse(
            raw_text=result.get("raw_text", ""),
            markdown_text=result.get("markdown_text", "")
        )
        
    except HTTPException:
        raise
    except ValueError as ve:
        logger.error(f"Value error: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {ve}")
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {e}")


# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=config.API_PORT,
        reload=False,
        log_level="info",
        workers=1  # Use 1 worker since we're async; scale with replicas instead
    )