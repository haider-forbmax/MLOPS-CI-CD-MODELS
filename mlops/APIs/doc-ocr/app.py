# testttt
"""
FastAPI Service for Enhanced Universal OCR
Single endpoint that handles 30+ document formats
"""
import subprocess
import os
import requests
import logging
import tempfile
import traceback
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from enhanced_document_ocr import EnhancedUniversalOCR, OCRConfig
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from pathlib import Path
from config import Config
from utils import validate_api_key, clean_page_text, clean_ocr_output


config = Config()

# Configure logging
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


# Initialize FastAPI app
app = FastAPI(
    title="Document OCR API",
    description="OCR service supporting multiple document formats",
    version="1.0.0"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize OCR client with config from environment
ocr_config = OCRConfig.from_env()
ocr_client = EnhancedUniversalOCR(ocr_config=ocr_config)

# Get supported formats
SUPPORTED_FORMATS = ocr_client.get_all_supported_formats()


@app.get("/")
async def root():
    """API information and supported formats (based on automated tests)"""
    return {
        "service": "Document OCR API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "document_ocr": "v1/document_ocr",
            "config": "/config"
        }
    }


@app.get("/health")
async def health_check():
    url = f"{Config.MODEL_URL}/health"

    try:
        response = requests.get(url, timeout=5)
        
        # Always capture status code
        status_code = response.status_code
        
        # Try to parse JSON, fallback to raw text
        try:
            data = response.json()
        except ValueError:
            data = {}

        # Build consistent 3-field output
        return {
            "success": data.get("success", True if status_code == 200 else False),
            "status_code": status_code,
            "response": data.get("response", response.text or "")
        }

    except requests.exceptions.RequestException as e:
        # Handle connection errors etc.
        return {
            "success": False,
            "status_code": None,
            "response": str(e)
        }


@app.get("/config")
async def get_config():
    """API information and supported formats (based on automated tests)"""

    # Flatten count
    total_formats = len(Config.SUPPORTED_EXTENSIONS)

    return {
        "model_name": Config.MODEL_NAME,
        "total_formats_supported": total_formats,
        "supported_formats": Config.SUPPORTED_EXTENSIONS,
        "ocr_config": {
            "max_concurrent_requests": ocr_config.max_concurrent_requests,
            "max_retries": ocr_config.max_retries,
            "base_delay": ocr_config.base_delay,
            "max_delay": ocr_config.max_delay,
            "timeout_per_page": ocr_config.timeout_per_page,
            "retry_strategy": ocr_config.retry_strategy.value
        }
    }


@app.post("/v1/document_ocr")
async def process_document(
    file: UploadFile = File(..., description="Document to process (any supported format)"),
    authorization: str = Header(None)
    ):
    """
    OCR endpoint that processes ANY supported document format
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

    logger.info(f"Received request for transcribe_video with file: {file.filename}")

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Unsupported file format: {file_ext}",
                "supported_formats": SUPPORTED_FORMATS,
                "hint": "Check /formats endpoint for complete list"
            }
        )
    
    # Validate output format
    output_format = "markdown"
    
    # Create temporary file
    temp_file = None
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file = tmp.name
        
        logger.info(f"Processing {file.filename} ({file_ext}) - {len(content)} bytes")
        
        # Process document
        result = ocr_client.process_document(
            file_path=temp_file
        )
        
        if result["success"]:
            logger.info(f"Success: {file.filename} - {result['pages']} pages, {result['total_tokens']} tokens")
            
            # Prepare response
            response_data = {}

            result["text"] = clean_page_text(result["text"])

            # Add text or page results based on combine_pages
            response_data["markdown_text"] = result["text"]
            response_data["markdown_text"] = clean_ocr_output(response_data["markdown_text"])


            response_data["raw_text"] = result["text"]
            
            return JSONResponse(content=response_data)
        
        else:
            logger.error(f"Failed: {file.filename} - {result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "OCR processing failed",
                    "details": result.get("error"),
                    "filename": file.filename
                }
            )
    
    except HTTPException:
        traceback.print_exc()
        raise
    
    except Exception as e:
        logger.error(f"Exception processing {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "details": str(e),
                "filename": file.filename
            }
        )
    
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass


@app.get("/system/dependencies")
async def check_dependencies():
    """Check if required system dependencies are installed"""
    dependencies = {}
    
    # Check LibreOffice
    try:
        result = subprocess.run(
            ['libreoffice', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        dependencies['libreoffice'] = {
            "installed": True,
            "version": result.stdout.strip()
        }
    except:
        dependencies['libreoffice'] = {
            "installed": False,
            "error": "Not found - install with: sudo apt-get install libreoffice"
        }
    
    # Check poppler (for pdf2image)
    try:
        result = subprocess.run(
            ['pdftoppm', '-v'],
            capture_output=True,
            text=True,
            timeout=5
        )
        dependencies['poppler'] = {
            "installed": True,
            "version": result.stderr.strip()
        }
    except:
        dependencies['poppler'] = {
            "installed": False,
            "error": "Not found - install with: sudo apt-get install poppler-utils"
        }
    
    return dependencies


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
