import time, uuid
import logging
from fastapi import FastAPI, HTTPException, Header
from config import Config
from schemas import EmbeddingRequest, EmbeddingResponse, Usage, ImageInfo
from utils import ResNetDetector

app = FastAPI(title="ResNet100 Face Embedding API", version="1.0.0")
detector = ResNetDetector()
config = Config()

# Setup logging
logger = config.setup_logging()

def validate_api_key(api_key: str) -> bool:
    """Validate the provided API key against the configured key"""
    expected_key = config.API_KEY
    if not expected_key or expected_key == "your_api_key_here":
        logger.warning("API_KEY not properly configured in environment")
        return False
    return api_key == expected_key

@app.get("/health")
async def health_check():
    """Health check endpoint with Triton server and model status"""
    logger.debug("Health check requested")
    try:
        triton_client = detector.get_triton_client()

        # Check server status
        server_live = triton_client.is_server_live()
        server_ready = triton_client.is_server_ready()
        model_status = triton_client.is_model_ready(config.TRITON_MODEL_NAME)

        logger.info(f"Server live: {server_live}, ready: {server_ready}, model ready: {model_status}")

        if not server_live or not server_ready or not model_status:
            triton_status = "unhealthy"
        else:
            triton_status = "healthy"

        # Overall service status
        overall_status = "healthy" if triton_status == "healthy" else "unhealthy"
        model_health = "healthy" if model_status else "unhealthy"

        response = {
            "status": overall_status,
            "service": "ResNet100 Face Embedding API",
            "triton_server": triton_status,
            "model_status": model_health
        }

        if overall_status == "unhealthy":
            logger.warning("Health check failed - service unhealthy")
            raise HTTPException(status_code=503, detail=response)

        logger.debug("Health check passed - service healthy")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        response = {
            "status": "unhealthy",
            "service": "ResNet100 Face Embedding API",
            "triton_server": "unreachable",
            "model_status": "unreachable"
        }
        raise HTTPException(status_code=503, detail=response)

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest, authorization: str = Header(None)):
    """Generate face embeddings using ResNet model"""
    logger.info(f"Embedding request received for model: {request.model}")

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

    start_time = time.time()

    # Prepare Triton client and inputs
    triton_client = detector.get_triton_client()

    model_status = triton_client.is_model_ready(request.model)
    if not model_status or request.model != config.TRITON_MODEL_NAME:
        logger.error(f"Model {request.model} not available. Available model: {config.TRITON_MODEL_NAME}")
        raise HTTPException(
            status_code=400,
            detail=f"Model {request.model} not available. Available model: {config.TRITON_MODEL_NAME}"
        )

    try:
        # Load and preprocess image
        preprocess_start = time.time()
        original_image = detector.load_image(request.image)
        preprocessed_image = detector.preprocess_image(original_image)

        # Get image info
        image_info = ImageInfo(
            width=original_image.width,
            height=original_image.height,
            format=original_image.format or "JPEG"
        )

        # Run inference
        inference_start = time.time()
        embedding = detector.run_triton_inference(preprocessed_image)
        inference_time = (time.time() - inference_start) * 1000

        total_time = (time.time() - start_time) * 1000

        logger.debug(f"Embedding generation completed in {total_time:.2f}ms")

        # Response
        return EmbeddingResponse(
            id=f"embed_resnet100_{uuid.uuid4().hex[:8]}",
            task=config.TASK_TYPE,
            created=int(time.time()),
            model=request.model,
            usage=Usage(
                inference_time_ms=round(inference_time, 2),
                total_time_ms=round(total_time, 2)
            ),
            embeddings=embedding.tolist() if request.include_embeddings else None,
            image_info=image_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "triton_server_url": config.TRITON_SERVER_URL,
        "model_name": config.TRITON_MODEL_NAME,
        "max_image_size": config.MAX_IMAGE_SIZE,
        "supported_formats": config.SUPPORTED_FORMATS,
        "task_type": config.TASK_TYPE,
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "ResNet100 Face Embedding API",
        "version": "1.0.0",
        "description": "Face embedding API using ResNet100 model with Triton Inference Server",
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "health": "/health",
            "config": "/config",
            "docs": "/docs"
        },
        "supported_formats": config.SUPPORTED_FORMATS,
        "max_image_size_mb": config.MAX_IMAGE_SIZE / (1024 * 1024)
    }

import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)