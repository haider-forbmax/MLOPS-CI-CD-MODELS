# test for scene detection
from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
import uuid
from scene_detection_inference.schemas import *
from scene_detection_inference.utils import *
import time
import traceback
import requests
import sys
from scene_detection_inference.config import Config

security = HTTPBearer()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Scene Detection API",
    version="1.0.0")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTPException and ensure consistent JSON format"""
    # If detail is already a dict with message and error, use it
    if isinstance(exc.detail, dict) and "message" in exc.detail and "error" in exc.detail:
        logger.error(f"HTTPException [{exc.status_code}]: {exc.detail['error']}")
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    
    # If detail is a string, wrap it in the expected format
    logger.error(f"HTTPException [{exc.status_code}]: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": str(exc.detail),
            "error": str(exc.detail)
        }
    )


@app.middleware("http")
async def block_multipart(request: Request, call_next):
    if request.url.path == "/v1/scene_detection":
        if not request.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse(
                status_code=415,
                content={
                    "message": "Only application/json is supported",
                    "error": "The content type provided is not supported for this endpoint. Please use application/json",
                },
            )
    return await call_next(request)

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle any unexpected exceptions"""
    logger.exception("Unhandled exception occurred")
    return JSONResponse(
        status_code=500,
        content={
            "message": "An internal server error occurred. Please try again later.",
            "error": f"Unexpected error: {str(exc)}"
        }
    )

def verify_api_key(authorization: str = Header(None)):
    """Verify Bearer token authorization"""
    if not authorization:
        logger.warning("Missing Authorization header")
        raise HTTPException(
            status_code=401,
            detail={
                "message": "Missing Authorization header",
                "error": "Authorization header not provided in request"
            }
        )

    try:
        
        api_key = authorization
        if api_key != Config.API_KEY:
            logger.warning("Invalid API key provided")
            raise HTTPException(
                status_code=401,
                detail={
                    "message": "Invalid API key",
                    "error": "Provided API key does not match"
                }
            )

        return api_key
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error("Error validating API key: %s", e)
        raise HTTPException(
            status_code=401,
            detail={
                "message": "Invalid Authorization header",
                "error": f"Error parsing authorization: {str(e)}"
            }
        )



@app.post("/v1/scene_detection",response_model = DetectionResponse)
async def scene_detection_endpoint(request: DetectionRequest,credentials: HTTPAuthorizationCredentials = Depends(security)):    
    logger.info(f"Detection request received for model: {request.model}")

    if not credentials:
        raise HTTPException(status_code=401, detail={"message":"Unauthorized","error":"Missing Authorization header"})
    try:
        api_key = verify_api_key(credentials.credentials)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message":"Internal server error","error":str(e)})
    start_time = time.time()
    try:
        preprocess_start = time.time()
        original_image = load_image(request.image)
        if original_image is None:
            logger.error(f" Image loading returned None")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to load image.",
                    "error": "Image loading returned None"
                }
            )
        preprocessing_time = (time.time() - preprocess_start) * 1000
        image_info = ImageInfo(
                width=original_image.width,
                height=original_image.height,
                format=original_image.format or "JPEG"
            )
        
        inference_start = time.time()
        result = process_image_with_florence(original_image)
        if not result:
            logger.error(f" Florence returned empty result")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Image processing completed but returned no results.",
                    "error": "Florence model returned empty result"
                }
            )
        inference_time = (time.time() - inference_start) * 1000

        postprocess_start = time.time()
        postprocessing_time = (time.time() - postprocess_start) * 1000

       
        total_time = (time.time() - start_time) * 1000
        response = DetectionResponse(
            id = f"detect_scene_{uuid.uuid4().hex[:8]}",
            created = int(time.time()),
            model = request.model,
            usage = Usage(inference_time_ms=round(inference_time, 2),
                          preprocessing_time_ms=round(preprocessing_time, 2),
                          postprocessing_time_ms=round(postprocessing_time, 2),
                          total_time_ms=round(total_time, 2)),
            detections = [Detection(
                          
                         class_name = result

        )],
            image_info = image_info
        )
        logger.info(f" Request completed successfully in {total_time:.2f}ms")
        return response

    
    finally:
        if original_image:
            try:
                original_image.close()
                logger.debug(f"Image resources cleaned up")
            except Exception as e:
                logger.warning(f" Error cleaning up image: {str(e)}")

# Health check route to ensure the API is running
@app.get("/live")
async def live_check():
    return {"live": True}

@app.get("/health")
async def health():
    try:
        # Root endpoint test
        root_resp = requests.get(f"{Config.API_URL}/", timeout=5)
        
        # Health endpoint test
        health_resp = requests.get(f"{Config.API_URL}/health", timeout=5)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Health check completed successfully.",
                "root_status": root_resp.status_code,
                "health_status": health_resp.status_code,
            }
        )
    except requests.exceptions.Timeout:
        return JSONResponse(
            status_code=504,
            content={
                "message": "Health check timed out.",
                "error": "Timeout when checking service health"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": "Health check failed.",
                "error": str(e)
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("scene_detection_inference.api:app", host="0.0.0.0", port=2010)
