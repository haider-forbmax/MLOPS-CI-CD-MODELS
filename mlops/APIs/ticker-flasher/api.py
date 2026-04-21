# ticker test workflow
import time, uuid
import logging
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from detector import YOLO5Detector
from config import Config
import tritonclient.http as httpclient

from schemas import (
    DetectionRequest, DetectionResponse, Detection, BoundingBox, ImageInfo, Usage,
    HealthResponse, ModelInfo
)

app = FastAPI(title="YOLOv5 Ticker Flasher Detection API", version="1.0.0")
detector = YOLO5Detector()
config = Config()

# Setup logging
logger = config.setup_logging()

# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Convert HTTPException to consistent error format"""
    logger.error(f"HTTPException: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.detail if isinstance(exc.detail, str) else "An error occurred",
            "error": exc.detail
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "message": "Invalid request data. Please check your input and try again.",
            "error": str(exc.errors())
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected exceptions"""
    logger.exception("Unhandled exception occurred")
    return JSONResponse(
        status_code=500,
        content={
            "message": "An unexpected error occurred. Please try again later.",
            "error": f"Internal server error: {str(exc)}"
        }
    )


def validate_api_key(api_key: str) -> bool:
    """Validate the provided API key against the configured key"""
    expected_key = config.API_KEY
    if not expected_key or expected_key == "your_api_key_here":
        logger.warning("API_KEY not properly configured in environment")
        return False
    return api_key == expected_key



@app.post("/v1/detect", response_model=DetectionResponse)
async def detect_objects(request: DetectionRequest, authorization: str = Header(None)):
    """Detect ticker, flasher in image using YOLO5 model"""
    ## overwrite model name from config to ensure correct model is used
    request.model = config.TRITON_MODEL_NAME
    
    logger.info(f"Detection request received for model: {request.model}")

    # Validate API key
    if not authorization:
        logger.warning("Missing Authorization header")
        return JSONResponse(
            status_code=401,
            content={
                "message": "Authorization header is required to access this endpoint.",
                "error": "Missing Authorization header"
            }
        )

    # Extract API key from Authorization header (expecting "Bearer <api_key>")
    try:
        auth_parts = authorization.split()
        if len(auth_parts) != 2 or auth_parts[0].lower() != "bearer":
            logger.warning("Invalid Authorization header format")
            return JSONResponse(
                status_code=401,
                content={
                    "message": "Invalid authorization format. Please use 'Bearer <api_key>' format.",
                    "error": "Invalid Authorization header format"
                }
            )

        api_key = auth_parts[1]
        if not validate_api_key(api_key):
            logger.warning("Invalid API key provided")
            return JSONResponse(
                status_code=401,
                content={
                    "message": "The provided API key is invalid. Please check your credentials.",
                    "error": "Invalid API key"
                }
            )
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        return JSONResponse(
            status_code=401,
            content={
                "message": "Failed to validate authorization credentials. Please try again.",
                "error": f"Authorization validation error: {str(e)}"
            }
        )


    # Validate parameters
    if not (0 <= request.parameters.confidence <= 1):
        logger.error(f"Invalid confidence value: {request.parameters.confidence}. Must be between 0 and 1")
        return JSONResponse(
            status_code=400,
            content={
                "message": "Confidence threshold must be between 0 and 1.",
                "error": f"Invalid confidence value: {request.parameters.confidence}"
            }
        )

    if not (0 <= request.parameters.nms_threshold <= 1):
        logger.error(f"Invalid nms_threshold value: {request.parameters.nms_threshold}. Must be between 0 and 1")
        return JSONResponse(
            status_code=400,
            content={
                "message": "NMS threshold must be between 0 and 1.",
                "error": f"Invalid nms_threshold value: {request.parameters.nms_threshold}"
            }
        )
        


    start_time = time.time()
    # Prepare Triton client and inputs
    try:
        # Prepare Triton client and inputs
        triton_client = detector.get_triton_client()
    except HTTPException as he:
        # HTTPException from detector will be caught by global handler
        raise
    except Exception as e:
        logger.error(f"Failed to connect to Triton server: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "message": "Unable to connect to the inference server. Please try again later.",
                "error": f"Triton server connection failed: {str(e)}"
            }
        )

    try:
        model_status = triton_client.is_model_ready(request.model)
        if not model_status or request.model != config.TRITON_MODEL_NAME:
            logger.error(f"Model {request.model} not available")
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"The requested model is not available. Available model: {config.TRITON_MODEL_NAME}",
                    "error": f"Model {request.model} not ready or not found"
                }
            )
    except Exception as e:
        logger.error(f"Error checking model status: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "message": "Unable to verify model availability. Please try again later.",
                "error": f"Model status check failed: {str(e)}"
            }
        )

    # Get model-specific classes from Triton model config metadata (cached for 10 minutes).
    model_classes = detector.extract_model_classes(triton_client, request.model)

    try:
        # Load and preprocess image
        preprocess_start = time.time()
        try:
            original_image = detector.load_image(request.image)
        except HTTPException as he:
            # Re-raise to be caught by global handler
            logger.error(f"Image loading failed: {he.detail}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading image: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={
                    "message": "Unable to process the provided image. Please verify the image is valid.",
                    "error": f"Image loading error: {str(e)}"
                }
            )
        preprocessed_image = detector.preprocess_image(original_image)
        preprocessing_time = (time.time() - preprocess_start) * 1000

        # Get image info
        image_info = ImageInfo(
            width=original_image.width,
            height=original_image.height,
            format=original_image.format or "JPEG"
        )
        logger.debug(f"Image info: {image_info}")
        inputs = []
        inputs.append(httpclient.InferInput(config.MODEL_INPUT_NAME, preprocessed_image.shape, "FP32"))
        inputs[0].set_data_from_numpy(preprocessed_image)

        outputs = []
        outputs.append(httpclient.InferRequestedOutput(config.MODEL_OUTPUT_NAME))

        # Run inference
        inference_start = time.time()
        try:
            results = triton_client.infer(config.TRITON_MODEL_NAME, inputs=inputs, outputs=outputs)
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            return JSONResponse(
                status_code=502,
                content={
                    "message": "Object detection inference failed. Please try again later.",
                    "error": f"Triton inference error: {str(e)}"
                }
            )
        inference_time = (time.time() - inference_start) * 1000


            # Get output and postprocess
        postprocess_start = time.time()
        try:
            output_data = results.as_numpy(config.MODEL_OUTPUT_NAME)
            final_detections = detector.postprocess_detections(
                output_data, 
                original_image.size,
                request.parameters.confidence,
                request.parameters.nms_threshold,
                model_classes
            )
        except Exception as e:
            logger.error(f"Postprocessing failed: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "message": "Failed to process detection results. Please try again.",
                    "error": f"Postprocessing error: {str(e)}"
                }
            )
        postprocessing_time = (time.time() - postprocess_start) * 1000

        # Convert detections to response format
        detection_objects = []
        for detection in final_detections:
            x, y, w, h, confidence = detection[:5]
            class_id, class_name = detector.get_class_info(detection, model_classes)

            # Calculate bounding box coordinates
            x1 = max(0, int(x - w / 2))
            y1 = max(0, int(y - h / 2))
            x2 = min(original_image.width, int(x + w / 2))
            y2 = min(original_image.height, int(y + h / 2))

            # Skip invalid detections with zero or negative area
            if x1 >= x2 or y1 >= y2 or w <= 0 or h <= 0:
                logger.warning(f"Skipping invalid detection {class_name} with bounding box [{x1}, {y1}, {x2}, {y2}]")
                continue

            logger.debug(f"Detection: {class_name} ({confidence:.2f}) at [{x1}, {y1}, {x2}, {y2}]")
            # Crop object if requested
            object_image_b64 = None
            if request.response_format.object_images:
                object_image_b64 = detector.crop_object(original_image, detection)
                if object_image_b64 is None:
                    logger.warning(f"Failed to crop object {class_name} with invalid bounding box [{x1}, {y1}, {x2}, {y2}]")

            detection_obj = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=float(confidence),
                bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                object_image=object_image_b64
            )
            detection_objects.append(detection_obj)

        # Include labled image if requested
        labeled_image_b64 = None
        if request.response_format.labeled_image:
            labeled_image = detector.draw_bounding_boxes_labels(original_image, final_detections, model_classes)
            labeled_image_b64 = detector.image_to_base64(labeled_image)

        # Include anotated image if requested
        annotated_image_b64 = None
        if request.response_format.annotated_image:
            annotated_image = detector.draw_bounding_boxes(original_image, final_detections, model_classes)
            annotated_image_b64 = detector.image_to_base64(annotated_image)


        total_time = (time.time() - start_time) * 1000

        logger.debug(f"Detection completed: {len(detection_objects)} objects found in {total_time:.2f}ms")

        response = DetectionResponse(
            id=f"detect_yolo5_ticker_flasher_{uuid.uuid4().hex[:8]}",
            task="detection",
            created=int(time.time()),
            model=request.model,
            usage=Usage(
                inference_time_ms=round(inference_time, 2),
                preprocessing_time_ms=round(preprocessing_time, 2),
                postprocessing_time_ms=round(postprocessing_time, 2),
                total_time_ms=round(total_time, 2)
            ),
            detections=detection_objects,
            image_info=image_info,
            labeled_image=labeled_image_b64,
            annotated_image=annotated_image_b64
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "message": "An internal server error occurred during object detection. Please try again later.",
                "error": f"Internal error: {str(e)}"
            }
        )

@app.get("/health", response_model=HealthResponse)
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
        model_status = "healthy" if model_status else "unhealthy"

        response = HealthResponse(
            status=overall_status,
            service="YOLO5 Ticker Flasher Detection API",
            triton_server=triton_status,
            model_status=model_status
        )

        if overall_status == "unhealthy":
            logger.warning("Health check failed - service unhealthy")
            return JSONResponse(
            status_code=503,
            content={
                "message": "Unable to perform health check. Service may be temporarily unavailable.",
                "error": f"Health check error: {str(e)}"
            }
        )

        logger.debug("Health check passed - service healthy")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        response = HealthResponse(
            status="unhealthy",
            service="YOLO5 Ticker Flasher Object Detection API",
            triton_server="unreachable",
            model_status="unreachable"
        )
        return JSONResponse(
            status_code=503,
            content={
                "message": "Unable to perform health check. Service may be temporarily unavailable.",
                "error": f"Health check error: {str(e)}"
            }
        )


@app.get("/classes")
async def get_all_classes():
    """Get class mappings from Triton model config metadata (cached for 10 minutes)."""
    triton_client = detector.get_triton_client()
    try:
        classes = detector.extract_model_classes(triton_client, config.TRITON_MODEL_NAME)
        return {"classes": classes}
    finally:
        triton_client.close()

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "triton_server_url": config.TRITON_SERVER_URL,
        "model_name": config.TRITON_MODEL_NAME,
        "default_confidence_threshold": config.DEFAULT_CONFIDENCE_THRESHOLD,
        "default_nms_threshold": config.DEFAULT_NMS_THRESHOLD,
        "max_image_size": config.MAX_IMAGE_SIZE,
        "supported_formats": config.SUPPORTED_FORMATS,
        "task_type": config.TASK_TYPE,
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "YOLO5 Ticker Flasher Detection API",
        "version": "1.0.0",
        "description": "Ticker Flasher detection API using YOLO5 models with Triton Inference Server",
        "endpoints": {
            "detect": "/v1/detect",
            "health": "/health",
            "classes": "/classes",
            "config": "/config",
            "docs": "/docs"
        },
        "supported_formats": config.SUPPORTED_FORMATS,
        "max_image_size_mb": config.MAX_IMAGE_SIZE / (1024 * 1024)
    }



import uvicorn

if __name__ == "__main__":
      uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
