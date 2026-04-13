#ttttttttttttttTtttest for the git ops"
from fastapi import FastAPI, HTTPException, status, Header, Depends, Request
from datetime import datetime
import time
from fastapi.responses import JSONResponse
import logging
import base64
import numpy as np
import cv2
import asyncio
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from schemas import *
from utils import face_client, milvus_client, generate_request_id, generate_face_id, overlay_face_names
from config import Config
security = HTTPBearer()

train_flush = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face-recognition-api")

app = FastAPI(title="Face Recognition API", version="1.0.0")

def verify_api_key(authorization: str = Header(None)):
    """Verify Bearer token authorization"""
    if not authorization:
        logger.warning("Missing Authorization header")
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )

    try:
        api_key = authorization
        if api_key != Config.API_KEY:
            logger.warning("Invalid API key provided")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )

        return api_key
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error validating API key: %s", e)
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header"
        )

@app.middleware("http")
async def block_multipart(request: Request, call_next):
    if request.url.path == "/v1/scene_detection":
        if not request.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse(
                status_code=415,
                content={
                    "message": "Only application/json is supported",
                    "error": "The content type provided is not supported for this endpoint. Please use application/json"
                    },
            )
    return await call_next(request)

@app.get("/health", response_model=HealthResponse)
async def health():
    # Check external services
    services_status = {}

    
    try:
        services_status["milvus"] = "healthy" if milvus_client.known_collection else "unhealthy"
    except Exception:
        services_status["milvus"] = "unhealthy"

    # Check for unhealthy services
    unhealthy_services = [name for name, status in services_status.items() if status != "healthy"]

    if unhealthy_services:
        # Return 503 Service Unavailable if any service is down
        error_message = f"Services unavailable: {', '.join(unhealthy_services)}"
        logger.warning("Health check failed: %s", error_message)

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "message": error_message,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "services": services_status,
                "unavailable_services": unhealthy_services
            }
        )

    # All services healthy
    logger.info("All services healthy")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": services_status
    }

@app.get("/config")
async def get_config():
    return Config.get_config()

def decode_base64_image(b64_string: str):
    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_image(image: np.ndarray) -> str:
    """
    Convert OpenCV image (numpy array) to base64 JPEG string.
    """
    if image is None:
        return None

    success, buffer = cv2.imencode(".jpg", image)

    if not success:
        raise ValueError("Failed to encode image")

    return base64.b64encode(buffer).decode("utf-8")

@app.post("/v1/predict", response_model=PredictResponse)
async def predict(request: PredictRequest,credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail={"message":"Unauthorized","error":"Missing Authorization header"})
    try:
        api_key = verify_api_key(credentials.credentials)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message":"Internal server error","error":str(e)})
    start_time = time.time()
    request_id = generate_request_id()
    logger.info("Processing prediction request %s", request_id)

    # Validate model name
    if request.model != "face-recognition":
        logger.warning("Invalid model name: %s", request.model)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name '{request.model}'. Expected 'face-recognition'"
        )

    try:
        # Step 1: Face detection
        detection_start = time.time()
        image_bytes = base64.b64decode(request.image.data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        height, width = image.shape[:2]

        # ---------------------------------
        # 2️⃣ Detection + Embedding
        # ---------------------------------
        detection_start = time.time()

        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        result = face_client.extract(image_bytes)
        embeddings = result['embeddings']
        detection_time = (time.time() - detection_start) * 1000

        if not result:
            return PredictResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                usage=Usage(
                    detection_time_ms=detection_time,
                    embedding_time_ms=0,
                    search_time_ms=0,
                    total_time_ms=detection_time
                ),
                detections=[],
                image_info=ImageInfo(width=width, height=height, format="jpg"),
            )
        # ---------------------------
        # 3. Generate embeddings in parallel
        # ---------------------------
        search_start = time.time()

        batch_matches = milvus_client.search_batch(
            embeddings,
            similarity_threshold=request.parameters.similarity_threshold,
        )

        search_time = (time.time() - search_start) * 1000

        # ---------------------------------
        # 5️⃣ Build Response + Store Unknowns
        # ---------------------------------
        detections: List[Detection] = []

        embeddings = result["embeddings"]
        bboxes = result["bboxes"]
        confidences = result["confidences"]

        for idx, matches in enumerate(batch_matches):
            best_match = matches[0]

            embedding = embeddings[idx]
            bbox = bboxes[idx]

            # Store new unknown
            if best_match.get("should_store") and best_match.get("is_unknown"):
                milvus_client.add_unknown(
                    unknown_id=best_match["name"],
                    embedding=embedding,
                )

            detections.append(
                Detection(
                    face_id=f"face_{idx:03d}",
                    name=best_match["name"],
                    confidence=float(confidences[idx]),
                    bounding_box=BoundingBox(
                        x1=int(bbox[0]),
                        y1=int(bbox[1]),
                        x2=int(bbox[2]),
                        y2=int(bbox[3]),
                    ),
                    similarity=best_match["similarity"],
                    distance=best_match["distance"],
                    face_image=None,
                )
            )
        # ---------------------------------
        # 6️⃣ Draw Bounding Boxes + Labels
        # ---------------------------------
        
        
        annotated_image = overlay_face_names(image, detections)
        annotated_base64 = encode_image(annotated_image)
        # milvus_client.flush_now()

        total_time = (time.time() - start_time) * 1000

        return PredictResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model,
            usage=Usage(
                detection_time_ms=detection_time,
                embedding_time_ms=0,
                search_time_ms=search_time,
                total_time_ms=total_time,
            ),
            detections=detections,
            image_info=ImageInfo(width=width, height=height, format="jpg"),
            annotated_image=annotated_base64
        )

    except Exception as e:
        logger.error("Request %s failed: %s", request_id, e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/v1/add_face", response_model=AddFaceResponse)
async def add_face(request: AddFaceRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail={"message":"Unauthorized","error":"Missing Authorization header"})
    try:
        api_key = verify_api_key(credentials.credentials)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail={"message":"Internal server error","error":str(e)})
    logger.info("Adding face for name: %s (extract_face=%s)", request.name, request.parameters.extract_face)
    try:
        face_crop = None

        if request.image.type != "base64":
            raise HTTPException(status_code=400, detail="Only base64 images supported")

        image = base64.b64decode(request.image.data)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        result = face_client.extract(image)
        embeddings = result['embeddings']
        
        if not result:
            raise HTTPException(status_code=400, detail="No face detected")

        # ---------------------------------
        # 2️⃣ Optional Face Crop Return
        # ---------------------------------
        if request.parameters.extract_face:

            bboxes = result["bboxes"]

            if len(bboxes) == 0:
                raise HTTPException(status_code=400, detail="No face detected")

            if len(bboxes) > 1:
                raise HTTPException(status_code=400, detail="Multiple faces detected. Provide single face image.")

            bbox = bboxes[0]  # first face

            x1, y1, x2, y2 = map(int, bbox)

            # Decode image properly
            np_arr = np.frombuffer(image, np.uint8)
            decoded_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if decoded_image is None:
                raise HTTPException(status_code=400, detail="Failed to decode image")

            crop = decoded_image[y1:y2, x1:x2]

            face_crop = encode_image(crop)
        embedding = embeddings[0]

        embeddings = result["embeddings"]

        if len(embeddings) != 1:
            raise HTTPException(
                status_code=400,
                detail="Image must contain exactly one face"
            )

        # Get numpy vector
        embedding = embeddings[0]

        # Ensure shape is (512,)
        embedding = np.array(embedding, dtype=np.float32).flatten()

        # Convert to python list
        embedding_list = embedding.tolist()

        # Wrap for Milvus batch insert
        embedding_batch = [embedding_list]


        print("here1")
        # ---------------------------------
        # 3️⃣ Add to Milvus
        # ---------------------------------
        face_id = generate_face_id(request.name)
        
        milvus_client.add_face(
            name=request.name,
            face_id=face_id,
            embedding=embedding_list
        )
        global train_flush
        if train_flush >=50:
            milvus_client.flush_now()
            train_flush = 0

        # print("Inserted norm:", np.linalg.norm(embeddings))
        print("here2")
        logger.info("Face added successfully: %s with ID %s",
                    request.name, face_id)

        return AddFaceResponse(
            success=True,
            message="Face added successfully",
            face_id=face_id,
            name=request.name,
            face_crop=face_crop
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add face for %s: %s", request.name, e)
        raise HTTPException(status_code=500, detail=f"Failed to add face: {str(e)}")

@app.delete("/v1/delete_face", response_model=DeleteFaceResponse)
async def delete_face(request: DeleteFaceRequest, _: str = Depends(verify_api_key)):
    logger.info("Deleting face for name: %s", request.name)
    try:
        # Get face IDs before deletion
        face_ids = milvus_client.get_face_ids_by_name(request.name)

        if not face_ids:
            raise HTTPException(status_code=404, detail=f"No faces found for name '{request.name}'")

        # Delete from Milvus
        deleted_count = milvus_client.delete_face(request.name)

        logger.info("Successfully deleted %d faces for name: %s", deleted_count, request.name)
        return DeleteFaceResponse(
            success=True,
            message=f"Successfully deleted {deleted_count} face instances for '{request.name}'",
            deleted_count=deleted_count,
            deleted_ids=face_ids
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete face for %s: %s", request.name, e)
        raise HTTPException(status_code=500, detail=f"Failed to delete face: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
