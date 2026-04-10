from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from config import Config
class ImageData(BaseModel):
    type: str  # "base64" or "url"
    data: Optional[str] = None
    url: Optional[str] = None

class PredictParameters(BaseModel):
    confidence: float = Config.PRED_CONFIDENCE
    nms_threshold: float = Config.PRED_NMS_THRESHOLD
    similarity_threshold: float = Config.PRED_SIMILARITY_THRESHOLD

class PredictResponseFormat(BaseModel):
    annotated_image: bool = True
    labeled_image: bool = False
    face_crops: bool = False

class PredictRequest(BaseModel):
    model: str = "face-recognition"
    image: ImageData
    parameters: Optional[PredictParameters] = PredictParameters()
    response_format: Optional[PredictResponseFormat] = PredictResponseFormat()

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class Detection(BaseModel):
    face_id: str = Field(..., serialization_alias="class_id")
    name: str = Field(..., serialization_alias="class_name")
    confidence: float
    bounding_box: BoundingBox
    similarity: float
    distance: float
    face_image: Optional[str] = Field(None, serialization_alias="object_image") 

class ImageInfo(BaseModel):
    width: int
    height: int
    format: str

class Usage(BaseModel):
    detection_time_ms: float = Field(..., serialization_alias="inference_time_ms")
    embedding_time_ms: float = Field(..., serialization_alias="preprocessing_time_ms")
    search_time_ms: float = Field(..., serialization_alias="postprocessing_time_ms")
    total_time_ms: float

class PredictResponse(BaseModel):
    id: str
    task: str = "face_recognition"
    created: int
    model: str
    usage: Usage
    detections: List[Detection]
    image_info: ImageInfo
    annotated_image: Optional[str] = None
    labeled_image: Optional[str] = None

class AddFaceParameters(BaseModel):
    extract_face: bool = True

class AddFaceRequest(BaseModel):
    name: str
    image: ImageData
    parameters: Optional[AddFaceParameters] = AddFaceParameters()

class AddFaceResponse(BaseModel):
    success: bool
    message: str
    face_id: str
    name: str
    face_crop: Optional[str] = None

class DeleteFaceRequest(BaseModel):
    name: str
    parameters: Optional[Dict[str, Any]] = {}

class DeleteFaceResponse(BaseModel):
    success: bool
    message: str
    deleted_count: int
    deleted_ids: List[str]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str = "1.0.0"

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    code: str