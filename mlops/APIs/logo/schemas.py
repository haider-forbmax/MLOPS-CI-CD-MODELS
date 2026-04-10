from pydantic import BaseModel
from typing import List, Optional, Dict
from config import Config

config = Config()

# Pydantic models for request/response
class ImageInput(BaseModel):
    type: str  # "base64" or "url"
    data: Optional[str] = None
    url: Optional[str] = None

class DetectionParameters(BaseModel):
    confidence: float = config.DEFAULT_CONFIDENCE_THRESHOLD
    nms_threshold: float = config.DEFAULT_NMS_THRESHOLD

class ResponseFormat(BaseModel):
    labeled_image: bool = False
    object_images: bool = True
    annotated_image: bool = False

class DetectionRequest(BaseModel):
    model: str = config.TRITON_MODEL_NAME
    image: ImageInput
    parameters: Optional[DetectionParameters] = DetectionParameters()
    response_format: Optional[ResponseFormat] = ResponseFormat()

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class Detection(BaseModel):
    class_id: str
    class_name: str
    confidence: float
    bounding_box: BoundingBox
    object_image: Optional[str] = None

class ImageInfo(BaseModel):
    width: int
    height: int
    format: str

class Usage(BaseModel):
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
    total_time_ms: float

class DetectionResponse(BaseModel):
    id: str
    task: str = "detection"
    created: int
    model: str
    usage: Usage
    detections: List[Detection]
    image_info: ImageInfo
    labeled_image: Optional[str] = None
    annotated_image: Optional[str] = None

class ModelInfo(BaseModel):
    name: str
    input_name: str
    output_name: str
    input_shape: List[int]
    output_shape: List[int]
    status: str
    classes: Optional[Dict[str, str]] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    triton_server: str
    model_status: str
