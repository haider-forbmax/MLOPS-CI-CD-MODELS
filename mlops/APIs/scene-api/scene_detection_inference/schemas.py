from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
# from config import Config

class Usage(BaseModel):
    inference_time_ms: float
    preprocessing_time_ms: float 
    postprocessing_time_ms: float 
    total_time_ms: float

class DetectionParameters(BaseModel):
    confidence: Optional[float] = None
    nms_threshold: Optional[float] = None

class ResponseFormat(BaseModel):
    labeled_image: bool = False
    object_images: bool = False
    annotated_image: bool = False

class ImageData(BaseModel):
    type: str  # "base64" or "url"
    data: Optional[str] = None
    url: Optional[str] = None

class BoundingBox(BaseModel):
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0

class Detection(BaseModel):
    class_id: str =None
    class_name: str
    confidence: Optional[float] = None
    bounding_box: Optional[BoundingBox] = BoundingBox()
    object_image: Optional[str] = None

class ImageInfo(BaseModel):
    width: int
    height: int
    format: str

class DetectionResponse(BaseModel):
    id: str = None
    task: str = "scene_detection"
    created: int
    model: str
    usage: Usage
    detections:list[Detection]
    image_info: ImageInfo
    labeled_image: Optional[str] = None
    annotated_image: Optional[str] = None

class DetectionRequest(BaseModel):
    # model: str = Config.TRITON_MODEL_NAME
    model : str = "microsoft/Florence-2-base"
    image: ImageData
    parameters: Optional[DetectionParameters] = DetectionParameters()
    response_format: Optional[ResponseFormat] = ResponseFormat() 






