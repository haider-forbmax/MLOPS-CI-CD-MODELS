from pydantic import BaseModel
from typing import Optional

class ImageInput(BaseModel):
    type: str  # "url" or "base64"
    url: Optional[str] = None
    data: Optional[str] = None

class EmbeddingRequest(BaseModel):
    model: str
    image: ImageInput
    include_embeddings: bool = True  # Optional flag to reduce payload size

class Usage(BaseModel):
    inference_time_ms: float
    total_time_ms: float

class ImageInfo(BaseModel):
    width: int
    height: int
    format: str

class EmbeddingResponse(BaseModel):
    id: str
    task: str
    created: int
    model: str
    image_info: ImageInfo
    usage: Usage
    embeddings: Optional[list] = None