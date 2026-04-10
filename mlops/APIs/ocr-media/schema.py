from pydantic import BaseModel, Field

class OCRRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image string (required)")


class OCRResponse(BaseModel):
    raw_text: str = Field(..., description="Extracted raww text content from the image")
    markdown_text : str = Field(..., description="Extracted markdown content from the image")