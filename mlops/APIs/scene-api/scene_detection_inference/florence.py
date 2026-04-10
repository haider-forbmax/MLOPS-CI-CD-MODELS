import requests
import json
from PIL import Image
import io
from scene_detection_inference.config import Config
# Configuration


def test_single_prediction(image , task: str = "<DETAILED_CAPTION>"):
    """Test single prediction endpoint"""
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        files = {'file': ('test.jpg', buffer, 'image/jpeg')}
        data = {
            'task': task,
            'max_new_tokens': 1024,
            'num_beams': 3
        }

        response = requests.post(f"{Config.API_URL}/predict", files=files, data=data)

        try:
            response.raise_for_status()
            return response.json().get("result")
        finally:
            response.close()