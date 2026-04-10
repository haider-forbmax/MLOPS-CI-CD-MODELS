# Scene Detection API

A FastAPI-based REST API for scene detection and image captioning using Microsoft's Florence-2 model.

## Features

- 🖼️ Image scene detection and detailed captioning
- 🔐 Bearer token authentication
- 📊 Performance metrics tracking
- 🌐 Support for both base64 and URL image inputs
- ⚡ Built with FastAPI for high performance

## Prerequisites

- Python 3.8+
- Access to Florence-2 model endpoint
- Valid API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd scene-detection-inference
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file from the example:
```bash
cp .env.example .env
```

4. Update your environment variables in `.env`:
```env
SCENE_API_KEY=your-api-key-here
FLORENCE_API_URL=https://your-florence-endpoint.com/
```

The application now loads configuration using `python-dotenv` from the project `.env` file.

## Progress - April 2, 2026

| Task | Date | Status |
| --- | --- | --- |
| Updated configuration to use `.env` with `load_dotenv`, added `.env.example`, added `python-dotenv`, and kept fallback defaults for compatibility | 2nd April 2026 | DONE |

## Usage

### Starting the Server

Run the API server:
```bash
python -m scene_detection_inference.api
```

The server will start on `http://0.0.0.0:2010`

### API Endpoints

#### 1. Scene Detection
**POST** `/v1/scene_detection`

Detect scenes and generate detailed captions for images.

**Headers:**
```
Authorization: Bearer <your-api-key>
Content-Type: application/json
```

**Request Body:**
```json
{
  "model": "microsoft/Florence-2-base",
  "image": {
    "type": "base64",
    "data": "<base64-encoded-image>"
  },
  "parameters": {
    "confidence": 0.5,
    "nms_threshold": 0.4
  },
  "response_format": {
    "labeled_image": false,
    "object_images": false,
    "annotated_image": false
  }
}
```

**Or with URL:**
```json
{
  "model": "microsoft/Florence-2-base",
  "image": {
    "type": "url",
    "url": "https://example.com/image.jpg"
  }
}
```

**Response:**
```json
{
  "id": "detect_scene_a1b2c3d4",
  "task": "scene_detection",
  "created": 1699876543,
  "model": "microsoft/Florence-2-base",
  "usage": {
    "inference_time_ms": 250.5,
    "preprocessing_time_ms": 45.2,
    "postprocessing_time_ms": 12.3,
    "total_time_ms": 308.0
  },
  "detections": [
    {
      "class_id": null,
      "class_name": "A detailed caption describing the scene",
      "confidence": null,
      "bounding_box": {
        "x1": 0,
        "y1": 0,
        "x2": 0,
        "y2": 0
      },
      "object_image": null
    }
  ],
  "image_info": {
    "width": 1920,
    "height": 1080,
    "format": "JPEG"
  },
  "labeled_image": null,
  "annotated_image": null
}
```

#### 2. Health Check
**GET** `/health`

Check if the Florence-2 backend is accessible.

**Response:**
```json
{
  "status": "healthy"
}
```

#### 3. Liveness Check
**GET** `/live`

Check if the API server is running.

**Response:**
```json
{
  "live": true
}
```

## Example Usage

### Python Example

```python
import requests
import base64

# Read and encode image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# API request
url = "http://localhost:2010/v1/scene_detection"
headers = {
    "Authorization": "Bearer 123",
    "Content-Type": "application/json"
}
payload = {
    "model": "microsoft/Florence-2-base",
    "image": {
        "type": "base64",
        "data": image_data
    }
}

response = requests.post(url, json=payload, headers=headers)
result = response.json()
print(result["detections"][0]["class_name"])
```

### cURL Example

```bash
curl -X POST "http://localhost:2010/v1/scene_detection" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/Florence-2-base",
    "image": {
      "type": "url",
      "url": "https://example.com/image.jpg"
    }
  }'
```

## Project Structure

```
scene_detection_inference/
├── __init__.py
├── api.py              # Main FastAPI application
├── config.py           # Configuration settings
├── florence.py         # Florence-2 model integration
├── schemas.py          # Pydantic models
└── utils.py            # Helper functions
```
