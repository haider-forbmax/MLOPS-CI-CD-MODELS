# YOLO11 Object Detection API

A production-ready REST API for real-time object detection using YOLOv11 models powered by NVIDIA Triton Inference Server. This API provides high-performance object detection capabilities for 80 COCO classes with enterprise-grade reliability and scalability.

## 🎯 API Overview

**Base URL**: `https://yolo11-dgx.nimar.gov.pk`
**Model**: YOLOv11n (Nano) - Optimized for speed and accuracy
**Classes**: 80 COCO dataset classes
**Input**: Images (JPEG, PNG, JPG, WebP)
**Output**: Bounding boxes, confidence scores, class labels

## 🏗️ System Architecture

```
Client Application
       ↓
NGINX Ingress (yolo11-dgx.nimar.gov.pk)
       ↓
Kubernetes Service (Load Balancer)
       ↓
YOLO11 API Pods (2-10 replicas)
       ↓
NVIDIA Triton Inference Server
       ↓
YOLOv11n Model
```

## 🎯 Supported Object Classes (COCO Dataset)

The YOLO11 API can detect 80 different object classes:

| Class ID | Object        | Class ID | Object         | Class ID | Object       | Class ID | Object       |
| -------- | ------------- | -------- | -------------- | -------- | ------------ | -------- | ------------ |
| 0        | person        | 20       | elephant       | 40       | wine glass   | 60       | dining table |
| 1        | bicycle       | 21       | bear           | 41       | cup          | 61       | toilet       |
| 2        | car           | 22       | zebra          | 42       | fork         | 62       | tv           |
| 3        | motorcycle    | 23       | giraffe        | 43       | knife        | 63       | laptop       |
| 4        | airplane      | 24       | backpack       | 44       | spoon        | 64       | mouse        |
| 5        | bus           | 25       | umbrella       | 45       | bowl         | 65       | remote       |
| 6        | train         | 26       | handbag        | 46       | banana       | 66       | keyboard     |
| 7        | truck         | 27       | tie            | 47       | apple        | 67       | cell phone   |
| 8        | boat          | 28       | suitcase       | 48       | sandwich     | 68       | microwave    |
| 9        | traffic light | 29       | frisbee        | 49       | orange       | 69       | oven         |
| 10       | fire hydrant  | 30       | skis           | 50       | broccoli     | 70       | toaster      |
| 11       | stop sign     | 31       | snowboard      | 51       | carrot       | 71       | sink         |
| 12       | parking meter | 32       | sports ball    | 52       | hot dog      | 72       | refrigerator |
| 13       | bench         | 33       | kite           | 53       | pizza        | 73       | book         |
| 14       | bird          | 34       | baseball bat   | 54       | donut        | 74       | clock        |
| 15       | cat           | 35       | baseball glove | 55       | cake         | 75       | vase         |
| 16       | dog           | 36       | skateboard     | 56       | chair        | 76       | scissors     |
| 17       | horse         | 37       | surfboard      | 57       | couch        | 77       | teddy bear   |
| 18       | sheep         | 38       | tennis racket  | 58       | potted plant | 78       | hair drier   |
| 19       | cow           | 39       | bottle         | 59       | bed          | 79       | toothbrush   |

## 🔗 Quick Start Integration

### 1. Basic Detection Request

```bash
curl -X POST "https://yolo11-dgx.nimar.gov.pk/v1/detect" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo11",
    "image": {
      "type": "base64",
      "data": "<your_base64_encoded_image>"
    },
    "parameters": {
      "confidence": 0.25,
      "nms_threshold": 0.45
    },
    "response_format": {
      "labeled_image": false,
      "object_images": true,
      "annotated_image": false
    }
  }'
```

### 2. Image URL Detection

```bash
curl -X POST "https://yolo11-dgx.nimar.gov.pk/v1/detect" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo11",
    "image": {
      "type": "url",
      "url": "https://example.com/image.jpg"
    }
  }'
```

### 3. Get Available Classes

```bash
curl -X GET "https://yolo11-dgx.nimar.gov.pk/classes"
```

### 4. Health Check

```bash
curl -X GET "https://yolo11-dgx.nimar.gov.pk/health"
```

## 📝 API Reference

### Authentication

**Required for detection endpoint only**

```http
Authorization: Bearer 123
```

### POST /v1/detect - Object Detection

**Request Body Schema:**

```json
{
  "model": "yolo11",                    // Required: Model name
  "image": {
    "type": "base64|url",              // Required: Image input type
    "data": "<base64_string>",         // Required if type=base64
    "url": "https://example.com/img"   // Required if type=url
  },
  "parameters": {                      // Optional: Detection parameters
    "confidence": 0.25,                // Default: 0.25, Range: 0.0-1.0
    "nms_threshold": 0.45              // Default: 0.45, Range: 0.0-1.0
  },
  "response_format": {                 // Optional: Response format
    "labeled_image": false,            // Include image with labels
    "object_images": true,             // Include cropped objects
    "annotated_image": false           // Include image with boxes only
  }
}
```

**Response Schema:**

```json
{
  "id": "detect_yolo11_abc123",        // Unique request ID
  "created": 1694123456,                // Unix timestamp
  "model": "yolo11",                   // Model used
  "task": "detection",                 // Task type
  "usage": {
    "inference_time_ms": 45.67,        // Model inference time
    "preprocessing_time_ms": 12.34,    // Image preprocessing time
    "postprocessing_time_ms": 8.90,    // Result postprocessing time
    "total_time_ms": 66.91             // Total processing time
  },
  "detections": [                      // Array of detected objects
    {
      "class_id": 0,                   // COCO class ID (0-79)
      "class_name": "person",          // Human-readable class name
      "confidence": 0.89,              // Detection confidence (0.0-1.0)
      "bounding_box": {
        "x1": 100,                     // Top-left X coordinate
        "y1": 150,                     // Top-left Y coordinate
        "x2": 300,                     // Bottom-right X coordinate
        "y2": 450                      // Bottom-right Y coordinate
      },
      "object_image": "<base64>"       // Cropped object (if requested)
    }
  ],
  "image_info": {
    "width": 640,                       // Original image width
    "height": 480,                      // Original image height
    "format": "JPEG"                    // Image format
  },
  "labeled_image": "<base64>",         // Full image with labels (if requested)
  "annotated_image": "<base64>"        // Full image with boxes (if requested)
}
```

### GET /health - Service Health

**Response:**

```json
{
  "status": "healthy",
  "service": "YOLO11 Object Detection API",
  "triton_server": "healthy",
  "model_status": "healthy"
}
```

### GET /classes - Available Classes

**Response:**

```json
{
  "classes": {
    "0": "person",
    "1": "bicycle",
    // ... all 80 COCO classes
  }
}
```

### GET /config - API Configuration

**Response:**

```json
{
  "triton_server_url": "triton-yolo11-service.models-inference.svc.cluster.local:8000",
  "model_name": "yolo11",
  "default_confidence_threshold": 0.25,
  "default_nms_threshold": 0.45,
  "max_image_size": 10485760,
  "supported_formats": ["jpeg", "png", "jpg", "webp"],
  "task_type": "detect"
}
```

## 💻 Code Examples

### Python Integration

```python
import requests
import base64
import json

class YOLO11Client:
    def __init__(self, base_url="https://yolo11-dgx.nimar.gov.pk", api_key="123"):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def detect_from_file(self, image_path, confidence=0.25, nms_threshold=0.45):
        """Detect objects in local image file"""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        payload = {
            "model": "yolo11",
            "image": {
                "type": "base64",
                "data": encoded_string
            },
            "parameters": {
                "confidence": confidence,
                "nms_threshold": nms_threshold
            },
            "response_format": {
                "labeled_image": False,
                "object_images": True,
                "annotated_image": False
            }
        }

        response = requests.post(
            f"{self.base_url}/v1/detect",
            headers=self.headers,
            data=json.dumps(payload)
        )

        return response.json()

    def detect_from_url(self, image_url, confidence=0.25):
        """Detect objects in image from URL"""
        payload = {
            "model": "yolo11",
            "image": {
                "type": "url",
                "url": image_url
            },
            "parameters": {
                "confidence": confidence
            }
        }

        response = requests.post(
            f"{self.base_url}/v1/detect",
            headers=self.headers,
            data=json.dumps(payload)
        )

        return response.json()

    def health_check(self):
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Usage example
client = YOLO11Client()

# Detect from local file
result = client.detect_from_file("path/to/image.jpg", confidence=0.3)
print(f"Found {len(result['detections'])} objects")

# Detect from URL
result = client.detect_from_url("https://example.com/image.jpg")
for detection in result['detections']:
    print(f"{detection['class_name']}: {detection['confidence']:.2f}")
```

### cURL Examples

```bash
# Basic detection with base64 image
curl -X POST "https://yolo11-dgx.nimar.gov.pk/v1/detect" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "yolo11",
  "image": {
    "type": "base64",
    "data": "$(base64 -w 0 image.jpg)"
  },
  "parameters": {
    "confidence": 0.3,
    "nms_threshold": 0.5
  },
  "response_format": {
    "labeled_image": true,
    "object_images": false,
    "annotated_image": true
  }
}
EOF

# Detection from URL
curl -X POST "https://yolo11-dgx.nimar.gov.pk/v1/detect" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo11",
    "image": {
      "type": "url",
      "url": "https://ultralytics.com/images/bus.jpg"
    }
  }'

# Get available classes
curl -X GET "https://yolo11-dgx.nimar.gov.pk/classes"

# Health check
curl -X GET "https://yolo11-dgx.nimar.gov.pk/health"

# Get configuration
curl -X GET "https://yolo11-dgx.nimar.gov.pk/config"
```

## ⚙️ Configuration Parameters

### Detection Parameters

- **confidence**: Detection confidence threshold (0.0-1.0, default: 0.25)
  - Lower values = more detections (including false positives)
  - Higher values = fewer, more confident detections
- **nms_threshold**: Non-Maximum Suppression threshold (0.0-1.0, default: 0.45)
  - Lower values = more aggressive overlap removal
  - Higher values = keep more overlapping boxes

### Image Constraints

- **Maximum size**: 10MB
- **Supported formats**: JPEG, PNG, JPG, WebP
- **Minimum dimensions**: 32x32 pixels
- **Maximum dimensions**: 4096x4096 pixels
- **Input processing**: Images are resized to 640x640 for inference

### Response Options

- **object_images**: Get cropped images of detected objects (base64)
- **labeled_image**: Get full image with class labels and confidence scores
- **annotated_image**: Get full image with bounding boxes only

## 🚀 Performance & Scaling

### API Performance

- **Average Response Time**: 50-150ms per image
- **Throughput**: 100+ requests/minute per pod
- **Auto-scaling**: 2-10 pods based on load
- **Load Balancing**: Round-robin across healthy pods

## 🔧 Error Handling

### Common Error Codes

- **400**: Invalid request format or parameters
- **401**: Missing or invalid API key
- **413**: Image too large (> 10MB)
- **422**: Invalid image format or corrupted image
- **503**: Service temporarily unavailable

### Error Response Format

```json
{
  "detail": "Image size exceeds maximum allowed size of 10485760 bytes"
}
```

## 📋 API Endpoints Summary

| Endpoint          | Method | Description              | Authentication |
| ----------------- | ------ | ------------------------ | -------------- |
| `POST /v1/detect` | POST   | Object detection         | Required       |
| `GET /health`     | GET    | Service health status    | None           |
| `GET /classes`    | GET    | Available object classes | None           |
| `GET /config`     | GET    | API configuration        | None           |
| `GET /docs`       | GET    | Interactive API docs     | None           |
| `GET /`           | GET    | API information          | None           |

## 🏥 System Status

**Current Deployment:**

- **Environment**: Production
- **URL**: https://yolo11-dgx.nimar.gov.pk
- **Model Version**: YOLOv11n
- **API Key**: `123` (Demo key - request production key)
- **Uptime**: 99.9% SLA

**Infrastructure:**

- **Kubernetes Cluster**: Production cluster
- **Namespace**: `yolo11-api`
- **Replicas**: 2-10 (auto-scaling)
- **Resources**: 512Mi-2Gi RAM, 250m-1000m CPU per pod
- **Ingress**: NGINX with SSL termination

---

*This API is designed for production use with enterprise-grade reliability. For support or production API keys, contact the MLOps team.*