# YOLOv5 Face Detection API

A production-ready REST API for real-time face detection using YOLOv5 models powered by NVIDIA Triton Inference Server. This API provides high-performance face detection capabilities with enterprise-grade reliability and scalability.

## 🎯 API Overview

**Base URL**: `https://face-dgx.nimar.gov.pk`
**Model**: YOLOv5m-face - Optimized for face detection accuracy
**Classes**: Face detection (single class)
**Input**: Images (JPEG, PNG, JPG, WebP)
**Output**: Face bounding boxes, confidence scores

## 🏗️ System Architecture

```
Client Application
       ↓
NGINX Ingress (face-dgx.nimar.gov.pk)
       ↓
Kubernetes Service (Load Balancer)
       ↓
YOLOv5 Face API Pods (2-10 replicas)
       ↓
NVIDIA Triton Inference Server
       ↓
YOLOv5m-face Model
```

## 🎯 Supported Detection Class

The YOLOv5 Face API detects faces with high accuracy:

| Class ID | Object |
| -------- | ------ |
| 0        | face   |

## 🔗 Quick Start Integration

### 1. Basic Detection Request

```bash
curl -X POST "https://face-dgx.nimar.gov.pk/v1/detect" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolov5m-face",
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
curl -X POST "https://face-dgx.nimar.gov.pk/v1/detect" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolov5m-face",
    "image": {
      "type": "url",
      "url": "https://example.com/image.jpg"
    }
  }'
```

### 3. Get Available Classes

```bash
curl -X GET "https://face-dgx.nimar.gov.pk/classes"
```

### 4. Health Check

```bash
curl -X GET "https://face-dgx.nimar.gov.pk/health"
```

## 📝 API Reference

### Authentication

**Required for detection endpoint only**

```http
Authorization: Bearer 123
```

### POST /v1/detect - Face Detection

**Request Body Schema:**

```json
{
  "model": "yolov5m-face",             // Required: Model name
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
  "id": "detect_yolov5_face_abc123",   // Unique request ID
  "created": 1694123456,                // Unix timestamp
  "model": "yolov5m-face",             // Model used
  "task": "face_detection",             // Task type
  "usage": {
    "inference_time_ms": 45.67,        // Model inference time
    "preprocessing_time_ms": 12.34,    // Image preprocessing time
    "postprocessing_time_ms": 8.90,    // Result postprocessing time
    "total_time_ms": 66.91             // Total processing time
  },
  "detections": [                      // Array of detected faces
    {
      "class_id": 0,                   // Face class ID (0)
      "class_name": "face",            // Human-readable class name
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
  "service": "YOLOv5 Face Detection API",
  "triton_server": "healthy",
  "model_status": "healthy"
}
```

### GET /classes - Available Classes

**Response:**

```json
{
  "classes": {
    "0": "face"
  }
}
```

### GET /config - API Configuration

**Response:**

```json
{
  "triton_server_url": "triton-yolo-face-service.models-inference.svc.cluster.local:8000",
  "model_name": "yolov5m-face",
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

class YOLOv5FaceClient:
    def __init__(self, base_url="https://face-dgx.nimar.gov.pk", api_key="123"):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def detect_from_file(self, image_path, confidence=0.25, nms_threshold=0.45):
        """Detect faces in local image file"""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        payload = {
            "model": "yolov5m-face",
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
        """Detect faces in image from URL"""
        payload = {
            "model": "yolov5m-face",
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
client = YOLOv5FaceClient()

# Detect from local file
result = client.detect_from_file("path/to/image.jpg", confidence=0.3)
print(f"Found {len(result['detections'])} faces")

# Detect from URL
result = client.detect_from_url("https://example.com/image.jpg")
for detection in result['detections']:
    print(f"{detection['class_name']}: {detection['confidence']:.2f}")
```

### cURL Examples

```bash
# Basic detection with base64 image
curl -X POST "https://face-dgx.nimar.gov.pk/v1/detect" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "yolov5m-face",
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
curl -X POST "https://face-dgx.nimar.gov.pk/v1/detect" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolov5m-face",
    "image": {
      "type": "url",
      "url": "https://ultralytics.com/images/bus.jpg"
    }
  }'

# Get available classes
curl -X GET "https://face-dgx.nimar.gov.pk/classes"

# Health check
curl -X GET "https://face-dgx.nimar.gov.pk/health"

# Get configuration
curl -X GET "https://face-dgx.nimar.gov.pk/config"
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

- **object_images**: Get cropped images of detected faces (base64)
- **labeled_image**: Get full image with face labels and confidence scores
- **annotated_image**: Get full image with face bounding boxes only

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

| Endpoint          | Method | Description            | Authentication |
| ----------------- | ------ | ---------------------- | -------------- |
| `POST /v1/detect` | POST   | Face detection         | Required       |
| `GET /health`     | GET    | Service health status  | None           |
| `GET /classes`    | GET    | Available face classes | None           |
| `GET /config`     | GET    | API configuration      | None           |
| `GET /docs`       | GET    | Interactive API docs   | None           |
| `GET /`           | GET    | API information        | None           |

## 🏥 System Status

**Current Deployment:**

- **Environment**: Production
- **URL**: https://face-dgx.nimar.gov.pk
- **Model Version**: YOLOv5m-face
- **API Key**: `123` (Demo key - request production key)
- **Uptime**: 99.9% SLA

**Infrastructure:**

- **Kubernetes Cluster**: Production cluster
- **Namespace**: `face-detection-api`
- **Replicas**: 2-10 (auto-scaling)
- **Resources**: 512Mi-2Gi RAM, 250m-1000m CPU per pod
- **Ingress**: NGINX with SSL termination

---

*This API is designed for production use with enterprise-grade reliability. For support or production API keys, contact the MLOps team.*