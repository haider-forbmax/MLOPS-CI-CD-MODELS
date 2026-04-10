# Face Recognition API
# test for the github actions
A production-ready REST API for real-time face detection and recognition using YOLOv5 face detection, ResNet face embeddings, and Milvus vector database. This API provides high-performance face recognition capabilities with enterprise-grade reliability and scalability.

## 🎯 API Overview

**Base URL**: `https://fr-dgx.nimar.gov.pk`
**Model**: Integrated face recognition pipeline (YOLO + ResNet + Milvus)
**Classes**: Face detection and recognition with person identification
**Input**: Images (JPEG, PNG, JPG, WebP)
**Output**: Face bounding boxes, confidence scores, face names, similarity scores

## 🏗️ System Architecture

```
Client Application
       ↓
NGINX Ingress (fr-dgx.nimar.gov.pk)
       ↓
Kubernetes Service (Load Balancer)
       ↓
Face Recognition API Pods
       ↓
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   YOLO Service  │    │  ResNet Service │    │ Milvus Database │
│ (Face Detection)│    │ (Face Embeddings│    │ (Vector Search) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎯 Supported Recognition Classes

The Face Recognition API detects faces and identifies registered persons:

| Class ID | Object Type  | Description                                    |
| -------- | ------------ | ---------------------------------------------- |
| Dynamic  | Person Names | Known individuals (John Doe, Jane Smith, etc.) |
| Unknown  | unknown      | Unrecognized faces                             |

## 🔗 Quick Start Integration

### 1. Face Recognition Request

```bash
curl -X POST "https://fr-dgx.nimar.gov.pk/v1/predict" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "face-recognition",
    "image": {
      "type": "base64",
      "data": "<your_base64_encoded_image>"
    },
    "parameters": {
      "confidence": 0.5,
      "nms_threshold": 0.4,
      "similarity_threshold": 0.7
    },
    "response_format": {
      "annotated_image": true,
      "labeled_image": true,
      "face_crops": true
    }
  }'
```

### 2. Add New Face

```bash
curl -X POST "https://fr-dgx.nimar.gov.pk/v1/add_face" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "image": {
      "type": "base64",
      "data": "<your_base64_encoded_image>"
    },
    "parameters": {
      "extract_face": true
    }
  }'
```

### 3. Delete Face

```bash
curl -X DELETE "https://fr-dgx.nimar.gov.pk/v1/delete_face" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe"
  }'
```

### 4. Health Check

```bash
curl -X GET "https://fr-dgx.nimar.gov.pk/health"
```

## 📝 API Reference

### Authentication

**Required for all protected endpoints**

```http
Authorization: Bearer 123
```

### POST /v1/predict - Face Recognition

**Request Body Schema:**

```json
{
  "model": "face-recognition",             // Required: Model name
  "image": {
    "type": "base64|url",                  // Required: Image input type
    "data": "<base64_string>",             // Required if type=base64
    "url": "https://example.com/img"       // Required if type=url
  },
  "parameters": {                          // Optional: Recognition parameters
    "confidence": 0.5,                     // Default: 0.5, Range: 0.0-1.0
    "nms_threshold": 0.4,                  // Default: 0.4, Range: 0.0-1.0
    "similarity_threshold": 0.7            // Default: 0.7, Range: 0.0-1.0
  },
  "response_format": {                     // Optional: Response format
    "annotated_image": true,               // Include image with bounding boxes
    "labeled_image": false,                // Include image with face names
    "face_crops": false                    // Include individual face crops
  }
}
```

**Response Schema:**

```json
{
  "id": "pred_face_abc123",                // Unique request ID
  "created": 1694123456,                   // Unix timestamp
  "model": "face-recognition",             // Model used
  "task": "face_recognition",              // Task type
  "usage": {
    "detection_time_ms": 45.67,            // Face detection time
    "embedding_time_ms": 89.12,            // Face embedding time
    "search_time_ms": 12.34,               // Vector search time
    "total_time_ms": 147.13                // Total processing time
  },
  "detections": [                          // Array of recognized faces
    {
      "face_id": "face_001",               // Face identifier
      "name": "John Doe",                  // Recognized person name or "unknown"
      "confidence": 0.95,                  // Detection confidence (0.0-1.0)
      "bounding_box": {
        "x1": 120,                         // Top-left X coordinate
        "y1": 80,                          // Top-left Y coordinate
        "x2": 220,                         // Bottom-right X coordinate
        "y2": 200                          // Bottom-right Y coordinate
      },
      "similarity": 0.89,                  // Recognition similarity (0.0-1.0)
      "distance": 0.11,                    // Vector distance (lower = more similar)
      "face_image": "<base64>"             // Face crop (if requested)
    }
  ],
  "image_info": {
    "width": 640,                          // Original image width
    "height": 480,                         // Original image height
    "format": "JPEG"                       // Image format
  },
  "annotated_image": "<base64>",           // Image with bounding boxes (if requested)
  "labeled_image": "<base64>"              // Image with face names (if requested)
}
```

### POST /v1/add_face - Add New Face

**Request Body Schema:**

```json
{
  "name": "Jane Smith",                    // Required: Person name
  "image": {
    "type": "base64|url",                  // Required: Image input type
    "data": "<base64_string>",             // Required if type=base64
    "url": "https://example.com/img"       // Required if type=url
  },
  "parameters": {                          // Optional: Parameters
    "extract_face": true                   // Extract face using YOLO (default: true)
  }
}
```

**Response Schema:**

```json
{
  "success": true,
  "message": "Face added successfully",
  "face_id": "face_jane_smith_001",
  "name": "Jane Smith",
  "face_crop": "<base64>"                  // Face crop (if extract_face=true)
}
```

### DELETE /v1/delete_face - Remove Face

**Request Body Schema:**

```json
{
  "name": "Jane Smith"                     // Required: Person name to delete
}
```

**Response Schema:**

```json
{
  "success": true,
  "message": "Successfully deleted 3 face instances for 'Jane Smith'",
  "deleted_count": 3,
  "deleted_ids": ["face_jane_smith_001", "face_jane_smith_002", "face_jane_smith_003"]
}
```

### GET /health - Service Health

**Response:**

```json
{
  "status": "healthy",                     // Overall status: healthy/degraded
  "timestamp": "2025-09-26T12:00:00.000Z",
  "version": "1.0.0",
  "services": {                            // Individual service status
    "yolo": "healthy",
    "resnet": "healthy",
    "milvus": "healthy"
  }
}
```

### GET /config - API Configuration

**Response:**

```json
{
  "face_detection": {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4
  },
  "face_recognition": {
    "similarity_threshold": 0.7,
    "max_faces_per_image": 10,
    "embedding_model": "resnet100"
  },
  "database": {
    "collection_name": "face_embeddings",
    "vector_dimension": 512
  },
  "services": {
    "yolo_endpoint": "http://yolo-service:8000/v1/detect",
    "resnet_endpoint": "http://resnet-service:8000/v1/embeddings",
    "milvus_endpoint": "milvus-standalone:19530"
  }
}
```

## 💻 Code Examples

### Python Integration

```python
import requests
import base64
import json

class FaceRecognitionClient:
    def __init__(self, base_url="https://fr-dgx.nimar.gov.pk", api_key="123"):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def recognize_faces(self, image_path, similarity_threshold=0.7):
        """Recognize faces in local image file"""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        payload = {
            "model": "face-recognition",
            "image": {
                "type": "base64",
                "data": encoded_string
            },
            "parameters": {
                "confidence": 0.5,
                "similarity_threshold": similarity_threshold
            },
            "response_format": {
                "annotated_image": True,
                "labeled_image": True,
                "face_crops": True
            }
        }

        response = requests.post(
            f"{self.base_url}/v1/predict",
            headers=self.headers,
            data=json.dumps(payload)
        )

        return response.json()

    def add_person(self, name, image_path):
        """Add a new person to the face database"""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        payload = {
            "name": name,
            "image": {
                "type": "base64",
                "data": encoded_string
            },
            "parameters": {
                "extract_face": True
            }
        }

        response = requests.post(
            f"{self.base_url}/v1/add_face",
            headers=self.headers,
            data=json.dumps(payload)
        )

        return response.json()

    def delete_person(self, name):
        """Delete a person from the face database"""
        payload = {"name": name}

        response = requests.delete(
            f"{self.base_url}/v1/delete_face",
            headers=self.headers,
            data=json.dumps(payload)
        )

        return response.json()

    def health_check(self):
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Usage example
client = FaceRecognitionClient()

# Add a new person
result = client.add_person("John Doe", "path/to/john_photo.jpg")
print(f"Added: {result['message']}")

# Recognize faces in an image
result = client.recognize_faces("path/to/group_photo.jpg")
print(f"Found {len(result['detections'])} faces:")
for detection in result['detections']:
    print(f"- {detection['name']}: {detection['similarity']:.2f} similarity")

# Delete a person
result = client.delete_person("John Doe")
print(f"Deleted: {result['message']}")
```

### cURL Examples

```bash
# Face recognition with all response formats
curl -X POST "https://fr-dgx.nimar.gov.pk/v1/predict" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "face-recognition",
    "image": {
      "type": "base64",
      "data": "'$(base64 -w 0 image.jpg)'"
    },
    "parameters": {
      "confidence": 0.5,
      "similarity_threshold": 0.8
    },
    "response_format": {
      "annotated_image": true,
      "labeled_image": true,
      "face_crops": true
    }
  }'

# Add face from local file
curl -X POST "https://fr-dgx.nimar.gov.pk/v1/add_face" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Alice Johnson",
    "image": {
      "type": "base64",
      "data": "'$(base64 -w 0 alice.jpg)'"
    },
    "parameters": {
      "extract_face": true
    }
  }'

# Delete face by name
curl -X DELETE "https://fr-dgx.nimar.gov.pk/v1/delete_face" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Alice Johnson"
  }'

# Health check
curl -X GET "https://fr-dgx.nimar.gov.pk/health"

# Get configuration
curl -X GET "https://fr-dgx.nimar.gov.pk/config"
```

## ⚙️ Configuration Parameters

### Recognition Parameters

- **confidence**: Face detection confidence threshold (0.0-1.0, default: 0.5)
  - Lower values = detect more faces (including false positives)
  - Higher values = fewer, more confident detections
- **nms_threshold**: Non-Maximum Suppression threshold (0.0-1.0, default: 0.4)
  - Lower values = more aggressive overlap removal
  - Higher values = keep more overlapping boxes
- **similarity_threshold**: Face recognition similarity threshold (0.0-1.0, default: 0.7)
  - Lower values = more lenient matching (more false positives)
  - Higher values = stricter matching (more unknowns)

### Image Constraints

- **Maximum size**: 10MB
- **Supported formats**: JPEG, PNG, JPG, WebP
- **Minimum dimensions**: 32x32 pixels
- **Maximum dimensions**: 4096x4096 pixels
- **Processing**: Images are automatically resized for optimal recognition

### Response Options

- **face_crops**: Include individual face crop images in detections (base64)
- **annotated_image**: Get full image with face bounding boxes
- **labeled_image**: Get full image with face names and confidence scores

## 🚀 Performance & Scaling

### API Performance

- **Average Response Time**: 150-300ms per image (depending on face count)
- **Throughput**: 50+ requests/minute per pod
- **Auto-scaling**: 2-10 pods based on load
- **Load Balancing**: Round-robin across healthy pods

### Face Database

- **Vector Storage**: Milvus vector database for fast similarity search
- **Search Performance**: Sub-10ms vector search for thousands of faces
- **Scalability**: Supports millions of face embeddings
- **Accuracy**: >95% recognition accuracy with proper face quality

## 🔧 Error Handling

### Common Error Codes

- **400**: Invalid request format, missing model name, or invalid parameters
- **401**: Missing or invalid API key
- **404**: Face name not found (delete operations)
- **413**: Image too large (> 10MB)
- **422**: Invalid image format or no face detected
- **503**: Service temporarily unavailable (external services down)

### Error Response Format

```json
{
  "success": false,
  "error": "No face detected in image",
  "code": "NO_FACE_DETECTED"
}
```

## 📋 API Endpoints Summary

| Endpoint                 | Method | Description               | Authentication |
| ------------------------ | ------ | ------------------------- | -------------- |
| `POST /v1/predict`       | POST   | Face recognition          | Required       |
| `POST /v1/add_face`      | POST   | Add new face to database  | Required       |
| `DELETE /v1/delete_face` | DELETE | Remove face from database | Required       |
| `GET /health`            | GET    | Service health status     | None           |
| `GET /config`            | GET    | API configuration         | None           |

## 🏥 System Status

**Current Deployment:**

- **Environment**: Production Ready
- **Model Pipeline**: YOLO + ResNet + Milvus
- **API Key**: `123` (Demo key - configure production key)
- **Vector Database**: Milvus with COSINE similarity
- **Embedding Model**: ResNet100 (512-dimensional vectors)

---

*This Face Recognition API provides face detection and recognition capabilities. For any issue or custom API keys, contact the MLOps team.*
