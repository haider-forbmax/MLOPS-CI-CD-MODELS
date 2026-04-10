# YOLO11 Logo Detection API

A production-ready REST API for real-time logo detection using YOLOv11 models powered by NVIDIA Triton Inference Server. This API provides high-performance logo detection capabilities for 40 Pakistani media and government organization logos with enterprise-grade reliability and scalability.

## 🎯 API Overview

**Base URL**: `http://yolo11-logo.models-inference.local:9002`
**Model**: YOLOv11n (Nano) - Optimized for logo detection
**Classes**: 40 Pakistani media and government organization logos
**Input**: Images (JPEG, PNG, JPG, WebP)
**Output**: Logo bounding boxes, confidence scores, class labels

## 🏗️ System Architecture

```
Client Application
       ↓
Logo Detection API (Port 9002)
       ↓
NVIDIA Triton Inference Server
       ↓
YOLOv11n Logo Detection Model
```

## 🎯 Supported Logo Classes (Pakistani Media & Government Organizations)

The YOLO11 Logo Detection API can detect 40 different Pakistani media and government organization logos:

### Media Organizations (News Channels)

| Class ID | Logo          | Class ID | Logo                | Class ID | Logo                        | Class ID | Logo         |
| -------- | ------------- | -------- | ------------------- | -------- | --------------------------- | -------- | ------------ |
| 0        | 24news        | 10       | dawn                | 20       | ministry of foreign affairs | 30       | ptvglobal    |
| 1        | 92news        | 11       | dunyanews           | 21       | ministry of human rights    | 31       | ptvhome      |
| 2        | aajnews       | 12       | election commission | 22       | nab                         | 32       | ptvnational  |
| 3        | abbtakk       | 13       | express             | 23       | neonews                     | 33       | ptvparliment |
| 4        | abnnews       | 14       | fbr                 | 24       | nepra                       | 34       | ptvsports    |
| 5        | ajktelevision | 15       | geo                 | 25       | onenews                     | 35       | ptvworld     |
| 6        | ary           | 16       | gnnnews             | 26       | pemra                       | 36       | publicnews   |
| 7        | bol           | 17       | gtvnews             | 27       | pta                         | 37       | rozenews     |
| 8        | capitaltvnews | 18       | hum                 | 28       | ptcl                        | 38       | samaa        |
| 9        | cda           | 19       | mashriqnews         | 29       | ptv                         | 39       | suchnews     |

### Categories:

- **News Channels**: 24news, 92news, aajnews, abbtakk, abnnews, ajktelevision, ary, bol, capitaltvnews, dawn, dunyanews, express, geo, gnnnews, gtvnews, hum, mashriqnews, neonews, onenews, publicnews, rozenews, samaa, suchnews
- **Government Agencies**: cda, election commission of pakistan, fbr, ministry of foreign affairs, ministry of human rights, nab, nepra, pemra, pta
- **State Television**: ptv, ptvglobal, ptvhome, ptvnational, ptvparliment, ptvsports, ptvworld
- **Telecommunications**: ptcl

## 🔗 Quick Start Integration

### 1. Basic Detection Request

```bash
curl -X POST "http://yolo11-logo.models-inference.local:9002/v1/detect" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo11n-logo",
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
curl -X POST "http://yolo11-logo.models-inference.local:9002/v1/detect" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo11n-logo",
    "image": {
      "type": "url",
      "url": "https://example.com/logo-image.jpg"
    }
  }'
```

### 3. Get Available Logo Classes

```bash
curl -X GET "http://yolo11-logo.models-inference.local:9002/classes"
```

### 4. Health Check

```bash
curl -X GET "http://yolo11-logo.models-inference.local:9002/health"
```

## 📝 API Reference

### Authentication

**Required for detection endpoint only**

```http
Authorization: Bearer 123
```

### POST /v1/detect - Logo Detection

**Request Body Schema:**

```json
{
  "model": "yolo11n-logo",              // Required: Model name
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
    "object_images": true,             // Include cropped logo images
    "annotated_image": false           // Include image with boxes only
  }
}
```

**Response Schema:**

```json
{
  "id": "detect_yolo11_abc123",        // Unique request ID
  "created": 1694123456,                // Unix timestamp
  "model": "yolo11n-logo",             // Model used
  "task": "detection",                 // Task type
  "usage": {
    "inference_time_ms": 45.67,        // Model inference time
    "preprocessing_time_ms": 12.34,    // Image preprocessing time
    "postprocessing_time_ms": 8.90,    // Result postprocessing time
    "total_time_ms": 66.91             // Total processing time
  },
  "detections": [                      // Array of detected logos
    {
      "class_id": 15,                  // Logo class ID (0-39)
      "class_name": "geo",             // Human-readable logo name
      "confidence": 0.89,              // Detection confidence (0.0-1.0)
      "bounding_box": {
        "x1": 100,                     // Top-left X coordinate
        "y1": 150,                     // Top-left Y coordinate
        "x2": 300,                     // Bottom-right X coordinate
        "y2": 450                      // Bottom-right Y coordinate
      },
      "object_image": "<base64>"       // Cropped logo (if requested)
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
  "service": "YOLO11 Logo Detection API",
  "triton_server": "healthy",
  "model_status": "healthy"
}
```

### GET /classes - Available Classes

**Response:**

```json
{
  "classes": {
    "0": "24news",
    "1": "92news",
    "2": "aajnews",
    // ... all 40 Pakistani logo classes
  }
}
```

### GET /config - API Configuration

**Response:**

```json
{
  "triton_server_url": "yolo11-logo.models-inference.local",
  "model_name": "yolo11n-logo",
  "default_confidence_threshold": 0.25,
  "default_nms_threshold": 0.45,
  "max_image_size": 10485760,
  "supported_formats": ["jpeg", "png", "jpg", "webp"],
  "task_type": "detect"
}
```

### cURL Examples

```bash
# Basic logo detection with base64 image
curl -X POST "http://yolo11-logo.models-inference.local:9002/v1/detect" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "yolo11n-logo",
  "image": {
    "type": "base64",
    "data": "$(base64 -w 0 news_screenshot.jpg)"
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

# Logo detection from URL
curl -X POST "http://yolo11-logo.models-inference.local:9002/v1/detect" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo11n-logo",
    "image": {
      "type": "url",
      "url": "https://example.com/news_channel_screenshot.jpg"
    }
  }'

# Get available logo classes
curl -X GET "http://yolo11-logo.models-inference.local:9002/classes"

# Health check
curl -X GET "http://yolo11-logo.models-inference.local:9002/health"

# Get configuration
curl -X GET "http://yolo11-logo.models-inference.local:9002/config"
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

- **object_images**: Get cropped images of detected logos (base64)
- **labeled_image**: Get full image with logo labels and confidence scores
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
| `POST /v1/detect` | POST   | Logo detection           | Required       |
| `GET /health`     | GET    | Service health status    | None           |
| `GET /classes`    | GET    | Available object classes | None           |
| `GET /config`     | GET    | API configuration        | None           |
| `GET /docs`       | GET    | Interactive API docs     | None           |
| `GET /`           | GET    | API information          | None           |

## 🏥 System Status

**Current Deployment:**

- **Environment**: Production
- **URL**: logo-dgx.nimar.gov.pk
- **Model Version**: YOLOv11n
- **API Key**: `123` (Demo key - request production key)
- **Uptime**: 99.9% SLA

**Infrastructure:**

- **Kubernetes Cluster**: Production cluster
- **Namespace**: `yolo-logo-api`
- **Replicas**: 2-10 (auto-scaling)
- **Resources**: 512Mi-2Gi RAM, 250m-1000m CPU per pod
- **Ingress**: NGINX with SSL termination

---

*This API is designed for production use with enterprise-grade reliability. For support or production API keys, contact the MLOps team.*