# YOLO5 Ticker Flasher Detection API

A production-ready REST API for real-time ticker and flasher detection using YOLOv5 models powered by NVIDIA Triton Inference Server. This API provides high-performance detection capabilities for channels tickers & flasher components with enterprise-grade reliability and scalability.

## API Overview

**Base URL**: `https://ticker-dgx.nimar.gov.pk` (production) / `http://localhost:8000` (local)
**Model**: YOLOv5 Ticker Flasher - Specialized for channels tickers & flasher detection
**Classes**: 9 ticker/flasher component classes
**Input**: Images (JPEG, PNG, JPG, WebP)
**Output**: Bounding boxes, confidence scores, class labels for channels tickers & flasher components

## System Architecture

```
Client Application
       ↓
NGINX Ingress (ticker-dgx.nimar.gov.pk)
       ↓
Kubernetes Service (Load Balancer)
       ↓
YOLO5 Ticker Flasher API Pods (2-10 replicas)
       ↓
NVIDIA Triton Inference Server
       ↓
YOLOv5 Ticker Flasher Model
```

## Supported Ticker/Flasher Classes

The YOLO5 Ticker Flasher API can detect 9 different channels tickers & flasher component classes:

| Class ID | Object         | Description                               |
| -------- | -------------- | ----------------------------------------- |
| 0        | BlurredFlasher | Blurred or out-of-focus flasher component |
| 1        | BlurredTicker  | Blurred or out-of-focus ticker component  |
| 2        | BlurredTop     | Blurred or out-of-focus top section       |
| 3        | Bottom         | Bottom section of channels tickers        |
| 4        | Flasher        | Clear flasher component                   |
| 5        | HalfFlasher    | Partially visible flasher component       |
| 6        | HalfTicker     | Partially visible ticker component        |
| 7        | HalfTop        | Partially visible top section             |
| 8        | Top            | Top section of channels                   |

## 🔗 Quick Start Integration

### 1. Basic Detection Request

```bash
curl -X POST "https://ticker-dgx.nimar.gov.pk/v1/detect" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo5-ticker-flasher",
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
curl -X POST "https://ticker-dgx.nimar.gov.pk/v1/detect" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo5-ticker-flasher",
    "image": {
      "type": "url",
      "url": "https://example.com/traffic-light.jpg"
    }
  }'
```

### 3. Get Available Classes

```bash
curl -X GET "https://ticker-dgx.nimar.gov.pk/classes"
```

### 4. Health Check

```bash
curl -X GET "https://ticker-dgx.nimar.gov.pk/health"
```

## API Reference

### Authentication

**Required for detection endpoint only**

```http
Authorization: Bearer 123
```

### POST /v1/detect - Object Detection

**Request Body Schema:**

```json
{
  "model": "yolo5-ticker-flasher",     // Required: Model name
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
  "id": "detect_yolo5_ticker_flasher_abc123", // Unique request ID
  "created": 1694123456,                      // Unix timestamp
  "model": "yolo5-ticker-flasher",           // Model used
  "task": "detection",                       // Task type
  "usage": {
    "inference_time_ms": 45.67,              // Model inference time
    "preprocessing_time_ms": 12.34,          // Image preprocessing time
    "postprocessing_time_ms": 8.90,          // Result postprocessing time
    "total_time_ms": 66.91                   // Total processing time
  },
  "detections": [                            // Array of detected objects
    {
      "class_id": 4,                         // Class ID (0-8)
      "class_name": "Flasher",               // Human-readable class name
      "confidence": 0.89,                    // Detection confidence (0.0-1.0)
      "bounding_box": {
        "x1": 100,                           // Top-left X coordinate
        "y1": 150,                           // Top-left Y coordinate
        "x2": 300,                           // Bottom-right X coordinate
        "y2": 450                            // Bottom-right Y coordinate
      },
      "object_image": "<base64>"             // Cropped object (if requested)
    }
  ],
  "image_info": {
    "width": 640,                            // Original image width
    "height": 480,                           // Original image height
    "format": "JPEG"                         // Image format
  },
  "labeled_image": "<base64>",               // Full image with labels (if requested)
  "annotated_image": "<base64>"              // Full image with boxes (if requested)
}
```

### GET /health - Service Health

**Response:**

```json
{
  "status": "healthy",
  "service": "YOLO5 Ticker Flasher Detection API",
  "triton_server": "healthy",
  "model_status": "healthy"
}
```

### GET /classes - Available Classes

**Response:**

```json
{
  "classes": {
    "0": "BlurredFlasher",
    "1": "BlurredTicker",
    "2": "BlurredTop",
    "3": "Bottom",
    "4": "Flasher",
    "5": "HalfFlasher",
    "6": "HalfTicker",
    "7": "HalfTop",
    "8": "Top"
  }
}
```

### GET /config - API Configuration

**Response:**

```json
{
  "triton_server_url": "yolo5-inference-engine",
  "model_name": "yolo5-ticker-flasher",
  "default_confidence_threshold": 0.25,
  "default_nms_threshold": 0.45,
  "max_image_size": 10485760,
  "supported_formats": ["jpeg", "png", "jpg", "webp"],
  "task_type": "detect"
}
```

## Code Examples

### cURL Examples

```bash
# Basic detection with base64 image
curl -X POST "https://ticker-dgx.nimar.gov.pk/v1/detect" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "yolo5-ticker-flasher",
  "image": {
    "type": "base64",
    "data": "$(base64 -w 0 tv_frame.jpg)"
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
curl -X POST "https://ticker-dgx.nimar.gov.pk/v1/detect" \
  -H "Authorization: Bearer 123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolo5-ticker-flasher",
    "image": {
      "type": "url",
      "url": "https://example.com/tv_frame.jpg"
    }
  }'

# Get available classes
curl -X GET "https://ticker-dgx.nimar.gov.pk/classes"

# Health check
curl -X GET "https://ticker-dgx.nimar.gov.pk/health"

# Get configuration
curl -X GET "https://ticker-dgx.nimar.gov.pk/config"
```

## Configuration Parameters

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

## Performance & Scaling

### API Performance

- **Average Response Time**: 50-150ms per image
- **Throughput**: Depends on Triton server capacity
- **Model**: YOLOv5 optimized for ticker/flasher detection
- **Hardware**: Runs on NVIDIA GPU via Triton Inference Server

## Error Handling

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

| Endpoint          | Method | Description                      | Authentication |
| ----------------- | ------ | -------------------------------- | -------------- |
| `POST /v1/detect` | POST   | Ticker/flasher detection         | Required       |
| `GET /health`     | GET    | Service health status            | None           |
| `GET /classes`    | GET    | Available ticker/flasher classes | None           |
| `GET /config`     | GET    | API configuration                | None           |
| `GET /docs`       | GET    | Interactive API docs             | None           |
| `GET /`           | GET    | API information                  | None           |

## System Status

**Current Deployment:**

- **Environment**: Production
- **URL**: `https://ticker-dgx.nimar.gov.pk`
- **Model Version**: YOLOv5 Ticker Flasher
- **API Key**: `123` (Demo key - request production key)

**Infrastructure:**

- **Kubernetes Cluster**: Production cluster with auto-scaling (2-10 replicas)
- **Triton Server**: triton-yolo-ticker-flasher-service.models-inference.svc.cluster.local:8000
- **Model Name**: yolo5-ticker-flasher
- **Classes**: 9 ticker/flasher component classes
- **Hardware**: NVIDIA GPU (via Triton Inference Server)
- **Ingress**: NGINX with SSL termination

**Configuration:**

For local development, copy `env_example` to `.env` and modify settings:

- TRITON_SERVER_URL: Triton inference server endpoint
- API_KEY: Production API key
- LOG_LEVEL: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

---

*This API is specialized for channels ticker and flasher component detection. For support or configuration help, contact the MLOps team.*
