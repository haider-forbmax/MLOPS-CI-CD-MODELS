---

# ResNet100 Face Embedding API

A **FastAPI service** that generates face embeddings using a **ResNet100 model** deployed on **NVIDIA Triton Inference Server**.

---

## 🚀 Features

* Accepts **image URL** or **base64-encoded image**.
* Runs inference on Triton (`resnet100` model).
* Returns face embeddings (512D vector by default).
* Health check endpoint for server and model status.

## ⚡ Endpoints

### 1. Health Check

**`GET /health`**

Check server and model readiness.

#### ✅ Example Request:

```bash
curl http://localhost:8080/health
```

#### 📤 Example Response:

```json
{
  "triton_server": { "healthy": true },
  "model": { "name": "resnet100", "ready": true },
  "timestamp": 1693312345
}
```

---

### 2. Generate Embeddings

**`POST /embeddings`**

Generate embeddings from an image. Supports **url** and **base64** input.

#### 📥 Request Body (URL example):

```json
{
  "model": "resnet100",
  "image": {
    "type": "url",
    "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
  },
  "include_embeddings": true
}
```

#### 📥 Request Body (Base64 example):

```json
{
  "model": "resnet100",
  "image": {
    "type": "base64",
    "data": "iVBORw0KGgoAAAANSUhEUgAA..."
  },
  "include_embeddings": false
}
```

---

#### 📤 Example Response (with embeddings):

```json
{
  "id": "embed_resnet100_a3c9f4d1",
  "task": "embedding",
  "created": 1693312345,
  "model": "resnet100",
  "usage": {
    "inference_time_ms": 12.34,
    "total_time_ms": 50.67
  },
  "embeddings": [0.012, -0.034, 0.998, ...], 
  "image_info": {
    "width": 512,
    "height": 512,
    "format": "rgb"
  }
}
```

---

## ⚙️ Configuration

```python
TRITON_HOST = "192.168.18.49:8000"
MODEL_NAME = "resnet100"
```

---

## 🛠 Notes

* Embedding size depends on your model (commonly **512D** for ResNet100).
* The API always converts input images to **RGB** before inference.
* Response time includes both preprocessing and Triton inference.

---
