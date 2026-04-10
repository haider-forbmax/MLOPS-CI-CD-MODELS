# Emotion Inference API

This repository contains a FastAPI service for speech emotion detection. It accepts an uploaded audio or video file, extracts audio when needed, prepares features with a Hugging Face feature extractor, sends inference requests to a Triton Inference Server, and returns the predicted emotion label.

The project is designed for CPU-based API serving. The local service does not run model inference directly on GPU; it prepares inputs and communicates with Triton over HTTP.

## What This Repo Does

- Exposes an HTTP API for voice emotion recognition
- Accepts both audio and video uploads
- Extracts audio from video files using `ffmpeg`
- Uses `transformers` to load the feature extractor and label mapping
- Sends inference requests to Triton using `tritonclient`
- Returns health, config, and root metadata endpoints

## Main Endpoints

- `POST /v1/voice_compassion`
  Upload an audio or video file and receive the detected emotion.
- `GET /health`
  Checks Triton server and model readiness.
- `GET /config`
  Returns current runtime configuration.
- `GET /`
  Returns API metadata and supported formats.

## Project Files

- [app.py](/home/nouman/data/FORBMAX/NIMAR/mlops/APIs/emotion-api/emotion_inference/app.py)
  FastAPI application and route handlers
- [utils.py](/home/nouman/data/FORBMAX/NIMAR/mlops/APIs/emotion-api/emotion_inference/utils.py)
  Audio preprocessing and Triton inference helpers
- [config.py](/home/nouman/data/FORBMAX/NIMAR/mlops/APIs/emotion-api/emotion_inference/config.py)
  Environment-based configuration and response schema
- [Dockerfile](/home/nouman/data/FORBMAX/NIMAR/mlops/APIs/emotion-api/emotion_inference/Dockerfile)
  Container image for CPU deployment
- [requirements.txt](/home/nouman/data/FORBMAX/NIMAR/mlops/APIs/emotion-api/emotion_inference/requirements.txt)
  Minimal Python dependencies for this service

## Requirements

- Python 3.11
- `ffmpeg`
- `libsndfile1`
- Access to a running Triton Inference Server

## Environment Variables

The service reads configuration from environment variables:

- `API_KEY`
- `TRITON_URL`
- `MODEL_NAME`
- `MODEL_ID`
- `APP_VERSION`
- `AUDIO_FORMATS`
- `VIDEO_FORMATS`
- `SUPPORTED_FORMATS`

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn emotion_inference.app:app --host 0.0.0.0 --port 8000 --workers 4
```

## Docker Run

Build:

```bash
docker build -t emotion-inference-api .
```

Run:

```bash
docker run -p 8000:8000 \
  -e API_KEY=123 \
  -e TRITON_URL=<your-triton-host>:8000 \
  -e MODEL_NAME=speech_emotion_recognition \
  emotion-inference-api
```

## Request Example

```bash
curl -X POST "http://localhost:8000/v1/voice_compassion" \
  -H "Authorization: Bearer 123" \
  -F "video_file=@sample.wav"
```

## Progress

| Date | Update | Status |
| --- | --- | --- |
| 2026-04-02 | Reviewed the codebase, removed unwanted and unnecessary Python libraries from `requirements.txt`, removed unused `moviepy` and `torch` usage from the code, and aligned the Docker image for CPU-only deployment. | Done |

## Notes

- The API expects Triton to host the configured speech emotion model.
- The feature extractor is loaded from Hugging Face using `MODEL_ID`.
- CPU-only here means this API container no longer installs CUDA-specific Python packages.
