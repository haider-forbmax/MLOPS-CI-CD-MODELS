# 🎙️ Emotion Inference API

The **Emotion Inference API** is a FastAPI-based microservice designed to detect emotional states from **audio** or **video** inputs. It automatically extracts audio from uploaded video files, sends the processed audio to a **Triton Inference Server**, and returns the detected **emotion label**.

This API integrates with a **Hugging Face speech emotion recognition model** served through Triton, supporting multiple audio and video formats.

---

## 🚀 Features

- 🎧 **Supports both audio & video** (automatically extracts audio from video)
- 🧠 **Triton Inference Server integration**
- 🔐 **API Key-based authentication**
- ⚙️ **Configurable model and Triton endpoint**
- 📈 **Health monitoring endpoints**
- 🧩 **Modular structure for easy deployment**

---

## 🏗️ Project Structure

```
emotion-api
├── emotion_inference/
├   ├── __init__.py
├   ├── app.py              # FastAPI main application
├   ├── config.py           # Configuration constants and model settings
├   ├── utils.py            # Triton client and inference utilities
├── requirements.txt    # Python dependencies
└── README.md           # Documentation (this file)
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd emotion-api
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧩 Configuration

You can configure model and server details in [`config.py`](./config.py):

```python
class Config:
    TRITON_URL = "192.168.18.22:32216"  # Triton server URL
    MODEL_NAME = "speech_emotion_recognition"
    MODEL_ID = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
    API_KEY = "123"
```

> ⚠️ Make sure the Triton server is running and accessible at the specified URL.

---

## ▶️ Running the API

Start the API with **Uvicorn**:

```bash
uvicorn emotion_inference.app:app --host 0.0.0.0 --port 2000
```

By default, the API runs on **`http://localhost:2000`**.

---

## 🧠 API Endpoints

### 🔹 `POST /v1/voice_compassion`

Detects the emotion from an uploaded audio or video file.

**Request:**

- **Headers:**
  ```
  Authorization: Bearer 123
  ```
- **Body:**
  - `video_file`: Audio or video file (`.wav`, `.mp3`, `.mp4`, etc.)

**Example (using `curl`):**

```bash
curl -X POST "http://localhost:2000/v1/voice_compassion"   -H "Authorization: Bearer 123"   -F "video_file=@sample_audio.wav"
```

**Response:**

```json
{
  "emotion": "joy"
}
```

---

### 🔹 `GET /health`

Checks the health of the Triton server and model.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-11-05T12:34:56.789Z",
  "version": "1.0.0"
}
```

---

### 🔹 `GET /config`

Returns current configuration details.

**Response:**

```json
{
  "triton_server_url": "192.168.18.22:32216",
  "model_name": "speech_emotion_recognition",
  "supported_extensions": [".wav", ".mp3", ".mp4", ".avi", ...]
}
```

---

### 🔹 `GET /`

Root endpoint providing basic API information.

**Response:**

```json
{
  "service": "Voice Compassion API",
  "version": "1.0.0",
  "description": "Voice Compassion API using Triton Inference Server",
  "endpoints": { ... }
}
```

---

## 🧰 Supported Formats

- **Audio:** `.wav`, `.mp3`, `.flac`, `.aac`, `.ogg`, `.opus`, `.m4a`, `.aiff`, `.amr`, `.wma`
- **Video:** `.mp4`, `.mkv`, `.webm`, `.avi`, `.mov`, `.mpg`, `.mpeg`, `.flv`, `.3gp`, `.ts`, `.f4v`

---

## 🧪 Example Workflow

1. Upload a `.mp4` video → API extracts the audio automatically.
2. Audio is processed using the **feature extractor** from the Hugging Face model.
3. Extracted features are sent to **Triton Inference Server**.
4. Triton returns emotion logits → API converts them into a label.
5. You receive the **predicted emotion** in the response.

---

## 🧾 Example Response

```json
{
  "emotion": "sadness"
}
```

---

## 🛠️ Dependencies

Main libraries used:

- `fastapi`
- `uvicorn`
- `moviepy`
- `librosa`
- `transformers`
- `torch`
- `tritonclient`
- `numpy`

(Full list available in `requirements.txt`)

---

## 🧑‍💻 Developer Notes

- Ensure your Triton server has the correct model (`speech_emotion_recognition`) loaded.
- If you face connection issues, check your `TRITON_URL` and network configuration.
- Temporary files are automatically cleaned after each request.

---

## 📄 License

This project is proprietary to your organization.  
For any reproduction, distribution, or modification, please seek permission from the author.
