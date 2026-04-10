FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install only the CPU-safe application dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create package dir and copy code
RUN mkdir -p emotion_inference
COPY __init__.py emotion_inference/
COPY config.py emotion_inference/
COPY utils.py emotion_inference/
COPY app.py emotion_inference/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://localhost:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "emotion_inference.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
