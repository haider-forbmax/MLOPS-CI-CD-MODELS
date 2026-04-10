# Much better choice in 2026 — newer OS, Python 3.10 by default, still supports CUDA 11.8
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 + pip + dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip & setuptools (important for modern builds)
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Set work directory
WORKDIR /app

# Copy & install requirements (should now succeed without puccinialin issues)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download medium model (cpu is fine for caching; GPU used at runtime)
RUN python3 -c "from faster_whisper import WhisperModel; \
    model = WhisperModel('medium', device='cpu', compute_type='float32'); \
    print('Medium Whisper model downloaded and cached.')"

# Copy app code
COPY app.py .

# Expose FastAPI port
EXPOSE 8000

# Run with uvicorn (workers=1 is usually enough for GPU-bound inference)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]