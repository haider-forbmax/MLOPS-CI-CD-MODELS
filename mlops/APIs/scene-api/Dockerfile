FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scene_detection_inference/ /app/scene_detection_inference/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "scene_detection_inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
