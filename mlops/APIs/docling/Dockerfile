# ==============================================================================
# PRODUCTION DOCKERFILE FOR DOCLING ON DGX H200
# ==============================================================================
# Optimized for NVIDIA H200 GPUs with CUDA 12.x and PyTorch 2.x
# Build: docker build -t your-registry/docling-h200:latest .
# Push: docker push your-registry.com/docling-h200:latest
# ==============================================================================

# Stage 1: Base with CUDA support
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Additional dependencies
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Stage 2: Builder with all dependencies
FROM base AS builder

# Set working directory
WORKDIR /build

# Install PyTorch with CUDA 12.4 support (optimized for H200)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install Docling with all dependencies
RUN pip install --no-cache-dir \
    "docling[api]==2.57.0" \
    fastapi==0.115.5 \
    uvicorn[standard]==0.32.1 \
    python-multipart==0.0.20 \
    aiofiles==24.1.0 \
    python-magic==0.4.27

# Install additional optimization libraries
RUN pip install --no-cache-dir \
    nvidia-ml-py==12.560.30 \
    pynvml==11.5.3 \
    prometheus-client==0.21.0

# Create cache directories to ensure they exist
RUN mkdir -p /root/.cache/huggingface \
             /root/.cache/torch \
             /root/.cache/pip

# Download Docling models during build
RUN python -c "from docling.document_converter import DocumentConverter; \
    from docling.datamodel.pipeline_options import PdfPipelineOptions; \
    print('Downloading Docling models...'); \
    converter = DocumentConverter(); \
    print('✓ Models downloaded successfully')"

# Stage 3: Production runtime
FROM base AS production

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create cache directory structure first
RUN mkdir -p /root/.cache

# Copy model cache from builder (only if it exists and has content)
COPY --from=builder /root/.cache /root/.cache

# Create application directory
WORKDIR /app

# Create directories
RUN mkdir -p /models /scratch /app && \
    chmod 777 /scratch

# Set environment for H200 optimization
ENV CUDA_VISIBLE_DEVICES="" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    # H200 Memory optimization
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True" \
    TORCH_CUDNN_V8_API_ENABLED=1 \
    # Threading
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    # Model cache paths
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/transformers \
    TORCH_HOME=/models/torch \
    # Docling settings
    DOCLING_DEVICE=cuda \
    DOCLING_NUM_THREADS=8

# Health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8000/health || exit 1' > /healthcheck.sh && \
    chmod +x /healthcheck.sh

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD /healthcheck.sh

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash docling && \
    chown -R docling:docling /app /models /scratch /root/.cache

# Switch to non-root user
USER docling

# Expose port
EXPOSE 8000

# Default command (will be overridden by ConfigMap in K8s)
CMD ["python", "-u", "/app/docling_server.py"]

# ==============================================================================
# BUILD INSTRUCTIONS:
# 
# 1. Build the image:
#    docker build -t your-registry.com/docling-h200:v1.0.0 -f Dockerfile .
#
# 2. Test locally with GPU:
#    docker run --gpus all -p 8000:8000 \
#      -v $(pwd)/models:/models \
#      -v $(pwd)/scratch:/scratch \
#      your-registry.com/docling-h200:v1.0.0
#
# 3. Push to registry:
#    docker push your-registry.com/docling-h200:v1.0.0
#
# 4. Update deployment.yaml with new image:
#    image: your-registry.com/docling-h200:v1.0.0
#
# ==============================================================================