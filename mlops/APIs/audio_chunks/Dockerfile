# Multi-stage build for minimal image size
FROM python:3.11-alpine as builder

WORKDIR /tmp

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    linux-headers \
    ffmpeg

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Final stage - ultra-lightweight
FROM python:3.11-alpine

WORKDIR /app

# Install runtime dependencies only (ffmpeg for audio processing)
RUN apk add --no-cache ffmpeg

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Copy application code
COPY app.py config.py utils.py ./

# Create non-root user for security
RUN adduser -D -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Lightweight health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=2 \
    CMD wget --quiet --tries=1 --spider http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run uvicorn with multiple workers for robustness
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
