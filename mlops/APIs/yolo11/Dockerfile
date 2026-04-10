# =============================================================================
# Multi-stage Docker build for YOLO11 API
# =============================================================================

# ------------------- Stage 1: Build Stage ------------------------------
FROM python:3.11 AS backend-builder

# Set the working directory to /app
WORKDIR /app

# Copy the contents of the backend directory into the container at /app
COPY requirements.txt .

# Install dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ------------------- Stage 2: Final Stage ------------------------------

# Use a slim Python 3.9 image as the final base image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the built dependencies from the backend-builder stage
COPY --from=backend-builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=backend-builder /usr/local/bin/ /usr/local/bin/

# Copy the application code from the backend-builder stage
COPY . /app

# Expose port 5000 for the Flask application
EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]