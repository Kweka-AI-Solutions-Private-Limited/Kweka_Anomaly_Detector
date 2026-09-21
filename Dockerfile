# ==============================================================
# Anomaly Detector — Optimized Multi-Stage Dockerfile (~8.8 GB)
# ==============================================================
# Stage 1: Build React frontend (Node 20 slim)
# Stage 2: Runtime (Python 3.11 slim + PyTorch cu128 GPU)
# ==============================================================

# ---------------------------------------------------------------
# STAGE 1: Build React Frontend
# ---------------------------------------------------------------
FROM node:20-slim AS frontend-builder

WORKDIR /build/frontend

# Copy package files first for layer caching
COPY frontend_react/package.json frontend_react/package-lock.json ./

# Install dependencies
RUN npm ci --no-audit --no-fund

# Copy frontend source
COPY frontend_react/ ./

# Build production bundle (VITE_API_BASE_URL empty for same-origin)
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build

# ---------------------------------------------------------------
# STAGE 2: Runtime Image (Python 3.11 Slim)
# Eliminates 4.23 GB redundant OS CUDA layers
# ---------------------------------------------------------------
FROM python:3.11-slim

# Prevents interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

# Install runtime C dependencies for OpenCV, PyTorch, and health check curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------
# Install Python production dependencies (CPU PyTorch ~1.8GB)
# ---------------------------------------------------------------
COPY backend/requirements.txt ./requirements.txt

# Strip +cu128 requirement tags for CPU wheel installation
RUN sed -i 's/+cu128//g' requirements.txt && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt && \
    find /usr/local/lib/python3.11/site-packages/ -name "*.pyc" -delete && \
    find /usr/local/lib/python3.11/site-packages/ -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------
# Copy backend application source
# ---------------------------------------------------------------
COPY backend/main.py ./backend/main.py
COPY backend/src/ ./backend/src/

# ---------------------------------------------------------------
# Copy built React frontend into the image
# ---------------------------------------------------------------
COPY --from=frontend-builder /build/frontend/dist ./frontend_dist

# ---------------------------------------------------------------
# Create storage directory structure
# (actual data is bind-mounted at runtime — fallback creation only)
# ---------------------------------------------------------------
RUN mkdir -p \
    backend/storage/inspections \
    backend/storage/artifacts \
    backend/storage/references \
    backend/storage/models \
    backend/storage/temp_uploads

# ---------------------------------------------------------------
# Environment defaults
# ---------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/backend/src
ENV PORT=8000
ENV HOST=0.0.0.0
ENV ENV=production

# ---------------------------------------------------------------
# Expose application port
# ---------------------------------------------------------------
EXPOSE 8000

# ---------------------------------------------------------------
# Health check (uses CMD form within HEALTHCHECK block)
# ---------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# ---------------------------------------------------------------
# Single application startup command
# ---------------------------------------------------------------
CMD ["python", "-m", "uvicorn", "api_server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
