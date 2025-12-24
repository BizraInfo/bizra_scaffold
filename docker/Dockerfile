# ============================================================================
# BIZRA AEON OMEGA - PRODUCTION DOCKERFILE
# ============================================================================
# Multi-stage build for optimal image size and security
# Target size: <500MB (optimized from 1.2GB)
# Python 3.10+ | Alpine-based for minimal footprint

# ──────────────────────────────────────────────────────────────────────────
# STAGE 1: BUILD - Install dependencies and compile
# ──────────────────────────────────────────────────────────────────────────
FROM python:3.10-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    libffi-dev \
    openssl-dev \
    cmake \
    make \
    git

# Install liboqs (post-quantum cryptography)
RUN git clone --depth 1 --branch 0.8.0 https://github.com/open-quantum-safe/liboqs.git \
    && cd liboqs \
    && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. \
    && make -j$(nproc) \
    && make install \
    && cd ../.. \
    && rm -rf liboqs

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements-production.txt /tmp/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/requirements-production.txt

# ──────────────────────────────────────────────────────────────────────────
# STAGE 2: PRODUCTION - Minimal runtime image
# ──────────────────────────────────────────────────────────────────────────
FROM python:3.10-alpine

# Install runtime dependencies only
RUN apk add --no-cache \
    libstdc++ \
    libgomp \
    libffi \
    openssl

# Copy liboqs from builder
COPY --from=builder /usr/local/lib/liboqs* /usr/local/lib/
COPY --from=builder /usr/local/include/oqs /usr/local/include/oqs

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN addgroup -g 1000 bizra \
    && adduser -D -u 1000 -G bizra bizra

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=bizra:bizra . /app/

# Create secure key storage directory
RUN mkdir -p /secure/keys \
    && chown -R bizra:bizra /secure \
    && chmod 700 /secure/keys

# Switch to non-root user
USER bizra

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Run application
CMD ["uvicorn", "core.engine.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================================
# BUILD INSTRUCTIONS
# ============================================================================
#
# Build:
#   docker build -t bizra:10.0.0 -f docker/Dockerfile .
#
# Run:
#   docker run -p 8000:8000 --env-file .env bizra:10.0.0
#
# Size verification:
#   docker images bizra:10.0.0
#   Expected: <500MB
#
# Security scan:
#   docker scan bizra:10.0.0
#
# ============================================================================
