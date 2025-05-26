# Multi-stage Docker build for OpenBehavior platform

# Stage 1: Rust builder
FROM rust:1.75 as rust-builder

WORKDIR /app/rust
COPY rust/Cargo.toml rust/Cargo.lock ./
COPY rust/src ./src

# Build Rust components
RUN cargo build --release

# Stage 2: Python builder
FROM python:3.11-slim as python-builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Stage 3: Node.js builder for dashboard
FROM node:18-alpine as node-builder

WORKDIR /app/dashboard
COPY dashboard/package*.json ./
RUN npm ci --only=production

COPY dashboard/ ./
RUN npm run build

# Stage 4: Final runtime image
FROM python:3.11-slim

# Install system runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy Python dependencies
COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy Rust binaries
COPY --from=rust-builder /app/rust/target/release/openbehavior-rust /usr/local/bin/

# Copy Python application
COPY python/ ./python/
COPY --from=node-builder /app/dashboard/build ./dashboard/

# Copy configuration files
COPY config/ ./config/
COPY templates/ ./templates/

# Create necessary directories
RUN mkdir -p /app/data /app/output /app/cache /app/logs \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app/python
ENV RUST_LOG=info
ENV OPENBEHAVIOR_CONFIG_PATH=/app/config/production.yaml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose ports
EXPOSE 8000 3000

# Start script
COPY scripts/start.sh ./
RUN chmod +x start.sh

CMD ["./start.sh"]