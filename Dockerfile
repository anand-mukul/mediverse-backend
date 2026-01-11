FROM python:3.11-slim

# System dependencies - essential for multimedia and networking
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m appuser
WORKDIR /app

# Install Python dependencies (cached separately for faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set proper permissions
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -f http://localhost:8000/api/ || exit 1

# Run application with production-grade settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
