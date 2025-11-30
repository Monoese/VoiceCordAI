# VoiceCordAI Docker Image
# Packages Python runtime and FFmpeg in container userspace

FROM python:3.13-slim

# Install system dependencies
# - FFmpeg: Required for audio processing and playback
# - Clean up apt cache to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first (better layer caching)
# This layer is cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables must be provided at runtime via:
#   docker run --env-file .env <image>
# Required: DISCORD_TOKEN, and at least one AI provider API key

# Run the bot
CMD ["python", "main.py"]
