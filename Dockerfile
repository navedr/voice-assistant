# Dockerfile for Raspberry Pi Voice Assistant
# ARM32v7 base for Pi 3 Model A
FROM arm32v7/python:3.11-slim

WORKDIR /app

# Install build deps, compile python packages, then remove build deps in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    python3-pyaudio \
    alsa-utils \
    espeak \
    gcc \
    libc-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove gcc libc-dev libffi-dev portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy voice assistant code and assets
COPY voice_assistant_simple.py .
COPY activate.wav .

# Environment variables (override at runtime)
ENV GROQ_API_KEY="" \
    OPENAI_API_KEY="" \
    ASSISTANT_NAME="Beans" \
    AUDIO_DEVICE=""

# Run the voice assistant
CMD ["python", "-u", "voice_assistant_simple.py"]
