# Voice Assistant for Raspberry Pi 3

A voice-activated AI assistant for kids, running on a Raspberry Pi with a USB speaker.

## Stack

- **Groq Whisper** — speech-to-text (free tier)
- **OpenAI GPT-4.1** — AI responses
- **Google Cloud Wavenet** — text-to-speech (4M chars/month free)
- **espeak** — TTS fallback (offline)

## Hardware

- Raspberry Pi 3 (or newer)
- Jabra Speak 510/750 USB speakerphone (auto-detected)
- WiFi connection

## Features

- Configurable wake word and assistant name via `ASSISTANT_NAME` env var
- Fuzzy wake word matching (handles mispronunciations, accents)
- Conversation mode — follow-up questions without repeating the wake word
- Cancel/stop phrases ("never mind", "cancel", "stop")
- Adaptive silence threshold (calibrates to ambient noise)
- Network retry with exponential backoff on all API calls
- Self-hearing prevention (mic flush after TTS playback)
- Audio pre-buffer to prevent clipping the start of speech
- Conversation history persisted to JSON across restarts
- Kid-friendly responses (simple language, no markdown)
- Auto-detects USB audio device (no hardcoded card numbers)
- Falls back to espeak if Google TTS credentials are missing

## Deployment (Docker)

### Prerequisites

- Docker with buildx on your build machine
- Docker on the Raspberry Pi
- A local Docker registry (or modify `deploy.sh` for Docker Hub)

### 1. Configure API keys on the Pi

```bash
ssh pi@raspberrypi.local
mkdir -p ~/docker/voice-assistant
cd ~/docker/voice-assistant
```

Create a `.env` file:
```bash
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
ASSISTANT_NAME=Opal
```

Copy your Google Cloud service account key:
```bash
# From your build machine:
scp your-gcp-key.json pi@raspberrypi.local:~/docker/voice-assistant/gcp-key.json
```

### 2. Build and push

On your build machine:
```bash
./deploy.sh
```

This builds an ARM32v7 image and pushes to the registry.

### 3. Deploy on the Pi

```bash
scp docker-compose.yml pi@raspberrypi.local:~/docker/voice-assistant/
ssh pi@raspberrypi.local
cd ~/docker/voice-assistant
docker compose pull
docker compose up -d
```

### 4. Check logs

```bash
docker logs -f beans
```

## Usage

1. Wait for "{name} is ready!" announcement
2. Say the wake word (e.g., "Hey Opal") or just the name ("Opal")
3. Hear the activation chime
4. Speak your question
5. Get a response — then ask follow-ups without repeating the wake word
6. After ~8 seconds of silence, returns to wake word listening

### Cancel/Exit

- Say "never mind", "cancel", or "stop" to cancel a command
- Say "bye", "that's all", or "I'm done" to exit conversation mode

## Configuration

All configuration is via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (required) | Groq API key for Whisper STT |
| `OPENAI_API_KEY` | (required) | OpenAI API key for GPT |
| `ASSISTANT_NAME` | `Beans` | Assistant name and wake word |
| `AUDIO_DEVICE` | (auto-detect) | ALSA device override, e.g. `plughw:2,0` |
| `HISTORY_FILE` | `/app/data/history.json` | Path to conversation history |

## Troubleshooting

### No audio / device not found
```bash
# Check USB speaker is connected
docker exec beans arecord -l
docker exec beans aplay -l
```

### Wake word not detected
- Speak clearly within 1-2 feet of the speaker
- Check logs for "Heard:" lines to see what Whisper transcribes
- The assistant handles variations like "hey {name}", "hay {name}", just "{name}"

### No TTS audio output
- Verify Google Cloud credentials: `gcp-key.json` must be mounted
- Falls back to espeak automatically if Google TTS fails
- Check `AUDIO_DEVICE` env var if using a non-Jabra USB speaker

### Registry connection error
Add insecure registry on the Pi:
```bash
echo '{"insecure-registries":["YOUR_REGISTRY:PORT"]}' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker
```

## License

MIT
