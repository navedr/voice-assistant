# Raspberry Pi Voice Assistant Project

## Overview
Voice assistant running on Raspberry Pi 3 Model A + Jabra Speak 510 USB speaker.
Uses Groq API (Whisper) for speech-to-text, OpenAI GPT-4o-mini for AI, and espeak for text-to-speech.

**Status:** Ready for deployment
**Created:** 2026-04-08

## Hardware
- **Raspberry Pi 3 Model A**
  - 1.4GHz quad-core ARM Cortex-A53
  - 512 MB RAM (416 MB user-available)
  - WiFi only, no Ethernet
  - IP: 192.168.68.104

- **Jabra Speak 510**
  - USB speaker/microphone
  - Plug-and-play, no drivers needed
  - ALSA auto-detection

## Architecture

**Standalone (Option A):**
```
[Jabra Speak 510] → [Pi Audio] → [PyAudio] → [Groq STT] → [OpenAI GPT] → [espeak TTS] → [Audio Output]
```

**Why this stack:**
- Groq Whisper: Free tier, fast STT
- OpenAI GPT-4o-mini: $0.15/$0.60 per 1M tokens (26x cheaper than Claude)
- espeak: Lightweight TTS (~5 MB RAM)
- PyAudio: Python audio I/O for USB devices

## Memory Analysis (Before Deployment)

**Initial state:**
- Total RAM: 416 MB
- Used: 255 MB (61%)
- Available: 161 MB
- **Swap usage: 158 MB** (problematic)

**Memory consumers:**
- Monitoring stack (Prometheus/cAdvisor/node-exporter): 91.5 MB
- Docker runtime (dockerd/containerd/coredns): 75.7 MB
- System services (NetworkManager, ModemManager, etc.): 43.6 MB

**After cleanup:**
- Stopped monitoring containers: +91 MB
- Disabled Bluetooth/ModemManager: +2 MB
- **Result: 254 MB available**

**Voice assistant needs: ~80 MB**
- PyAudio + libs: 10 MB
- Groq/OpenAI clients: 15 MB
- Python overhead: 30 MB
- espeak: 5 MB
- Runtime buffers: 20 MB

**Headroom: 174 MB** ✓

## Files

### Core Files
- **voice_assistant_simple.py** - Main application (OpenAI version)
- **requirements.txt** - Python dependencies
- **setup.sh** - One-command setup for direct install

### Docker Files
- **Dockerfile** - ARM32v7 container build
- **.dockerignore** - Build optimization
- **docker-compose.yml** - Compose file for Pi deployment
- **.env.example** - Environment variables template
- **deploy.sh** - Build and push to registry
- **run-on-pi.sh** - Pull and run on Pi (alternative to compose)

### Documentation
- **README.md** - Setup and configuration
- **QUICKSTART.md** - Step-by-step deployment
- **OPENAI_VERSION.md** - Why OpenAI vs Claude
- **PROJECT.md** - This file

## Configuration

### Environment Variables
```bash
GROQ_API_KEY=<your_groq_key>
OPENAI_API_KEY=<your_openai_key>
ASSISTANT_NAME=Jarvis  # Optional
```

### Pi SSH Access
- Host: 192.168.68.104
- User: pi
- Pass: creative

### Registry
- URL: 192.168.68.168:3030
- Type: Docker Registry v2

## Deployment

### Method 1: Docker Compose (Recommended)

**On build server:**
```bash
cd /app/workspace/pi-voice-assistant
./deploy.sh
```

**On Raspberry Pi:**
```bash
# Copy docker-compose.yml and .env.example to Pi
scp docker-compose.yml .env.example pi@192.168.68.104:~/voice-assistant/

# SSH to Pi
ssh pi@192.168.68.104
cd ~/voice-assistant

# Set up environment
cp .env.example .env
nano .env  # Add your API keys

# Pull and start
docker-compose pull
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Method 2: Docker Run Script

**On Raspberry Pi:**
```bash
# Copy run-on-pi.sh to Pi
scp run-on-pi.sh pi@192.168.68.104:~/

# SSH to Pi
ssh pi@192.168.68.104

# Run
export GROQ_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
./run-on-pi.sh

# Check logs
docker logs -f jarvis
```

### Method 3: Direct Install

**On Raspberry Pi:**
```bash
# Copy files to Pi
scp -r /app/workspace/pi-voice-assistant/* pi@192.168.68.104:~/voice-assistant/

# SSH and install
ssh pi@192.168.68.104
cd ~/voice-assistant
export GROQ_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
./setup.sh

# Run
python3 voice_assistant_simple.py
```

## Testing

### 1. Verify Audio Device
```bash
arecord -l  # Should show Jabra Speak 510
aplay -l
```

### 2. Test Recording
```bash
arecord -D plughw:1,0 -d 5 test.wav
aplay test.wav
```

### 3. Test TTS
```bash
espeak "Hello, this is a test"
```

### 4. Test Container
```bash
docker logs -f jarvis  # Watch for "Listening for wake word..."
```

## Troubleshooting

### Issue: "No module named 'pyaudio'"
**Fix:** Install system package first
```bash
sudo apt-get install python3-pyaudio portaudio19-dev
```

### Issue: "ALSA device not found"
**Fix:** Check USB connection
```bash
lsusb | grep Jabra
arecord -l
```

### Issue: High memory usage / swap
**Fix:** Stop monitoring stack
```bash
docker stop $(docker ps -q)
sudo systemctl disable --now bluetooth ModemManager
```

### Issue: Container can't access audio
**Fix:** Ensure `--device /dev/snd` is in docker run command

### Issue: "Permission denied" on /dev/snd
**Fix:** Add user to audio group
```bash
sudo usermod -aG audio pi
# Logout and login again
```

## Performance Optimization

### 1. Reduce GPU Memory
Edit `/boot/firmware/config.txt`:
```
gpu_mem=16
```
Reboot. Saves 48 MB.

### 2. Disable Services
```bash
sudo systemctl disable --now bluetooth
sudo systemctl disable --now ModemManager
sudo systemctl disable --now avahi-daemon  # If not using .local names
```

### 3. Increase Swap (if needed)
```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## Cost Analysis

**OpenAI GPT-4o-mini:**
- Input: $0.15 per 1M tokens
- Output: $0.60 per 1M tokens

**Groq Whisper:**
- Free tier: Generous limits
- Paid: Very cheap if needed

**Estimated monthly cost:** < $1 for casual use

## Next Steps

1. Deploy container to Pi
2. Test wake word detection
3. Test full conversation flow
4. Set up systemd service for auto-start
5. Monitor memory usage under load
6. Tune wake word sensitivity

## Notes

- Pi runs Prometheus/cAdvisor for monitoring (stopped for voice assistant)
- WireGuard VPN container running on Pi
- coredns running (can be stopped if not needed)
- Monitoring can be re-enabled by restarting containers

## References

- Groq API: https://console.groq.com/
- OpenAI API: https://platform.openai.com/
- Jabra Speak 510: USB Audio Class compliant
- PyAudio docs: https://people.csail.mit.edu/hubert/pyaudio/
