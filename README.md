# Voice Assistant for Raspberry Pi 3

A voice-activated AI assistant using:
- **Jabra Speak 510** for audio I/O
- **Groq** for speech-to-text (Whisper)
- **Claude** for AI responses
- **Google TTS** or **espeak** for text-to-speech

## Hardware Requirements

- Raspberry Pi 3 (or newer)
- Jabra Speak 510 USB speakerphone
- Internet connection

## Setup Instructions

### 1. Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-pyaudio \
    portaudio19-dev \
    espeak \
    mpg123 \
    alsa-utils
```

### 2. Configure Jabra Speak 510

Plug in your Jabra Speak 510 and verify it's detected:

```bash
arecord -l   # List capture devices
aplay -l     # List playback devices
```

Find your Jabra device number and set it as default:

```bash
# Test recording
arecord -D plughw:1,0 -d 5 test.wav
aplay test.wav

# Test playback
speaker-test -D plughw:1,0 -c2
```

### 3. Install Python Dependencies

```bash
cd /app/pi-voice-assistant
pip3 install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file:

```bash
export GROQ_API_KEY="your_groq_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
# Optional for Google TTS (or use espeak fallback)
export GOOGLE_API_KEY="your_google_api_key"
```

Load environment:
```bash
source .env
```

### 5. Run the Assistant

```bash
python3 voice_assistant.py
```

## Usage

1. Wait for "Voice Assistant ready!"
2. Say **"Hey Jarvis"** to activate
3. Wait for the beep/response
4. Speak your command
5. Assistant will respond

## Customization

### Change Wake Word

Edit `voice_assistant.py`:
```python
WAKE_WORD = "hey jarvis"  # Change to your preferred wake word
```

### Adjust Sensitivity

```python
SILENCE_THRESHOLD = 500  # Lower = more sensitive
SILENCE_DURATION = 2.0   # Seconds of silence to end recording
```

### Use Different TTS

**Option 1: espeak (lightweight, works offline)**
```python
def speak(self, text):
    os.system(f'espeak "{text}"')
```

**Option 2: Piper (better quality, local)**
```bash
# Install Piper
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz
tar -xzf piper_arm64.tar.gz

# Download voice
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx
```

```python
def speak(self, text):
    os.system(f'echo "{text}" | ./piper --model en_US-lessac-medium.onnx --output-raw | aplay -r 22050 -f S16_LE')
```

## Troubleshooting

### No audio input detected
```bash
# Check ALSA configuration
arecord -L
# Set default device in ~/.asoundrc
```

### Groq API errors
- Check your API key
- Verify internet connection
- Check Groq API status

### High CPU usage
- Use smaller Groq model (if available)
- Increase SILENCE_THRESHOLD
- Reduce SAMPLE_RATE to 8000

## Running on Boot

Create systemd service:

```bash
sudo nano /etc/systemd/system/voice-assistant.service
```

```ini
[Unit]
Description=Voice Assistant
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/app/pi-voice-assistant
Environment="GROQ_API_KEY=your_key"
Environment="ANTHROPIC_API_KEY=your_key"
ExecStart=/usr/bin/python3 /app/pi-voice-assistant/voice_assistant.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable voice-assistant
sudo systemctl start voice-assistant
```

## Performance Tips

1. **Disable GUI** (if not needed):
   ```bash
   sudo systemctl set-default multi-user.target
   ```

2. **Overclock Pi 3** (optional):
   Edit `/boot/config.txt`:
   ```
   arm_freq=1350
   over_voltage=4
   ```

3. **Use lightweight TTS** (espeak instead of Google)

4. **Batch requests** if making multiple API calls

## License

MIT
