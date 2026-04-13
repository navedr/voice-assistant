#!/usr/bin/env python3
"""
Voice Assistant for Raspberry Pi 3 + Jabra USB Speaker
Uses: Groq (STT), OpenAI (AI), Google Cloud Wavenet (TTS)
"""

import os
import io
import json
import wave
import time

# Load .env file if present (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import platform
import tempfile
import struct
import math
import subprocess
import logging
from groq import Groq
from openai import OpenAI
from google.cloud import texttospeech

# Audio input: pyaudio on Linux (Pi), sounddevice on macOS
try:
    import pyaudio
    import pyaudio._portaudio  # verify C module loads
    USE_SOUNDDEVICE = False
except (ImportError, OSError):
    import sounddevice as sd
    import numpy as np
    USE_SOUNDDEVICE = True

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("beans")
log.setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

ASSISTANT_NAME = os.environ.get("ASSISTANT_NAME", "Beans")
AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE", "")
WAKE_WORD = f"hey {ASSISTANT_NAME.lower()}"
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 2.0
HISTORY_FILE = os.environ.get("HISTORY_FILE", "/app/data/history.json")
MEMORY_FILE = os.environ.get("MEMORY_FILE", "/app/data/memory.json")
MAX_HISTORY_PAIRS = 5

# M4: Adaptive threshold constants
MAX_WAKE_WORDS = 20
NOISE_FLOOR_MIN = 150
NOISE_FLOOR_MAX = 1500
NOISE_FLOOR_MULTIPLIER = 1.5

# Whisper hallucinates these on silence/noise
HALLUCINATIONS = {
    "", ".", "..", "...", "thank you", "thanks", "thank you.", "thanks.",
    "you", "the", "bye", "bye.", "goodbye", "goodbye.", "okay", "okay.",
    "right", "so", "uh", "um", "hmm", "huh", "yeah", "yes",
    "thank you for watching", "thank you for watching.",
    "thanks for watching", "thanks for watching.",
    "please subscribe", "like and subscribe",
    "subtitles by the amara.org community",
}

# M3: Cancel and exit phrases
CANCEL_PHRASES = {
    "never mind", "nevermind", "cancel", "stop", "forget it", "nothing",
}
EXIT_PHRASES = {
    "that's all", "thats all", "thanks bye", "goodbye", "bye",
    "i'm done", "im done",
}

# M6: Kid-friendly system prompt
SYSTEM_PROMPT = (
    f"You are {ASSISTANT_NAME}, a friendly voice assistant for kids. "
    "Use simple words a five-year-old can understand. "
    "Answer in 1 to 3 short sentences. "
    "Never use markdown, bullet points, numbered lists, or special formatting. "
    "If someone asks something inappropriate or unsafe, gently change the subject "
    "to something fun instead."
)

# M4: Fuzzy wake word variations
def build_wake_variations(name):
    n = name.lower().strip()
    prefixes = ["hey", "hay", "a", "hey hey", "he"]
    suffixes = list(dict.fromkeys(
        s for s in [n, n[:-1], n + "s"] if s
    ))
    variations = set()
    for prefix in prefixes:
        for suffix in suffixes:
            variations.add(f"{prefix} {suffix}")
    for suffix in suffixes:
        variations.add(suffix)
    return variations

WAKE_VARIATIONS = build_wake_variations(ASSISTANT_NAME)

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# M5: Retry wrapper
def retry_api_call(fn, max_retries=2, delay=1.0):
    last_exception = None
    current_delay = delay
    for attempt in range(1 + max_retries):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                log.warning(f"  retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(current_delay)
                current_delay *= 2
            else:
                log.warning(f"  all {max_retries} retries exhausted: {e}")
    raise last_exception


def find_audio_device():
    if AUDIO_DEVICE:
        return AUDIO_DEVICE
    try:
        output = subprocess.check_output(
            ["arecord", "-l"], text=True, stderr=subprocess.DEVNULL
        )
        for line in output.splitlines():
            if "jabra" in line.lower():
                card = line.split("card")[1].split(":")[0].strip()
                return f"plughw:{card},0"
    except Exception:
        pass
    return "plughw:1,0"

IS_MACOS = platform.system() == "Darwin"


class VoiceAssistant:
    def __init__(self):
        self.audio = None if USE_SOUNDDEVICE else pyaudio.PyAudio()
        self.conversation_history = self._load_history()
        self.memories = self._load_memory()
        self.device = find_audio_device()
        self.last_interaction_time = time.time()
        self.current_threshold = SILENCE_THRESHOLD
        self.is_speaking = False

        # Defer TTS client init
        try:
            self.tts_client = texttospeech.TextToSpeechClient()
        except Exception as e:
            log.warning("Google TTS unavailable (%s), using espeak fallback", e)
            self.tts_client = None

        log.info(f"🎙️  Voice Assistant initialized")
        log.info(f"Wake word: '{WAKE_WORD}'")
        log.info(f"Audio device: {self.device}")
        self._startup_self_test()

    def _load_history(self):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
                log.info(f"Loaded {len(history)} messages from history")
                return history
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_history(self):
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        # Keep last N pairs (2 messages per pair)
        history = self.conversation_history[-(MAX_HISTORY_PAIRS * 2):]
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)

    def _load_memory(self):
        try:
            with open(MEMORY_FILE, 'r') as f:
                memories = json.load(f)
                log.info(f"Loaded {len(memories)} memories")
                return memories
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_memory(self, memory):
        memories = self._load_memory()
        memories.append(memory)
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        with open(MEMORY_FILE, 'w') as f:
            json.dump(memories, f, indent=2)

    def _open_mic(self):
        """Open mic stream — returns (stream, read_fn, close_fn)."""
        if USE_SOUNDDEVICE:
            buffer = []
            def callback(indata, frames, time_info, status):
                buffer.append(bytes(indata))
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE,
                channels=CHANNELS, dtype='int16', callback=callback
            )
            stream.start()
            def read_chunk(_=None):
                while not buffer:
                    time.sleep(0.01)
                return buffer.pop(0)
            def close():
                stream.stop()
                stream.close()
            return stream, read_chunk, close
        else:
            stream = self.audio.open(
                format=pyaudio.paInt16, channels=CHANNELS,
                rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE
            )
            def read_chunk(_=None):
                return stream.read(CHUNK_SIZE, exception_on_overflow=False)
            def close():
                stream.stop_stream()
                stream.close()
            return stream, read_chunk, close

    def _get_sample_width(self):
        if USE_SOUNDDEVICE:
            return 2  # int16 = 2 bytes
        return self.audio.get_sample_size(pyaudio.paInt16)

    def _startup_self_test(self):
        # Set USB speaker volume to max
        try:
            card = self.device.split(":")[1].split(",")[0] if ":" in self.device else "1"
            subprocess.run(["amixer", "-c", card, "sset", "PCM", "90%"], capture_output=True)
            log.info("🔊 Speaker volume set to max")
        except Exception:
            pass

        try:
            _, read_fn, close_fn = self._open_mic()
            read_fn()
            close_fn()
            log.info("✅ Audio self-test passed")
        except Exception as e:
            log.error(f" Audio self-test FAILED: {e}")
        self.speak(f"{ASSISTANT_NAME} is ready!")

    def flush_mic(self, duration=0.5):
        """Discard buffered mic audio to prevent self-hearing."""
        try:
            _, read_fn, close_fn = self._open_mic()
            for _ in range(int(duration * SAMPLE_RATE / CHUNK_SIZE)):
                read_fn()
            close_fn()
        except Exception:
            pass

    def _play_wav(self, path):
        if IS_MACOS:
            subprocess.run(["afplay", path], stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["aplay", "-D", self.device, path], stderr=subprocess.DEVNULL)

    def play_beep(self):
        self.is_speaking = True
        beep_path = "activate.wav" if IS_MACOS else "/app/activate.wav"
        self._play_wav(beep_path)
        self.is_speaking = False

    def get_rms(self, data):
        count = len(data) // 2
        shorts = struct.unpack("%dh" % count, data)
        sum_squares = sum(s ** 2 for s in shorts)
        return math.sqrt(sum_squares / count)

    # M4: Adaptive noise calibration
    def calibrate_noise_floor(self, read_fn, duration=1.0):
        num_chunks = int(duration * SAMPLE_RATE / CHUNK_SIZE)
        total_rms = 0.0
        for _ in range(num_chunks):
            data = read_fn()
            total_rms += self.get_rms(data)
        avg_rms = total_rms / max(num_chunks, 1)
        threshold = NOISE_FLOOR_MULTIPLIER * avg_rms
        return int(max(NOISE_FLOOR_MIN, min(NOISE_FLOOR_MAX, threshold)))

    # M4: Fuzzy wake word matching
    def is_wake_word(self, text):
        text = text.lower().strip()
        if len(text.split()) > MAX_WAKE_WORDS:
            return False
        # Normalize punctuation so "hey, beans" matches "hey beans"
        normalized = text.replace(",", "").replace("!", "").replace(".", "")
        for check in (text, normalized):
            for variation in sorted(WAKE_VARIATIONS, key=len, reverse=True):
                if variation in check:
                    # Extract everything after the wake word
                    after = check.split(variation, 1)[1].strip().strip(".,!?")
                    return after if after else True
        return False

    def detect_wake_word(self):
        try:
            _, read_fn, close_fn = self._open_mic()
        except OSError:
            log.warning(" Audio device error, re-detecting...")
            self.device = find_audio_device()
            log.info(f"Audio device: {self.device}")
            return False

        # M4: Adaptive threshold from ambient noise
        self.current_threshold = self.calibrate_noise_floor(read_fn, duration=1.0)
        log.info(f"👂 Listening (threshold: {self.current_threshold})")

        pre_buffer = []  # Rolling buffer to capture audio before threshold crossed
        pre_buffer_size = int(1.0 * SAMPLE_RATE / CHUNK_SIZE)  # ~1 second
        frames = []
        recording_started = False
        silence_frames = 0
        max_recording_frames = int(5 * SAMPLE_RATE / CHUNK_SIZE)

        max_wait_frames = int(10 * SAMPLE_RATE / CHUNK_SIZE)  # Max 10s waiting for speech

        try:
            frame_count = 0
            total_frames = 0
            while frame_count < max_recording_frames:
                data = read_fn()
                rms = self.get_rms(data)
                total_frames += 1

                if rms > self.current_threshold and not recording_started:
                    recording_started = True
                    frames = list(pre_buffer)

                if not recording_started:
                    pre_buffer.append(data)
                    if len(pre_buffer) > pre_buffer_size:
                        pre_buffer.pop(0)
                    if total_frames >= max_wait_frames:
                        break
                    continue

                frames.append(data)
                frame_count += 1

                if rms < self.current_threshold:
                    silence_frames += 1
                else:
                    silence_frames = 0

                if silence_frames > int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE):
                    break
        finally:
            close_fn()

        if not frames or len(frames) < 5:
            return False

        # Transcribe with Groq (M5: with retry)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wf = wave.open(f.name, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self._get_sample_width())
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            try:
                with open(f.name, 'rb') as audio_file:
                    audio_data = audio_file.read()

                def transcribe():
                    return groq_client.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=("audio.wav", io.BytesIO(audio_data)),
                        language="en"
                    )

                transcription = retry_api_call(transcribe)
                text = transcription.text.lower().strip()

                if text in HALLUCINATIONS or len(text.strip(".,!? ")) < 3:
                    return False
                log.info(f"Heard: {text}")

                # M4: Fuzzy wake word check
                return self.is_wake_word(text)
            except Exception as e:
                log.warning(f"Transcription error: {e}")
                return False
            finally:
                os.unlink(f.name)

    def record_command(self, max_seconds=7):
        log.info("🎤 Listening for command...")

        try:
            _, read_fn, close_fn = self._open_mic()
        except OSError:
            log.warning(" Audio device error, re-detecting...")
            self.device = find_audio_device()
            return None

        frames = []
        speech_detected = False
        silence_frames = 0
        max_recording_frames = int(max_seconds * SAMPLE_RATE / CHUNK_SIZE)
        silence_threshold = max(self.current_threshold, SILENCE_THRESHOLD)

        try:
            for _ in range(max_recording_frames):
                data = read_fn()
                rms = self.get_rms(data)
                frames.append(data)

                if rms > silence_threshold:
                    speech_detected = True
                    silence_frames = 0
                elif speech_detected:
                    silence_frames += 1
                    if silence_frames > int(1.5 * SAMPLE_RATE / CHUNK_SIZE):
                        break
        finally:
            close_fn()

        if not frames or len(frames) < 5:
            return None

        # Transcribe (M5: with retry)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wf = wave.open(f.name, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self._get_sample_width())
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            try:
                with open(f.name, 'rb') as audio_file:
                    audio_data = audio_file.read()

                def transcribe():
                    return groq_client.audio.transcriptions.create(
                        model="whisper-large-v3",
                        file=("audio.wav", io.BytesIO(audio_data)),
                        language="en"
                    )

                transcription = retry_api_call(transcribe)
                command = transcription.text.strip()
                log.info(f"Command: {command}")
            except Exception as e:
                log.warning(f"Transcription error: {e}")
                command = None
            finally:
                os.unlink(f.name)

        return command

    def get_ai_response(self, user_message):
        log.info("🤖 Thinking...")

        self.conversation_history.append({
            "role": "user", "content": user_message
        })

        # Build system prompt with memories
        prompt = SYSTEM_PROMPT
        if self.memories:
            prompt += "\n\nThings you have been asked to remember:\n"
            prompt += "\n".join(f"- {m}" for m in self.memories)

        messages = [
            {"role": "system", "content": prompt}
        ] + self.conversation_history

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "remember",
                    "description": "Save something to memory when the user asks you to remember it",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fact": {"type": "string", "description": "The fact to remember"}
                        },
                        "required": ["fact"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "forget",
                    "description": "Remove something from memory when the user asks you to forget it",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fact": {"type": "string", "description": "The fact to forget (matched against existing memories)"}
                        },
                        "required": ["fact"]
                    }
                }
            }
        ]

        attempt_count = 0

        def call_gpt():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 2:
                if IS_MACOS:
                    subprocess.run(["say", "Hold on"], stderr=subprocess.DEVNULL)
                else:
                    espeak = subprocess.Popen(
                        ["espeak", "-s", "150", "-v", "en-us", "--stdout", "Hold on"],
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                    )
                    subprocess.run(
                        ["aplay", "-D", self.device],
                        stdin=espeak.stdout, stderr=subprocess.DEVNULL
                    )
                    espeak.wait()
            return openai_client.chat.completions.create(
                model="gpt-5.4-nano",
                max_completion_tokens=200,
                messages=messages,
                tools=tools,
            )

        try:
            response = retry_api_call(call_gpt)
            message = response.choices[0].message
            reply = message.content or ""

            # Handle tool calls (remember/forget)
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    if tool_call.function.name == "remember":
                        fact = args["fact"]
                        self._save_memory(fact)
                        self.memories.append(fact)
                        log.info(f"💾 Saved memory: {fact}")
                        if not reply:
                            reply = "Got it, I'll remember that!"
                    elif tool_call.function.name == "forget":
                        fact = args["fact"].lower()
                        self.memories = [m for m in self.memories if fact not in m.lower()]
                        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
                        with open(MEMORY_FILE, 'w') as f:
                            json.dump(self.memories, f, indent=2)
                        log.info(f"🗑️ Forgot memory matching: {fact}")
                        if not reply:
                            reply = "Okay, I've forgotten that!"

            self.conversation_history.append({
                "role": "assistant", "content": reply
            })

            # Keep last N pairs
            if len(self.conversation_history) > MAX_HISTORY_PAIRS * 2:
                self.conversation_history = self.conversation_history[-(MAX_HISTORY_PAIRS * 2):]
            self._save_history()

            log.info(f"Response: {reply}")
            return reply
        except Exception as e:
            log.warning(f"AI error: {e}")
            return "I'm having trouble connecting right now."

    def speak(self, text):
        log.info(f"🔊 Speaking: {text}")
        self.is_speaking = True

        # Try Groq TTS first (fastest)
        try:
            def call_groq_tts():
                return groq_client.audio.speech.create(
                    model="canopylabs/orpheus-v1-english",
                    voice="hannah",
                    input=text,
                    response_format="wav"
                )

            response = retry_api_call(call_groq_tts)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                response.write_to_file(f.name)
                self._play_wav(f.name)
                os.unlink(f.name)
            self.is_speaking = False
            return
        except Exception as e:
            log.warning(f"Groq TTS error: {e}, trying Google TTS")

        # Google Cloud TTS fallback
        if self.tts_client:
            try:
                synthesis_input = texttospeech.SynthesisInput(text=text)
                voice = texttospeech.VoiceSelectionParams(
                    language_code="en-US", name="en-US-Wavenet-F",
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    sample_rate_hertz=24000,
                )

                def call_google_tts():
                    return self.tts_client.synthesize_speech(
                        input=synthesis_input, voice=voice,
                        audio_config=audio_config
                    )

                response = retry_api_call(call_google_tts)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(response.audio_content)
                    self._play_wav(f.name)
                    os.unlink(f.name)
                self.is_speaking = False
                return
            except Exception as e:
                log.warning(f"Google TTS error: {e}, falling back to espeak")

        # espeak/say fallback (subprocess with list args — no shell injection)
        if IS_MACOS:
            subprocess.run(["say", text], stderr=subprocess.DEVNULL)
        else:
            espeak = subprocess.Popen(
                ["espeak", "-s", "150", "-v", "en-us", "--stdout", text],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            subprocess.run(
                ["aplay", "-D", self.device],
                stdin=espeak.stdout, stderr=subprocess.DEVNULL
            )
            espeak.wait()
        self.is_speaking = False

    def run(self):
        log.info("\n" + "=" * 50)
        log.info(f"✅ {ASSISTANT_NAME} Ready!")
        log.info(f"Say '{WAKE_WORD}' to activate")
        log.info("Press Ctrl+C to exit")
        log.info("=" * 50 + "\n")

        try:
            while True:
                # M2: Session timeout — clear stale history
                if time.time() - self.last_interaction_time > 300:
                    if self.conversation_history:
                        log.info("Session timeout — clearing conversation history")
                        self.conversation_history = []
                        self._save_history()

                result = self.detect_wake_word()
                if result:
                    if isinstance(result, str):
                        # Inline command — beep and go straight to GPT
                        self.play_beep()
                        command = result
                    else:
                        # Wake word only — beep, then listen
                        self.play_beep()
                        command = self.record_command()

                    if command:
                        # M3: Check cancel phrases
                        if command.lower().strip().strip(".,!?") in CANCEL_PHRASES:
                            self.speak("Okay!")
                            self.last_interaction_time = time.time()
                            continue

                        self.last_interaction_time = time.time()
                        response = self.get_ai_response(command)
                        self.speak(response)

                        # M2: Conversation mode — listen for follow-ups
                        while True:
                            self.flush_mic()  # Discard echo of TTS playback
                            log.info("👂 Listening for follow-up (5s)...")
                            follow_up = self.record_command(max_seconds=8)

                            if not follow_up:
                                log.info("No follow-up, returning to wake word")
                                break

                            follow_lower = follow_up.lower().strip().strip(".,!?")

                            # M3: Exit phrases end conversation mode
                            if follow_lower in EXIT_PHRASES:
                                self.speak("Talk to you later!")
                                self.last_interaction_time = time.time()
                                break

                            # M3: Cancel phrases end conversation mode
                            if follow_lower in CANCEL_PHRASES:
                                self.speak("Okay!")
                                self.last_interaction_time = time.time()
                                break

                            # Filter hallucinations in follow-up
                            if follow_lower in HALLUCINATIONS or len(follow_lower.strip(".,!? ")) < 3:
                                break

                            # Handle wake word said during follow-up
                            wake_result = self.is_wake_word(follow_lower)
                            if wake_result is not False:
                                if isinstance(wake_result, str):
                                    follow_up = wake_result
                                else:
                                    continue  # Just wake word, keep listening

                            self.last_interaction_time = time.time()
                            response = self.get_ai_response(follow_up)
                            self.speak(response)
                    else:
                        # M3: Friendlier message
                        self.speak("I'm here if you need me!")

        except KeyboardInterrupt:
            log.info("\n\n👋 Shutting down...")
        finally:
            if self.audio:
                self.audio.terminate()


if __name__ == "__main__":
    if not GROQ_API_KEY:
        log.error("GROQ_API_KEY not set")
        exit(1)

    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set")
        exit(1)

    assistant = VoiceAssistant()
    assistant.run()
