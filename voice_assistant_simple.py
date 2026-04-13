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
import urllib.request
from datetime import datetime

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

try:
    import pyaudio
    import pyaudio._portaudio
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

# API timeouts
API_TIMEOUT_STT = 15
API_TIMEOUT_GPT = 20
API_TIMEOUT_TTS = 15

# Adaptive threshold
MAX_WAKE_WORDS = 20
NOISE_FLOOR_MIN = 150
NOISE_FLOOR_MAX = 1500
NOISE_FLOOR_MULTIPLIER = 1.5

# Weather
WEATHER_LAT = os.environ.get("WEATHER_LAT", "33.9806")
WEATHER_LON = os.environ.get("WEATHER_LON", "-118.1506")
CONFIRMATION_ECHO = os.environ.get("CONFIRMATION_ECHO", "false").lower() == "true"

WMO_CODES = {
    0: "clear skies", 1: "mostly clear", 2: "partly cloudy", 3: "cloudy",
    45: "foggy", 48: "foggy", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "heavy rain showers", 85: "snow showers", 95: "thunderstorms",
}

# Quiet hours
QUIET_HOURS_ENABLED = os.environ.get("QUIET_HOURS_ENABLED", "false").lower() == "true"
QUIET_HOURS_START = int(os.environ.get("QUIET_HOURS_START", "21"))
QUIET_HOURS_END = int(os.environ.get("QUIET_HOURS_END", "7"))
MAX_CRASH_RESTARTS = 5
CRASH_COOLDOWN = 10

# Whisper hallucinations
HALLUCINATIONS = {
    "", ".", "..", "...", "thank you", "thanks", "thank you.", "thanks.",
    "you", "the", "bye", "bye.", "goodbye", "goodbye.", "okay", "okay.",
    "right", "so", "uh", "um", "hmm", "huh", "yeah", "yes",
    "thank you for watching", "thank you for watching.",
    "thanks for watching", "thanks for watching.",
    "please subscribe", "like and subscribe",
    "subtitles by the amara.org community",
}

CANCEL_PHRASES = {
    "never mind", "nevermind", "cancel", "stop", "forget it", "nothing",
}
EXIT_PHRASES = {
    "that's all", "thats all", "thanks bye", "goodbye", "bye",
    "i'm done", "im done",
}

SYSTEM_PROMPT = (
    f"You are {ASSISTANT_NAME}, a friendly voice assistant for kids. "
    "Use simple words a five-year-old can understand. "
    "Answer in 1 to 3 short sentences. "
    "Never use markdown, bullet points, numbered lists, or special formatting. "
    "If someone asks something inappropriate or unsafe, gently change the subject "
    "to something fun instead."
)

# GPT tools
TOOLS = [
    {"type": "function", "function": {
        "name": "remember",
        "description": "Save something to memory when the user asks you to remember it",
        "parameters": {"type": "object", "properties": {
            "fact": {"type": "string", "description": "The fact to remember"}
        }, "required": ["fact"]}}},
    {"type": "function", "function": {
        "name": "forget",
        "description": "Remove something from memory when the user asks you to forget it",
        "parameters": {"type": "object", "properties": {
            "fact": {"type": "string", "description": "The fact to forget"}
        }, "required": ["fact"]}}},
    {"type": "function", "function": {
        "name": "set_timer",
        "description": "Set a countdown timer. Convert spoken duration to seconds (e.g., '5 minutes' = 300).",
        "parameters": {"type": "object", "properties": {
            "duration_seconds": {"type": "integer", "description": "Timer duration in seconds"},
            "label": {"type": "string", "description": "Name/label for the timer"}
        }, "required": ["duration_seconds", "label"]}}},
    {"type": "function", "function": {
        "name": "check_timers",
        "description": "Check active timers and time remaining.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "cancel_timer",
        "description": "Cancel an active timer by name.",
        "parameters": {"type": "object", "properties": {
            "label": {"type": "string", "description": "Name of the timer to cancel"}
        }, "required": ["label"]}}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Get current date and time. Call when user asks what time or day it is.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get current weather and forecast. Call when user asks about weather, temperature, or what to wear.",
        "parameters": {"type": "object", "properties": {}}}},
]

def build_wake_variations(name):
    n = name.lower().strip()
    prefixes = ["hey", "hay", "a", "hey hey", "he"]
    suffixes = list(dict.fromkeys(s for s in [n, n[:-1], n + "s"] if s))
    variations = set()
    for prefix in prefixes:
        for suffix in suffixes:
            variations.add(f"{prefix} {suffix}")
    for suffix in suffixes:
        variations.add(suffix)
    return variations

WAKE_VARIATIONS = build_wake_variations(ASSISTANT_NAME)

groq_client = Groq(api_key=GROQ_API_KEY, timeout=API_TIMEOUT_STT)
openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=API_TIMEOUT_GPT)


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
        self.active_timers = []
        self._ensure_audio_cues()

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

    def _save_memory(self):
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        with open(MEMORY_FILE, 'w') as f:
            json.dump(self.memories, f, indent=2)

    def _generate_tone(self, freq_sequence, sample_rate=44100):
        if not hasattr(self, "_tone_cache"):
            self._tone_cache = {}
        cache_key = tuple(freq_sequence)
        if cache_key in self._tone_cache:
            return self._tone_cache[cache_key]

        samples = []
        for freq_hz, duration_ms, pause_ms in freq_sequence:
            n = int(sample_rate * duration_ms / 1000)
            fade = min(n // 8, 200)
            for i in range(n):
                val = math.sin(2 * math.pi * freq_hz * i / sample_rate)
                if i < fade:
                    val *= i / fade
                elif i > n - fade:
                    val *= (n - i) / fade
                samples.append(int(val * 16000))
            samples.extend([0] * int(sample_rate * pause_ms / 1000))

        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wf = wave.open(f.name, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        wf.close()
        f.close()
        self._tone_cache[cache_key] = f.name
        return f.name

    def _ensure_audio_cues(self):
        self._thinking_wav = self._generate_tone([(523, 100, 50), (659, 100, 0)])
        self._error_wav = self._generate_tone([(330, 150, 50), (220, 150, 0)])
        self._timer_alarm_wav = self._generate_tone([
            (880, 150, 100), (880, 150, 100), (880, 150, 0)
        ])

    def _open_mic(self):
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
            return 2
        return self.audio.get_sample_size(pyaudio.paInt16)

    def _startup_self_test(self):
        if not IS_MACOS:
            try:
                card = self.device.split(":")[1].split(",")[0] if ":" in self.device else "1"
                subprocess.run(["amixer", "-c", card, "sset", "PCM", "90%"], capture_output=True)
                log.info("🔊 Speaker volume set to 90%")
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
        try:
            _, read_fn, close_fn = self._open_mic()
            for _ in range(int(duration * SAMPLE_RATE / CHUNK_SIZE)):
                read_fn()
            close_fn()
        except Exception:
            pass

    def _play_wav(self, path):
        if IS_MACOS:
            subprocess.run(["afplay", path], stderr=subprocess.DEVNULL, timeout=10)
        else:
            subprocess.run(["aplay", "-D", self.device, path], stderr=subprocess.DEVNULL, timeout=10)

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

    def calibrate_noise_floor(self, read_fn, duration=1.0):
        num_chunks = int(duration * SAMPLE_RATE / CHUNK_SIZE)
        total_rms = 0.0
        for _ in range(num_chunks):
            data = read_fn()
            total_rms += self.get_rms(data)
        avg_rms = total_rms / max(num_chunks, 1)
        threshold = NOISE_FLOOR_MULTIPLIER * avg_rms
        return int(max(NOISE_FLOOR_MIN, min(NOISE_FLOOR_MAX, threshold)))

    def is_wake_word(self, text):
        text = text.lower().strip()
        if len(text.split()) > MAX_WAKE_WORDS:
            return False
        normalized = text.replace(",", "").replace("!", "").replace(".", "")
        for check in (text, normalized):
            for variation in sorted(WAKE_VARIATIONS, key=len, reverse=True):
                if variation in check:
                    after = check.split(variation, 1)[1].strip().strip(".,!?")
                    return after if after else True
        return False

    def _is_quiet_hours(self):
        if not QUIET_HOURS_ENABLED:
            return False
        hour = datetime.now().hour
        if QUIET_HOURS_START > QUIET_HOURS_END:
            return hour >= QUIET_HOURS_START or hour < QUIET_HOURS_END
        return QUIET_HOURS_START <= hour < QUIET_HOURS_END

    def _check_expired_timers(self):
        now = time.time()
        expired = [t for t in self.active_timers if now >= t["end_time"]]
        if expired:
            self.active_timers = [t for t in self.active_timers if now < t["end_time"]]
            for timer in expired:
                self._play_wav(self._timer_alarm_wav)
                self.speak(f"Time's up! Your {timer['label']} timer is done!")

    def detect_wake_word(self):
        try:
            _, read_fn, close_fn = self._open_mic()
        except OSError:
            log.warning(" Audio device error, re-detecting...")
            self.device = find_audio_device()
            log.info(f"Audio device: {self.device}")
            return False

        self.current_threshold = self.calibrate_noise_floor(read_fn, duration=1.0)
        log.info(f"👂 Listening (threshold: {self.current_threshold})")

        pre_buffer = []
        pre_buffer_size = int(1.0 * SAMPLE_RATE / CHUNK_SIZE)
        frames = []
        recording_started = False
        silence_frames = 0
        max_recording_frames = int(5 * SAMPLE_RATE / CHUNK_SIZE)
        max_wait_frames = int(10 * SAMPLE_RATE / CHUNK_SIZE)

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
        # Play thinking chime (non-blocking, cleaned up in except)
        if IS_MACOS:
            chime = subprocess.Popen(["afplay", self._thinking_wav], stderr=subprocess.DEVNULL)
        else:
            chime = subprocess.Popen(["aplay", "-D", self.device, self._thinking_wav], stderr=subprocess.DEVNULL)

        self.conversation_history.append({"role": "user", "content": user_message})

        prompt = SYSTEM_PROMPT
        if self.memories:
            prompt += "\n\nThings you have been asked to remember:\n"
            prompt += "\n".join(f"- {m}" for m in self.memories)

        messages = [{"role": "system", "content": prompt}] + self.conversation_history

        attempt_count = 0

        def call_gpt():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 2:
                if IS_MACOS:
                    subprocess.run(["say", "Hold on"], stderr=subprocess.DEVNULL, timeout=10)
                else:
                    espeak = subprocess.Popen(
                        ["espeak", "-s", "150", "-v", "en-us", "--stdout", "Hold on"],
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                    )
                    subprocess.run(["aplay", "-D", self.device],
                                   stdin=espeak.stdout, stderr=subprocess.DEVNULL)
                    espeak.wait(timeout=10)
            return openai_client.chat.completions.create(
                model="gpt-5.4-nano",
                max_completion_tokens=200,
                messages=messages,
                tools=TOOLS,
            )

        try:
            response = retry_api_call(call_gpt)
            message = response.choices[0].message
            reply = message.content or ""

            if message.tool_calls:
                for tc in message.tool_calls:
                    args = json.loads(tc.function.arguments)
                    name = tc.function.name

                    if name == "remember":
                        fact = args["fact"]
                        self.memories.append(fact)
                        self._save_memory()
                        log.info(f"💾 Saved memory: {fact}")
                        if not reply:
                            reply = "Got it, I'll remember that!"

                    elif name == "forget":
                        fact = args["fact"].lower()
                        self.memories = [m for m in self.memories if fact not in m.lower()]
                        self._save_memory()
                        log.info(f"🗑️ Forgot memory matching: {fact}")
                        if not reply:
                            reply = "Okay, I've forgotten that!"

                    elif name == "set_timer":
                        label = args.get("label", "timer")
                        duration = args["duration_seconds"]
                        self.active_timers.append({"label": label, "end_time": time.time() + duration})
                        log.info(f"⏱️ Timer set: {label} for {duration}s")
                        if not reply:
                            reply = f"Timer set for {label}!"

                    elif name == "check_timers":
                        if self.active_timers:
                            now = time.time()
                            parts = []
                            for t in self.active_timers:
                                remaining = int(t["end_time"] - now)
                                mins, secs = divmod(max(remaining, 0), 60)
                                if mins > 0:
                                    parts.append(f"{t['label']}: {mins} minutes {secs} seconds left")
                                else:
                                    parts.append(f"{t['label']}: {secs} seconds left")
                            reply = ". ".join(parts)
                        else:
                            reply = "No timers running."

                    elif name == "cancel_timer":
                        label = args.get("label", "").lower()
                        before = len(self.active_timers)
                        self.active_timers = [t for t in self.active_timers if t["label"].lower() != label]
                        if len(self.active_timers) < before:
                            log.info(f"⏱️ Timer cancelled: {label}")
                            if not reply:
                                reply = "Timer cancelled!"
                        elif not reply:
                            reply = f"I don't see a timer called {label}."

                    elif name == "get_current_time":
                        now = datetime.now()
                        if not reply:
                            reply = now.strftime("It's %I:%M %p on %A, %B %d.")

                    elif name == "get_weather":
                        try:
                            url = (
                                f"https://api.open-meteo.com/v1/forecast?"
                                f"latitude={WEATHER_LAT}&longitude={WEATHER_LON}"
                                f"&current=temperature_2m,weather_code,wind_speed_10m"
                                f"&daily=temperature_2m_max,temperature_2m_min,weather_code"
                                f"&temperature_unit=fahrenheit&wind_speed_unit=mph"
                                f"&timezone=auto&forecast_days=2"
                            )
                            resp = urllib.request.urlopen(url, timeout=5)
                            data = json.loads(resp.read().decode())
                            current = data["current"]
                            daily = data["daily"]
                            temp = round(current["temperature_2m"])
                            desc = WMO_CODES.get(current["weather_code"], "unknown conditions")
                            high = round(daily["temperature_2m_max"][0])
                            low = round(daily["temperature_2m_min"][0])
                            if not reply:
                                reply = (
                                    f"Right now it's {temp} degrees with {desc}. "
                                    f"Today's high is {high} and low is {low} degrees."
                                )
                        except Exception as e:
                            log.warning(f"Weather API error: {e}")
                            if not reply:
                                reply = "I couldn't check the weather right now."

            self.conversation_history.append({"role": "assistant", "content": reply})

            if len(self.conversation_history) > MAX_HISTORY_PAIRS * 2:
                self.conversation_history = self.conversation_history[-(MAX_HISTORY_PAIRS * 2):]
            self._save_history()

            log.info(f"Response: {reply}")
            return reply
        except Exception as e:
            log.warning(f"AI error: {e}")
            chime.wait(timeout=2)
            self._play_wav(self._error_wav)
            return "I'm having trouble connecting right now."

    def speak(self, text):
        log.info(f"🔊 Speaking: {text}")
        self.is_speaking = True

        try:
            def call_groq_tts():
                return groq_client.audio.speech.create(
                    model="canopylabs/orpheus-v1-english",
                    voice="hannah", input=text, response_format="wav"
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
                        audio_config=audio_config, timeout=API_TIMEOUT_TTS
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

        if IS_MACOS:
            subprocess.run(["say", text], stderr=subprocess.DEVNULL, timeout=30)
        else:
            espeak = subprocess.Popen(
                ["espeak", "-s", "150", "-v", "en-us", "--stdout", text],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            subprocess.run(["aplay", "-D", self.device],
                           stdin=espeak.stdout, stderr=subprocess.DEVNULL)
            espeak.wait(timeout=10)
        self.is_speaking = False

    def run(self):
        log.info("\n" + "=" * 50)
        log.info(f"✅ {ASSISTANT_NAME} Ready!")
        log.info(f"Say '{WAKE_WORD}' to activate")
        log.info("Press Ctrl+C to exit")
        log.info("=" * 50 + "\n")

        while True:
            # Session timeout
            if time.time() - self.last_interaction_time > 300:
                if self.conversation_history:
                    log.info("Session timeout — clearing conversation history")
                    self.conversation_history = []
                    self._save_history()

            # Check expired timers
            self._check_expired_timers()

            result = self.detect_wake_word()

            # Quiet hours
            if result and self._is_quiet_hours():
                log.info("Quiet hours — ignoring wake word")
                continue

            if result:
                if isinstance(result, str):
                    self.play_beep()
                    command = result
                else:
                    self.play_beep()
                    command = self.record_command()

                if CONFIRMATION_ECHO and command:
                    self.speak(f"I heard: {command}")

                if command:
                    if command.lower().strip().strip(".,!?") in CANCEL_PHRASES:
                        self.speak("Okay!")
                        self.last_interaction_time = time.time()
                        continue

                    self.last_interaction_time = time.time()
                    response = self.get_ai_response(command)
                    self.speak(response)

                    # Conversation mode
                    while True:
                        self.flush_mic()
                        log.info("👂 Listening for follow-up...")
                        follow_up = self.record_command(max_seconds=8)

                        if not follow_up:
                            log.info("No follow-up, returning to wake word")
                            break

                        follow_lower = follow_up.lower().strip().strip(".,!?")

                        if follow_lower in EXIT_PHRASES:
                            self.speak("Talk to you later!")
                            self.last_interaction_time = time.time()
                            break

                        if follow_lower in CANCEL_PHRASES:
                            self.speak("Okay!")
                            self.last_interaction_time = time.time()
                            break

                        if follow_lower in HALLUCINATIONS or len(follow_lower.strip(".,!? ")) < 3:
                            break

                        wake_result = self.is_wake_word(follow_lower)
                        if wake_result is not False:
                            if isinstance(wake_result, str):
                                follow_up = wake_result
                            else:
                                continue

                        self.last_interaction_time = time.time()
                        response = self.get_ai_response(follow_up)
                        self.speak(response)
                else:
                    self.speak("I'm here if you need me!")


if __name__ == "__main__":
    if not GROQ_API_KEY:
        log.error("GROQ_API_KEY not set")
        exit(1)
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set")
        exit(1)

    for restart in range(MAX_CRASH_RESTARTS):
        try:
            assistant = VoiceAssistant()
            if restart > 0:
                assistant.speak("I had a hiccup, but I'm back!")
            assistant.run()
            break
        except KeyboardInterrupt:
            log.info("\n\n👋 Shutting down...")
            break
        except Exception as e:
            log.error(f"Crash #{restart + 1}: {e}")
            if restart < MAX_CRASH_RESTARTS - 1:
                time.sleep(CRASH_COOLDOWN)
    else:
        log.error("Too many crashes, giving up")
