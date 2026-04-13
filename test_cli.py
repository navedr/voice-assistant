#!/usr/bin/env python3
"""
CLI test harness — uses the real VoiceAssistant but replaces audio with text I/O.

Usage: venv/bin/python test_cli.py
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Disable audio before importing the assistant
os.environ.setdefault("AUDIO_DEVICE", "")

from voice_assistant_simple import VoiceAssistant, ASSISTANT_NAME, WAKE_WORD, log


class CLIAssistant(VoiceAssistant):
    """Subclass that replaces audio I/O with terminal I/O."""

    def _startup_self_test(self):
        log.info(f"✅ {ASSISTANT_NAME} CLI mode — no audio")

    def _open_mic(self):
        raise RuntimeError("No mic in CLI mode")

    def play_beep(self):
        print("  *beep*")

    def speak(self, text):
        log.info(f"🔊 {ASSISTANT_NAME}: {text}")

    def flush_mic(self, duration=0.5):
        pass

    def _play_wav(self, path):
        pass

    def _ensure_audio_cues(self):
        pass

    def detect_wake_word(self):
        """Replace audio wake word detection with text input."""
        try:
            text = input(f"\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise KeyboardInterrupt

        if not text:
            return False

        # Check if it's a wake word + command
        result = self.is_wake_word(text.lower())
        if result:
            return result

        # No wake word — treat as direct command (skip wake word in CLI)
        return text

    def record_command(self, max_seconds=7):
        """Replace audio recording with text input."""
        try:
            text = input(f"You: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise KeyboardInterrupt
        return text if text else None


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set")
        exit(1)

    assistant = CLIAssistant()
    assistant.run()
