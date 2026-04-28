"""
Text-to-Speech (TTS) Module
=============================
Uses Piper TTS to convert text into spoken audio.

Piper runs locally — no API key needed. Very lightweight (~50MB model).
Produces WAV audio files from text input.

How it works:
  1. Load the Piper voice model (ONNX format, ~60MB)
  2. Pass text string to the model
  3. Model generates raw audio samples (PCM waveform)
  4. Save as WAV file

We downloaded the voice model earlier to:
  models/en_US-lessac-medium.onnx      (the neural network)
  models/en_US-lessac-medium.onnx.json (voice config: sample rate, phonemes)

Usage:
    from src.tts import synthesize_speech
    result = synthesize_speech("Hello world", "output.wav")
    # result = {"output_path": "output.wav", "duration_sec": 1.2, ...}
"""

import io
import logging
import time
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
DEFAULT_VOICE = "en_US-lessac-medium"

# Lazy-loaded Piper voice
_voice = None


def _get_voice(voice_name: str = DEFAULT_VOICE):
    """Load Piper voice model (cached after first call)."""
    global _voice
    if _voice is None:
        from piper import PiperVoice

        model_path = MODELS_DIR / f"{voice_name}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Piper voice model not found: {model_path}\n"
                f"Download it from: https://huggingface.co/rhasspy/piper-voices"
            )

        logger.info(f"Loading Piper voice '{voice_name}'...")
        _voice = PiperVoice.load(str(model_path))
        logger.info("Piper voice loaded.")
    return _voice


def synthesize_speech(
    text: str,
    output_path: str = "output.wav",
) -> dict:
    """
    Convert text to speech and save as WAV file.

    Args:
        text: The text to speak
        output_path: Where to save the WAV file

    Returns:
        dict with keys: output_path, text_length, duration_sec, synthesis_time_sec
    """
    voice = _get_voice()

    start = time.perf_counter()

    # synthesize_wav sets up WAV headers (channels, sample rate) automatically
    with wave.open(output_path, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    elapsed = time.perf_counter() - start

    # Calculate audio duration from the WAV file
    with wave.open(output_path, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / rate

    return {
        "output_path": output_path,
        "text_length": len(text),
        "duration_sec": round(duration, 2),
        "synthesis_time_sec": round(elapsed, 4),
    }


def synthesize_to_bytes(text: str) -> tuple[bytes, dict]:
    """
    Convert text to speech and return raw WAV bytes (for streaming via API).

    Returns:
        tuple of (wav_bytes, metadata_dict)
    """
    voice = _get_voice()

    start = time.perf_counter()

    # Write WAV to in-memory buffer instead of file
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    elapsed = time.perf_counter() - start
    wav_bytes = buffer.getvalue()

    return wav_bytes, {
        "text_length": len(text),
        "audio_size_bytes": len(wav_bytes),
        "synthesis_time_sec": round(elapsed, 4),
    }
