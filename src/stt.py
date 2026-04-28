"""
Speech-to-Text (STT) Module
============================
Uses OpenAI Whisper to transcribe audio files to text.

Whisper runs locally — no API key needed.
We use the "base" model (~150MB) for a good speed/accuracy trade-off on M4 Mac.

Model sizes available:
  tiny  (~75MB)  — fastest, lower accuracy
  base  (~150MB) — good balance (our default)
  small (~500MB) — better accuracy, slower
  medium (~1.5GB) — great accuracy, slower
  large  (~3GB)  — best accuracy, slowest

The model downloads automatically on first use to ~/.cache/whisper/

Usage:
    from src.stt import transcribe_audio
    result = transcribe_audio("path/to/audio.wav")
    # result = {"text": "Hello world", "language": "en", ...}
"""

import logging
import time
from pathlib import Path

import whisper

logger = logging.getLogger(__name__)

# Load model once (lazy loading on first use)
_model = None


def _get_model(model_size: str = "base"):
    """
    Load Whisper model (cached after first call).

    On first call: downloads model from OpenAI servers to ~/.cache/whisper/
    On subsequent calls: returns the already-loaded model from memory.
    """
    global _model
    if _model is None:
        logger.info(f"Loading Whisper '{model_size}' model...")
        _model = whisper.load_model(model_size)
        logger.info("Whisper model loaded.")
    return _model


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
) -> dict:
    """
    Transcribe an audio file to text using Whisper.

    Args:
        audio_path: Path to audio file (wav, mp3, ogg, etc.)
        model_size: Whisper model size — tiny, base, small, medium, large

    Returns:
        dict with keys: text, language, audio_duration_sec, transcription_time_sec
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _get_model(model_size)

    # fp16=False because Apple M4 runs on CPU, not NVIDIA GPU
    # fp16 (16-bit float) only helps on NVIDIA GPUs, causes errors on CPU
    start = time.perf_counter()
    result = model.transcribe(str(path), fp16=False)
    elapsed = time.perf_counter() - start

    # Get audio duration from the raw audio data
    audio = whisper.load_audio(str(path))
    audio_duration = len(audio) / whisper.audio.SAMPLE_RATE

    return {
        "text": result["text"].strip(),
        "language": result.get("language", "unknown"),
        "audio_duration_sec": round(audio_duration, 2),
        "transcription_time_sec": round(elapsed, 4),
    }
