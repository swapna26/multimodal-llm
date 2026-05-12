# Multimodal LLM Pipeline

A real-time multimodal pipeline that combines voice, image, and text processing with latency budgets and graceful degradation. All models run locally — no API keys needed.

## Architecture

```
User speaks ──→ Whisper (STT) ──→ Text prompt ──┐
                                                  ├──→ Ollama LLaVA ──→ Text response ──→ Piper (TTS) ──→ Audio
User uploads image ──→ Base64 encode ────────────┘
```

### Components

| Component | Tool | What it does | Size |
|-----------|------|-------------|------|
| **STT** | OpenAI Whisper (base) | Audio → Text | ~150MB |
| **Vision** | Ollama LLaVA 7B | Image + Text → Text | ~4.7GB |
| **TTS** | Piper TTS | Text → Audio (WAV) | ~60MB |

## Endpoints

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/analyze` | POST | Image file + text prompt | Text analysis |
| `/transcribe` | POST | Audio file | Text transcription |
| `/speak` | POST | JSON text | WAV audio file |
| `/pipeline` | POST | Audio + Image + Prompt (any combo) | Text + Audio response |
| `/pipeline/latency` | GET | — | Latency budget config |
| `/health` | GET | — | Component health status |

## Quick Start

```bash
# Prerequisites: Ollama running, ffmpeg installed
brew install ffmpeg
ollama pull llava:7b

# Install dependencies
uv sync

# Start the API
uv run uvicorn src.app:app --port 8000

# Open Swagger docs
open http://localhost:8000/docs
```

## Usage Examples

```bash
# Analyze an image
curl -X POST http://localhost:8000/analyze \
  -F "image=@sample_images/shapes.png" \
  -F "prompt=What shapes do you see?"

# Transcribe audio
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@recording.wav"

# Text to speech
curl -X POST http://localhost:8000/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  -o output.wav

# Full pipeline (image + text prompt → text + audio)
curl -X POST http://localhost:8000/pipeline \
  -F "image=@photo.png" \
  -F "prompt=What is in this image?"
```

## Latency Budget

Each stage has a maximum time limit. If exceeded, a warning is logged but processing continues.

| Stage | Budget | Purpose |
|-------|--------|---------|
| STT | 10s | Transcribe audio to text |
| LLM | 30s | Analyze image + generate response |
| TTS | 15s | Convert response to speech |
| **Total** | **55s** | End-to-end maximum |

## Graceful Degradation

If one component fails, the system still works partially:

```
Normal:     Audio ──→ STT ──→ LLM ──→ TTS ──→ Audio response
STT fails:  Audio ──→ ✗      (falls back to text prompt if provided)
TTS fails:  Audio ──→ STT ──→ LLM ──→ ✗   (returns text response only)
LLM fails:  Returns error with timing breakdown
```

## Project Structure

```
multimodal-llm/
├── src/
│   ├── app.py         # FastAPI endpoints
│   ├── stt.py         # Speech-to-Text (Whisper)
│   ├── vision.py      # Image analysis (Ollama LLaVA)
│   ├── tts.py         # Text-to-Speech (Piper)
│   ├── pipeline.py    # Orchestrator with latency budgets
│   └── schemas.py     # Pydantic models
├── tests/
│   └── test_app.py    # 9 unit tests
├── models/            # Piper voice model (ONNX)
├── sample_images/     # Test images
└── pyproject.toml
```

## Test Results

```
Pipeline test with image:
  STT:  skipped (text prompt provided)
  LLM:  7.6s  within budget (30s)
  TTS:  1.6s  within budget (15s)
  Total: 9.2s

STT test (Whisper):
  Input:  1.66s audio ("What is the capital of France?")
  Output: "What is the capital of France?"
  Time:   0.6s  within budget (10s)

TTS test (Piper):
  Input:  "Hello, I am a multimodal assistant."
  Output: 2.43s audio
  Time:   instant
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Ollama with LLaVA model pulled
- ffmpeg (`brew install ffmpeg`) — Required by Whisper. Whisper doesn't read audio formats directly. It uses ffmpeg under the hood to decode audio files (MP3, WAV, M4A, etc.), resample them to 16kHz mono, and convert to raw PCM samples that the neural network can process. Without ffmpeg, audio transcription will fail.
- Python 3.12+
