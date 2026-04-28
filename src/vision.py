"""
Vision Module
==============
Sends images to Ollama's LLaVA model for analysis.

LLaVA (Large Language and Vision Assistant) is a multimodal model that
understands images + text and generates text responses.

How it works:
  1. Read image file from disk
  2. Convert to base64 string (how images are sent over HTTP)
  3. Send to Ollama's /api/generate with the base64 image
  4. Stream the response back word by word

Base64 encoding: Images are binary data (pixels). HTTP/JSON only supports text.
Base64 converts binary → text so we can include the image inside a JSON body.
Example: a 100KB image becomes ~133KB of base64 text.

Usage:
    from src.vision import analyze_image
    result = analyze_image("photo.png", "What do you see?")
    # result = {"response": "I see a cat...", "model": "llava:7b", ...}
"""

import base64
import json
import logging
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llava:7b"


def _encode_image(image_path: str) -> str:
    """Read an image file and convert to base64 string."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image(
    image_path: str,
    prompt: str = "Describe this image in detail.",
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Send an image to LLaVA via Ollama and get a text description.

    Non-streaming — waits for the full response.

    Args:
        image_path: Path to image file (png, jpg, etc.)
        prompt: Question to ask about the image
        model: Ollama model name (must be multimodal like llava)

    Returns:
        dict with keys: response, model, total_time_sec
    """
    image_b64 = _encode_image(image_path)

    start = time.perf_counter()
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
            },
        )
        data = resp.json()

    elapsed = time.perf_counter() - start

    return {
        "response": data.get("response", ""),
        "model": model,
        "total_time_sec": round(elapsed, 4),
    }


async def analyze_image_stream(
    image_path: str,
    prompt: str = "Describe this image in detail.",
    model: str = DEFAULT_MODEL,
):
    """
    Send an image to LLaVA via Ollama and stream the response.

    Yields chunks of text as the model generates them.
    Also yields a final summary with timing metrics.

    Args:
        image_path: Path to image file
        prompt: Question to ask about the image
        model: Ollama model name
    """
    image_b64 = _encode_image(image_path)

    start = time.perf_counter()
    ttft = None
    token_count = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": True,
            },
        ) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue

                chunk = json.loads(line)

                if "response" in chunk and chunk["response"]:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    token_count += 1
                    yield {"type": "token", "text": chunk["response"]}

                if chunk.get("done", False):
                    break

    total_time = time.perf_counter() - start
    gen_time = total_time - (ttft or 0)

    yield {
        "type": "done",
        "ttft_sec": round(ttft or 0, 4),
        "total_time_sec": round(total_time, 4),
        "tokens": token_count,
        "tokens_per_sec": round(token_count / gen_time, 2) if gen_time > 0 else 0,
    }
