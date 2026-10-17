"""
One chat call for every VLM request in the pipeline (detection, captions,
descriptions, the video-summary combine), on either backend
(config.VLM_BACKEND):

  llama   llama-server's OpenAI-compatible /v1/chat/completions. Images go
          as base64 data URIs; thinking is disabled through the chat
          template (enable_thinking=False) — MiniCPM-V 4.x otherwise spends
          the token budget on hidden reasoning.
  ollama  ollama.chat with think=False for MiniCPM-V 4.x
          (config.vlm_extra_kwargs) and keep_alive so the model stays warm
          between files.

Every call caps generation (num_predict): an uncapped description once
looped the same sentence 40+ times.
"""
import base64
import json
import urllib.error
import urllib.request
from pathlib import Path

import config

OLLAMA_KEEP_ALIVE = "30m"  # covers any per-file gap within a batch
LLAMA_TIMEOUT = 900  # seconds; a long-form description on a cold model


class VLMError(RuntimeError):
    """The backend call itself failed (connection refused, HTTP error,
    timeout) — distinct from an empty answer."""


def image_b64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def chat(messages: list[dict], *, model: str | None = None,
         temperature: float = 0.2, num_predict: int = 512) -> str:
    """messages: [{"role", "content", "images": [b64, ...]?}] (Ollama shape).
    Returns the assistant text. Raises VLMError on transport failure."""
    model = model or config.VISION_MODEL
    if config.VLM_BACKEND == "llama":
        return _chat_llama(messages, temperature, num_predict)
    return _chat_ollama(messages, model, temperature, num_predict)


def _chat_llama(messages, temperature, num_predict) -> str:
    out = []
    for m in messages:
        if m.get("images"):
            content = [{"type": "text", "text": m["content"]}] + [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}}
                for b in m["images"]]
        else:
            content = m["content"]
        out.append({"role": m["role"], "content": content})
    body = {"messages": out, "temperature": temperature, "max_tokens": num_predict,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        config.LLAMA_HOST.rstrip("/") + "/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=LLAMA_TIMEOUT) as r:
            data = json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise VLMError(f"llama-server call failed: {e}") from e
    try:
        return data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError) as e:
        raise VLMError(f"unexpected llama-server response: {str(data)[:200]}") from e


def _chat_ollama(messages, model, temperature, num_predict) -> str:
    import ollama
    try:
        response = ollama.chat(
            model=model, messages=messages,
            options={"temperature": temperature, "num_predict": num_predict},
            keep_alive=OLLAMA_KEEP_ALIVE, **config.vlm_extra_kwargs(model))
    except Exception as e:
        raise VLMError(f"Ollama call failed: {e}") from e
    return response.message.content or ""
