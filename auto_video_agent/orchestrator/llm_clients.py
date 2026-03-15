from __future__ import annotations

import base64
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _read_image_as_data_url(image_path: str | os.PathLike[str]) -> str:
    path = Path(image_path)
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    ext = path.suffix.lower().lstrip(".")
    if ext in {"jpg", "jpeg"}:
        mime = "image/jpeg"
    elif ext == "png":
        mime = "image/png"
    elif ext == "webp":
        mime = "image/webp"
    else:
        mime = "application/octet-stream"
    return f"data:{mime};base64,{b64}"


def _extract_response_text(resp: Any) -> str:
    if resp is None:
        raise ValueError("Empty response from OpenAI")

    if hasattr(resp, "output_text") and isinstance(getattr(resp, "output_text"), str):
        text = resp.output_text.strip()
        if text:
            return text

    if hasattr(resp, "output") and isinstance(resp.output, list):
        chunks: list[str] = []
        for item in resp.output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for c in content:
                t = getattr(c, "text", None)
                if isinstance(t, str) and t.strip():
                    chunks.append(t.strip())
        if chunks:
            return "\n".join(chunks)

    if isinstance(resp, dict):
        for key in ("output_text", "text", "content"):
            v = resp.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        raise ValueError(f"Unsupported dict response shape: keys={list(resp.keys())}")

    raise ValueError(f"Unsupported response type: {type(resp)}")


def _loads_json_strict(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Empty JSON text")
    return json.loads(text)


def _loads_json_lenient(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Empty JSON text")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str | None = None
    base_url: str | None = None
    model: str = "gpt-4o-mini"
    max_output_tokens: int = 1200
    request_timeout_sec: float = 120.0
    max_retries: int = 4
    min_retry_sleep_sec: float = 0.8
    max_retry_sleep_sec: float = 6.0


class OpenAIClient:
    def __init__(self, config: OpenAIConfig | None = None):
        self.config = config or OpenAIConfig()
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY (set env var or pass OpenAIConfig.api_key)")

        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai package not installed. Please install openai>=1.0.0.") from e

        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = OpenAI(**kwargs)

    def responses_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: Iterable[str | os.PathLike[str]] = (),
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "result",
        strict_schema: bool = True,
    ) -> Any:
        image_items = [
            {
                "type": "input_image",
                "image_url": _read_image_as_data_url(p),
            }
            for p in images
        ]

        content: list[dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
        content.extend(image_items)

        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_output_tokens": int(self.config.max_output_tokens),
        }

        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema,
                    "strict": bool(strict_schema),
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}

        last_err: Exception | None = None
        for attempt in range(1, int(self.config.max_retries) + 1):
            try:
                resp = self._client.responses.create(**payload)
                text = _extract_response_text(resp)
                try:
                    return _loads_json_strict(text)
                except Exception:
                    return _loads_json_lenient(text)
            except Exception as e:
                last_err = e
                if attempt >= int(self.config.max_retries):
                    break
                sleep_s = _retry_sleep(attempt, self.config.min_retry_sleep_sec, self.config.max_retry_sleep_sec)
                time.sleep(sleep_s)

        raise RuntimeError("OpenAI request failed") from last_err


def _retry_sleep(attempt: int, min_s: float, max_s: float) -> float:
    base = min_s * (2 ** max(0, attempt - 1))
    base = min(base, max_s)
    jitter = random.random() * 0.25 * base
    return float(min(max_s, base + jitter))

