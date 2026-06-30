"""LLM client for BIMClaw.

Defaults to the Anthropic API (claude-sonnet-4-6) via the ``anthropic`` SDK.
Falls back to an OpenAI-compatible endpoint (e.g. vLLM) when
``BIMAgentConfig.llm_base_url`` is set to a non-empty value.

API is intentionally simple — no load-balancing needed for single-LLM use.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


def _image_to_base64(img: Image.Image, max_edge: int = 1024) -> str:
    """Encode PIL Image as base64 JPEG string (no prefix)."""
    w, h = img.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def image_to_base64_url(img: Image.Image, max_edge: int = 1024) -> str:
    """Encode PIL Image as a base64 JPEG data URI (with data: prefix)."""
    return f"data:image/jpeg;base64,{_image_to_base64(img, max_edge)}"


class LLMResponse:
    """Unified response object returned by all backends."""

    def __init__(self, content: str, usage: Optional[Dict] = None):
        self.content = content
        self.usage = usage or {}

    def __repr__(self) -> str:
        return f"LLMResponse({len(self.content)} chars)"


class LLMClient:
    """LLM client that supports Anthropic API and OpenAI-compatible endpoints.

    Parameters
    ----------
    model:
        Model ID, e.g. "claude-sonnet-4-6" or "Qwen/Qwen2.5-VL-72B".
    api_key:
        API key for the chosen backend.
    base_url:
        If set, use OpenAI-compatible endpoint at this URL.
        If empty/None, use Anthropic API.
    max_tokens:
        Default max tokens for generation.
    temperature:
        Default sampling temperature.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str = "",
        base_url: str = "",
        max_tokens: int = 16384,
        temperature: float = 0.6,
    ):
        self.model = model
        self.api_key = api_key or ""
        self.base_url = base_url or ""
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._use_anthropic = not bool(self.base_url)

        if self._use_anthropic:
            self._init_anthropic()
        else:
            self._init_openai()

    def _init_anthropic(self) -> None:
        try:
            import anthropic
            key = self.api_key or _env("ANTHROPIC_API_KEY", "")
            self._client = anthropic.AsyncAnthropic(api_key=key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

    def _init_openai(self) -> None:
        try:
            from openai import AsyncOpenAI
            key = self.api_key or _env("OPENAI_API_KEY", "bearer")
            self._client = AsyncOpenAI(
                api_key=key,
                base_url=self.base_url,
            )
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        role_params=None,
        session_id: str = "",
        usage_session_id: str = "",
        **kwargs,
    ):
        """SpatialClaw-compatible generate() that returns (text, reasoning) tuple."""
        response = await self.agenerate(messages, params=role_params)
        return response.content, ""

    async def agenerate(
        self,
        messages: List[Dict[str, Any]],
        params=None,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Parameters
        ----------
        messages:
            List of dicts with "role" and "content" keys.
            Content can be a string or a list of content blocks for multimodal.
        params:
            Optional LLMRoleParams / dict with max_tokens, temperature, etc.
            When given, overrides instance defaults.
        """
        max_tokens = self.max_tokens
        temperature = self.temperature
        if params is not None:
            if hasattr(params, "max_tokens"):
                max_tokens = params.max_tokens
                temperature = params.temperature
            elif isinstance(params, dict):
                max_tokens = params.get("max_tokens", max_tokens)
                temperature = params.get("temperature", temperature)

        if self._use_anthropic:
            return await self._anthropic_generate(messages, max_tokens, temperature)
        else:
            return await self._openai_generate(messages, max_tokens, temperature)

    async def _anthropic_generate(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        # Separate system prompt from conversation messages
        system_content = ""
        conv_messages = []
        for msg in messages:
            if msg["role"] == "system":
                if isinstance(msg["content"], str):
                    system_content = msg["content"]
                continue
            conv_messages.append(_convert_message_anthropic(msg))

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conv_messages,
        }
        if system_content:
            kwargs["system"] = system_content

        response = await self._client.messages.create(**kwargs)
        content = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return LLMResponse(content=content, usage=usage)

    async def _openai_generate(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        conv = [_convert_message_openai(m) for m in messages]
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=conv,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        return LLMResponse(content=content, usage=usage)


def _convert_message_anthropic(msg: Dict) -> Dict:
    """Convert a message dict to Anthropic API format."""
    content = msg["content"]
    if isinstance(content, str):
        return {"role": msg["role"], "content": content}
    # List of content blocks — convert image_url blocks to Anthropic format
    blocks = []
    for block in content:
        if block.get("type") == "text":
            blocks.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "image_url":
            url = block["image_url"]["url"]
            if url.startswith("data:"):
                media_type, b64data = url.split(";base64,", 1)
                media_type = media_type.replace("data:", "")
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64data,
                    },
                })
    return {"role": msg["role"], "content": blocks}


def _convert_message_openai(msg: Dict) -> Dict:
    """Convert a message dict to OpenAI chat format (already compatible)."""
    return msg


def _env(key: str, default: str = "") -> str:
    import os
    return os.environ.get(key, default)
