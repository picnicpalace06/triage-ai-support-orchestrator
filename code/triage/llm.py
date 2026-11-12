"""LLM client wrapper.

Supports two providers selected by environment variables:

- ``ANTHROPIC_API_KEY``  -> uses Claude (default model: claude-3-5-sonnet-latest)
- ``OPENAI_API_KEY``     -> uses OpenAI Chat Completions (default: gpt-4o-mini)

If neither key is set (or the SDK is missing), :class:`LLMClient.complete`
returns ``None`` and the agent falls back to an extractive answer built
from retrieved passages.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str


class LLMClient:
    def __init__(self) -> None:
        self.provider: str | None = None
        self.model: str | None = None
        self._client = None

        anth_key = os.getenv("ANTHROPIC_API_KEY")
        oai_key = os.getenv("OPENAI_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        if anth_key:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=anth_key)
                self.provider = "anthropic"
                self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
                logger.info("LLM provider: anthropic (%s)", self.model)
                return
            except Exception as exc:  # pragma: no cover - SDK optional
                logger.warning("Anthropic init failed: %s", exc)

        if groq_key:
            try:
                from openai import OpenAI

                base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
                self._client = OpenAI(api_key=groq_key, base_url=base_url)
                self.provider = "groq"
                self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
                logger.info("LLM provider: groq (%s)", self.model)
                return
            except Exception as exc:  # pragma: no cover
                logger.warning("Groq init failed: %s", exc)

        if oai_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=oai_key)
                self.provider = "openai"
                self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                logger.info("LLM provider: openai (%s)", self.model)
                return
            except Exception as exc:  # pragma: no cover
                logger.warning("OpenAI init failed: %s", exc)

        logger.info("No LLM API key found; running in extractive-only mode.")

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 400,
        temperature: float = 0.1,
    ) -> LLMResponse | None:
        if not self.available:
            return None

        try:
            if self.provider == "anthropic":
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                parts = []
                for block in msg.content:
                    if getattr(block, "type", None) == "text":
                        parts.append(block.text)
                text = "\n".join(parts).strip()
                return LLMResponse(text=text, provider=self.provider, model=self.model)

            if self.provider in ("openai", "groq"):
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                text = resp.choices[0].message.content or ""
                return LLMResponse(text=text.strip(), provider=self.provider, model=self.model)
        except Exception as exc:
            logger.warning("LLM call failed (%s): %s", self.provider, exc)
            return None

        return None
