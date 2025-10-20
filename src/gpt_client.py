"""Wrapper around the OpenAI API with graceful fallbacks."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Iterable

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class GPTClient:
    """Small helper around the OpenAI Chat Completions API."""

    model: str = "gpt-4o"
    temperature: float = 0.5
    _client: "OpenAI | None" = field(init=False, default=None, repr=False)

    def _requires_openai(self) -> None:
        if OpenAI is None:
            raise RuntimeError(
                "The openai package is not installed. Install it or provide a custom client."
            )
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    def _client_instance(self) -> "OpenAI":
        self._requires_openai()
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def chat_completion(self, system_prompt: str, messages: Iterable[dict[str, str]]) -> str:
        """Execute a chat completion call and return the assistant content."""

        client = self._client_instance()
        completion = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system_prompt}, *messages],
        )
        choice = completion.choices[0].message.content
        LOGGER.debug("Received completion: %s", choice)
        return str(choice)

    def offline_stub(self, prompt: str) -> str:
        """Fallback response when OpenAI API is unavailable."""

        LOGGER.warning("Using offline stub for prompt: %.120s", prompt)
        return json.dumps({"message": "OpenAI API unavailable", "prompt": prompt[:400]})

    def summarize_articles(self, articles: Iterable[dict], background: str = "") -> str:
        """Ask GPT to propose research directions from the supplied articles."""

        system_prompt = (
            "You are a biomedical researcher specializing in breast cancer. "
            "Synthesize insights from the provided PubMed summaries to propose a translational research direction."
        )
        messages = [
            {
                "role": "user",
                "content": (
                    "Background context:\n"
                    f"{background}\n\n"
                    "Articles:\n"
                    f"{json.dumps(list(articles), ensure_ascii=False, indent=2)}"
                ),
            }
        ]
        try:
            return self.chat_completion(system_prompt, messages)
        except Exception as exc:  # pragma: no cover - network fallback
            LOGGER.error("Falling back to offline stub: %s", exc)
            return self.offline_stub(json.dumps(messages[-1]))

    def draft_proposal(self, outline: str, word_count: int) -> str:
        """Ask GPT to write a structured project proposal."""

        system_prompt = (
            "You are drafting a Chinese research project proposal on translational breast cancer studies. "
            "Produce formal, well-structured language with numbered sections and sub-sections."
        )
        messages = [
            {
                "role": "user",
                "content": (
                    "Using the following outline and requirements, write a project plan in Chinese of approximately "
                    f"{word_count} words. Ensure the sections cover: research background, objectives, detailed design, "
                    "feasibility analysis, expected outcomes, milestones, and references.\n\n"
                    f"Outline:\n{outline}"
                ),
            }
        ]
        try:
            return self.chat_completion(system_prompt, messages)
        except Exception as exc:  # pragma: no cover - network fallback
            LOGGER.error("Falling back to offline stub for proposal: %s", exc)
            return self.offline_stub(json.dumps(messages[-1]))
