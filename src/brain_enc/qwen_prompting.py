"""Shared prompt helpers for Qwen multimodal extraction."""

from __future__ import annotations

import hashlib
import typing as tp

QwenPromptMode = tp.Literal["manual", "chat_template"]

DEFAULT_QWEN_PROMPT_MODE: QwenPromptMode = "manual"


def normalize_system_prompt(prompt: str | None) -> str | None:
    """Collapse blank prompt strings to ``None`` for stable cache identity."""
    if prompt is None:
        return None
    normalized = str(prompt).strip()
    return normalized or None


def system_prompt_hash(prompt: str | None) -> str:
    """Return a short stable identifier for one normalized system prompt."""
    normalized = normalize_system_prompt(prompt)
    if normalized is None:
        return "none"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return digest[:12]


def cache_prompt_id(
    *,
    prompt_mode: QwenPromptMode,
    system_prompt: str | None,
) -> str | None:
    """Return the cache-path identity suffix for one prompt configuration."""
    normalized_prompt = normalize_system_prompt(system_prompt)
    if prompt_mode == "manual":
        if normalized_prompt is not None:
            raise ValueError("system_prompt requires prompt_mode='chat_template'")
        return None
    return f"chattmpl-{system_prompt_hash(normalized_prompt)}"


def prompt_metadata(
    *,
    prompt_mode: QwenPromptMode,
    system_prompt: str | None,
) -> dict[str, tp.Any]:
    """Return stable metadata fields describing prompt construction."""
    normalized_prompt = normalize_system_prompt(system_prompt)
    prompt_id = cache_prompt_id(
        prompt_mode=prompt_mode,
        system_prompt=normalized_prompt,
    )
    return {
        "prompt_mode": prompt_mode,
        "prompted": prompt_mode != "manual",
        "chat_template_applied": prompt_mode == "chat_template",
        "system_prompt": normalized_prompt or "",
        "system_prompt_id": system_prompt_hash(normalized_prompt) if prompt_mode == "chat_template" else "",
        "prompt_id": prompt_id or "",
    }
