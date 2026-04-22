"""Baseline vs trajectory-conditioned replies. Uses OpenAI only if OPENAI_API_KEY is set."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional, Tuple

from .prompts import BASELINE_SYSTEM_PROMPT, BASELINE_TEMPLATE, CONDITIONED_SYSTEM_PREFIX


def _template_conditioned(last_user_text: str, traj: dict[str, Any]) -> str:
    summ = traj.get("summary", "")
    vol = traj.get("volatility", traj.get("volatility_index", 0.0))
    esc = traj.get("escalation_score", 0.0)
    lines = [
        "(Conditioned template — set OPENAI_API_KEY for LLM replies.)",
        "",
        "I hear you. Here is what I'm noticing about how things have felt across your messages:",
        summ,
        "",
        "If it's okay, let's take this one step at a time. What's the smallest next step that would feel manageable right now?",
        f"(Diagnostics: volatility≈{vol:.2f}, escalation≈{esc:+.2f})",
    ]
    return "\n".join(lines)


def _openai_chat(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = "gpt-4o-mini",
    max_tokens: int = 350,
) -> Optional[str]:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, IndexError, TimeoutError):
        return None


def build_prompt(last_user_text: str, traj: dict[str, Any]) -> str:
    """Human-content for the conditioned call (the system prompt carries the emotion context)."""
    return (
        "User's latest message:\n"
        f"{last_user_text}\n\n"
        "Emotional trajectory summary for this conversation:\n"
        f"{traj.get('summary','')}\n\n"
        "Write a short, supportive reply (4-8 sentences). "
        "Match the emotional intensity without being dramatic. "
        "Include validation + one grounding suggestion."
    )


def generate_baseline_reply(last_user_text: str, *, model: str = "gpt-4o-mini") -> str:
    """Generic non-emotion-aware reply. Used as the A/B baseline."""
    llm = _openai_chat(BASELINE_SYSTEM_PROMPT, last_user_text, model=model)
    return llm if llm else BASELINE_TEMPLATE


def generate_reply(
    last_user_text: str,
    traj: dict[str, Any],
    *,
    conditioning_prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> str:
    """Trajectory-conditioned reply. Pass `conditioning_prompt` from module2.build_conditioning_prompt()."""
    user_prompt = build_prompt(last_user_text, traj)
    if conditioning_prompt:
        system_prompt = CONDITIONED_SYSTEM_PREFIX.format(conditioning_prompt=conditioning_prompt)
    else:
        system_prompt = (
            "You are an empathetic assistant. Acknowledge emotions briefly, "
            "avoid minimizing, offer grounding and one concrete small step."
        )
    llm = _openai_chat(system_prompt, user_prompt, model=model)
    if llm:
        return llm
    return _template_conditioned(last_user_text, traj)


def generate_ab_pair(
    last_user_text: str,
    traj: dict[str, Any],
    *,
    conditioning_prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> Tuple[str, str]:
    """
    Produce both responses for a single user turn — the core A/B design.

    Returns
    -------
    (baseline, conditioned)
    """
    baseline = generate_baseline_reply(last_user_text, model=model)
    conditioned = generate_reply(
        last_user_text,
        traj,
        conditioning_prompt=conditioning_prompt,
        model=model,
    )
    return baseline, conditioned
