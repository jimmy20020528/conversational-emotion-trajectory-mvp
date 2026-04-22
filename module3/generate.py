"""Baseline vs trajectory-conditioned replies. Uses OpenAI only if OPENAI_API_KEY is set."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def _template_reply(last_user_text: str, traj: dict[str, Any]) -> str:
    summ = traj.get("summary", "")
    vol = traj.get("volatility", 0.0)
    esc = traj.get("escalation_score", 0.0)
    lines = [
        "(Template fallback — set OPENAI_API_KEY for LLM replies.)",
        "",
        "I hear you. Here is what I'm noticing about how things have felt across your messages:",
        summ,
        "",
        "If it's okay, let's take this one step at a time. What's the smallest next step that would feel manageable right now?",
        f"(Diagnostics: volatility≈{vol:.2f}, escalation≈{esc:+.2f})",
    ]
    return "\n".join(lines)


def _openai_chat(prompt: str, *, model: str = "gpt-4o-mini") -> str | None:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an empathetic assistant. Acknowledge emotions briefly, "
                    "avoid minimizing, offer grounding and one concrete small step."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 350,
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
    return (
        "User's latest message:\n"
        f"{last_user_text}\n\n"
        "Emotional trajectory summary for this conversation:\n"
        f"{traj.get('summary','')}\n\n"
        "Write a short, supportive reply (4-8 sentences). "
        "Match the emotional intensity without being dramatic. "
        "Include validation + one grounding suggestion."
    )


def generate_reply(last_user_text: str, traj: dict[str, Any]) -> str:
    prompt = build_prompt(last_user_text, traj)
    llm = _openai_chat(prompt)
    if llm:
        return llm
    return _template_reply(last_user_text, traj)
