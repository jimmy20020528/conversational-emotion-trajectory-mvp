"""
Module 3 — System prompts for baseline vs trajectory-conditioned replies.

The A/B contrast is the whole experimental design: same LLM, same user turn,
only the system prompt changes. Keeping the two strings in one module makes
the comparison explicit and reviewable.
"""

from __future__ import annotations

BASELINE_SYSTEM_PROMPT = (
    "You are a helpful conversational assistant. "
    "Respond naturally to what the user says."
)

# {conditioning_prompt} is substituted at call time with the output of
# module2.build_conditioning_prompt(signals).
CONDITIONED_SYSTEM_PREFIX = (
    "You are an emotionally intelligent conversational assistant. "
    "Use the emotional context below to calibrate your response.\n\n"
    "{conditioning_prompt}\n\n"
    "Respond with appropriate empathy, tone, and support."
)

# Template reply used when OPENAI_API_KEY is not set, to keep the A/B
# comparison meaningful even offline. The baseline template is deliberately
# generic; the conditioned template echoes the trajectory summary so the
# difference is visible.
BASELINE_TEMPLATE = (
    "(Baseline template — set OPENAI_API_KEY for a real LLM reply.)\n"
    "\n"
    "Thanks for sharing. Could you tell me a bit more about what you're hoping to do next?"
)
