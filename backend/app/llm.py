"""Pick the analyst model from whatever credentials are present.

ANTHROPIC_API_KEY -> Claude, OPENAI_API_KEY -> OpenAI, neither -> None, and
the pipeline falls back to the rule-based baseline analyst.
"""
import os

from . import config


def get_chat_model():
    if os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=config.ANTHROPIC_MODEL, temperature=0, max_tokens=700, timeout=60)
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.OPENAI_MODEL, temperature=0, max_tokens=700, timeout=60)
    return None


def describe(model):
    if model is None:
        return "rule-based baseline (no LLM key configured)"
    return f"{type(model).__name__}: {getattr(model, 'model', None) or getattr(model, 'model_name', '')}"
