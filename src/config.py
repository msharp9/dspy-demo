"""Shared model setup for the demos.

Every demo talks to a local Ollama server. Change MODEL to try a different model
(e.g. "ollama_chat/llama3.1:8b"), or point it at any provider LiteLLM supports.
"""

import dspy

MODEL = "ollama_chat/llama3.2"  # 3B; the 1B model often fails structured outputs
API_BASE = "http://localhost:11434/v1"  # dspy 3.x uses the OpenAI-compatible /v1 endpoint


# GEPA's "reflection" model reads the student's mistakes and rewrites its prompt, so it should be
# stronger than MODEL. Larger local options: "ollama_chat/qwen3.5:9b", "ollama_chat/gpt-oss".
REFLECTION_MODEL = "ollama_chat/llama3.1:8b"


def local_lm() -> dspy.LM:
    return dspy.LM(MODEL, api_base=API_BASE)  # Ollama is keyless; no api_key needed


def reflection_lm() -> dspy.LM:
    return dspy.LM(REFLECTION_MODEL, api_base=API_BASE, temperature=1.0, max_tokens=8000)
