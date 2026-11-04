"""Shared model setup for the demos.

Every demo talks to a local Ollama server. Change MODEL to try a different model
(e.g. "ollama_chat/llama3.1:8b"), or point it at any provider LiteLLM supports.
"""

import dspy

MODEL = "ollama_chat/llama3.2"  # 3B; the 1B model often fails structured outputs
API_BASE = "http://localhost:11434/v1"  # dspy 3.x uses the OpenAI-compatible /v1 endpoint


def local_lm() -> dspy.LM:
    return dspy.LM(MODEL, api_base=API_BASE)  # Ollama is keyless; no api_key needed
