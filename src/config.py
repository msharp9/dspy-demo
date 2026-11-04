"""Shared model setup for the demos.

Every demo talks to a local Ollama server. Change MODEL to try a different model
(e.g. "ollama_chat/llama3.1:8b"), or point it at any provider LiteLLM supports.
"""

import dspy

MODEL = "ollama_chat/llama3.2:1b"
API_BASE = "http://localhost:11434"


def local_lm() -> dspy.LM:
    return dspy.LM(MODEL, api_base=API_BASE, api_key="")
