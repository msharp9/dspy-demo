import dspy
import pytest
from dspy.utils import DummyLM

# One canned response that satisfies every signature in the demos. The adapter ignores
# fields a signature doesn't ask for, so a single response works everywhere.
CANNED = {
    "reasoning": "Thinking it through.",
    "answer": "5",
    "sentiment": "positive",
    "confidence": "0.9",
    "title": "World Cup 2002",
    "sections": '["Overview", "Final"]',
    "section_subheadings": '{"Overview": ["Hosts"], "Final": ["Result"]}',
    "content": "## Section\n\nSome text.",
    "label": "card_arrival",
    # ReAct: finish immediately
    "next_thought": "I can answer now.",
    "next_tool_name": "finish",
    "next_tool_args": "{}",
}


@pytest.fixture
def fake_lm():
    return DummyLM([CANNED] * 500)


@pytest.fixture
def gsm8k_trainset():
    return [
        dspy.Example(question=f"What is {i} + {i}?", gold_reasoning="add", answer=str(2 * i)).with_inputs("question")
        for i in range(1, 6)
    ]
