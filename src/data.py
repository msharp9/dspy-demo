"""Dataset loaders shared by the demos.

dspy ships `dspy.datasets.GSM8K`, but it loads the dataset under its old Hub name ("gsm8k"),
which current versions of `datasets` reject. These loaders use the current names.
"""

import random

import dspy
from datasets import load_dataset


def load_gsm8k(split="train", n=200, seed=0):
    """Grade-school math word problems as dspy.Examples with `question` -> `answer`."""
    rows = list(load_dataset("openai/gsm8k", "main", split=split))
    random.Random(seed).shuffle(rows)
    examples = []
    for row in rows[:n]:
        reasoning, _, answer = row["answer"].rpartition("####")
        examples.append(
            dspy.Example(
                question=row["question"],
                gold_reasoning=reasoning.strip(),
                answer=answer.strip().replace(",", ""),
            ).with_inputs("question")
        )
    return examples


def load_banking77(split="train", n=200, seed=0):
    """Customer-support messages labeled with one of 77 banking intents.

    Returns (examples, class_names).
    """
    dataset = load_dataset("PolyAI/banking77", split=split)
    classes = dataset.features["label"].names
    rows = list(dataset)
    random.Random(seed).shuffle(rows)
    examples = [dspy.Example(text=row["text"], label=classes[row["label"]]).with_inputs("text") for row in rows[:n]]
    return examples, classes
