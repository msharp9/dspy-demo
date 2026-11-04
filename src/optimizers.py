"""Optimize an intent classifier with MIPROv2 and measure accuracy before and after.

MIPROv2 searches over instructions and few-shot examples, scoring each candidate on your
metric. Always evaluate on data the optimizer never saw (here, the Banking77 test split).
"""

from typing import Literal

import dspy

from config import local_lm
from data import load_banking77


def accuracy(example, prediction, trace=None):
    return example.label == prediction.label


def build_classifier(classes):
    signature = dspy.Signature("text -> label", "Classify the banking intent of a customer message.")
    # Restrict `label` to the 77 known intents so the model can't invent new ones.
    signature = signature.with_updated_fields("label", type_=Literal[tuple(classes)])
    return dspy.ChainOfThought(signature)


def main(lm=None, trainset=None, devset=None, classes=None, num_threads=4):
    dspy.configure(lm=lm or local_lm())
    if trainset is None:
        trainset, classes = load_banking77("train", n=100)
    if devset is None:
        devset, _ = load_banking77("test", n=50)

    classify = build_classifier(classes)
    evaluate = dspy.Evaluate(devset=devset, metric=accuracy, num_threads=num_threads, display_progress=True)

    before = evaluate(classify)

    optimizer = dspy.MIPROv2(metric=accuracy, auto="light", num_threads=num_threads)
    optimized = optimizer.compile(classify, trainset=trainset)

    after = evaluate(optimized)
    print(f"Accuracy on held-out data: {before.score:.1f}% -> {after.score:.1f}%")
    print(optimized(text="What does a pending cash withdrawal mean?"))
    return before, after, optimized


if __name__ == "__main__":
    main()
