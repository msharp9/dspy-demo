"""Optimize a math solver with GEPA (new in DSPy 3).

GEPA (Genetic-Pareto) is a reflective prompt optimizer. Unlike BootstrapFewShot or MIPROv2,
its metric returns written *feedback*, not just a score. A stronger "reflection" model reads
the program's attempts plus that feedback, figures out what went wrong, and rewrites the
instructions. It keeps a Pareto front of candidates that each win on different examples.

Guide: https://dspy.ai/current/getting-started/gepa-optimization/
"""

from pathlib import Path

import dspy
from dspy.datasets.gsm8k import parse_integer_answer

from config import local_lm, reflection_lm
from data import load_gsm8k


def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Score 1.0 or 0.0, plus feedback the reflection model can learn from."""
    expected = parse_integer_answer(str(example.answer))
    got = parse_integer_answer(str(prediction.answer))
    if got == expected:
        return dspy.Prediction(score=1.0, feedback=f"Correct: the answer is {example.answer}.")
    feedback = (
        f"Incorrect: the answer field was {prediction.answer!r} but the correct answer is {example.answer}. "
        "The answer field should contain only the final number. "
        f"Here is a reference solution:\n{example.gold_reasoning}"
    )
    return dspy.Prediction(score=0.0, feedback=feedback)


def accuracy(example, prediction, trace=None):
    return metric_with_feedback(example, prediction).score


def main(lm=None, reflection=None, trainset=None, valset=None, testset=None, max_metric_calls=200, num_threads=4):
    dspy.configure(lm=lm or local_lm())
    # GEPA learns from trainset and picks its best candidate on valset; testset stays unseen.
    trainset = trainset if trainset is not None else load_gsm8k("train", n=20, seed=1)
    valset = valset if valset is not None else load_gsm8k("train", n=20, seed=2)
    testset = testset if testset is not None else load_gsm8k("test", n=50)

    solver = dspy.ChainOfThought("question -> answer")
    evaluate = dspy.Evaluate(devset=testset, metric=accuracy, num_threads=num_threads, display_progress=True)
    before = evaluate(solver)

    optimizer = dspy.GEPA(
        metric=metric_with_feedback,
        reflection_lm=reflection or reflection_lm(),
        # A fixed budget keeps this demo to a few minutes. For real work use auto="light"/"medium"/"heavy".
        max_metric_calls=max_metric_calls,
        num_threads=num_threads,
    )
    optimized = optimizer.compile(solver, trainset=trainset, valset=valset)

    after = evaluate(optimized)
    print("\nInstructions GEPA wrote:\n", optimized.predict.signature.instructions, sep="")
    print(f"\nAccuracy on held-out GSM8K: {before.score:.1f}% -> {after.score:.1f}%")
    Path("dspy_gepa").mkdir(exist_ok=True)
    optimized.save("dspy_gepa/program.json")
    return before, after, optimized


if __name__ == "__main__":
    main()
