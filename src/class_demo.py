"""Optimize a math program with BootstrapFewShot, then save it and load it back."""

import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

from config import local_lm

QUESTIONS = [
    "If it takes 1 hour for 60 people to play an orchestra how long will it take for 600 people to play the same orchestra?",
    "Dan has 2 quarters, 3 dimes, 3 nickels, and 4 pennies. Does he have enough money to buy a pen for 75 cents?",
]


def load_trainset(n=1000):
    return GSM8K().train[:n]


def main(lm=None, trainset=None, save_dir="dspy_demo"):
    dspy.configure(lm=lm or local_lm())
    trainset = trainset if trainset is not None else load_trainset()

    program = dspy.ChainOfThought("question -> answer")
    optimizer = dspy.BootstrapFewShot(metric=gsm8k_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=10)
    compiled = optimizer.compile(program, trainset=trainset)

    compiled.save(f"./{save_dir}/", save_program=True)
    compiled.save(f"./{save_dir}/program.json", save_program=False)  # save the program.json for reference

    loaded = dspy.load(f"./{save_dir}/")
    results = [loaded(question=q) for q in QUESTIONS]
    for result in results:
        print(result)
    return results


if __name__ == "__main__":
    main()
