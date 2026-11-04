"""Two ways to save a compiled program: state only (JSON) vs. the whole program."""

from pathlib import Path

import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

from config import local_lm

QUESTION = "If it takes 1 hour for 60 people to play an orchestra how long will it take for 600 people to play the same orchestra?"


def load_trainset(n=1000):
    return GSM8K().train[:n]


def as_dict(demo):
    # Demos can come back as plain dicts or as dspy.Example objects depending on how they were loaded.
    return demo if isinstance(demo, dict) else demo.toDict()


def main(lm=None, trainset=None, out_dir="dspy_program"):
    dspy.configure(lm=lm or local_lm())
    trainset = trainset if trainset is not None else load_trainset()

    program = dspy.ChainOfThought("question -> answer")
    optimizer = dspy.BootstrapFewShot(metric=gsm8k_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5)
    compiled = optimizer.compile(program, trainset=trainset)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1. State only: saves demos/instructions. To load, recreate the same program and call .load().
    compiled.save(f"./{out_dir}/state.json", save_program=False)
    from_state = dspy.ChainOfThought("question -> answer")
    from_state.load(f"./{out_dir}/state.json")

    # 2. Whole program: saves the architecture too, so dspy.load() rebuilds it without the original code.
    compiled.save(f"./{out_dir}/whole/", save_program=True)
    from_program = dspy.load(f"./{out_dir}/whole/")

    for loaded in (from_state, from_program):
        assert len(compiled.predict.demos) == len(loaded.predict.demos)
        for original_demo, loaded_demo in zip(compiled.predict.demos, loaded.predict.demos, strict=True):
            assert as_dict(original_demo) == as_dict(loaded_demo)
        assert str(compiled.predict.signature) == str(loaded.predict.signature)

    result = from_program(question=QUESTION)
    print(result)
    return result


if __name__ == "__main__":
    main()
