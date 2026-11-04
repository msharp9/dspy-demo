import dspy

import optimizers

CLASSES = ["card_arrival", "lost_or_stolen_card"]


def examples(n):
    return [dspy.Example(text=f"msg {i}", label="card_arrival").with_inputs("text") for i in range(n)]


class StubMIPROv2:
    """Stands in for MIPROv2 so the test checks our wiring, not dspy's optimizer internals."""

    def __init__(self, metric, **kwargs):
        self.metric = metric

    def compile(self, student, trainset):
        return student


def test_build_classifier_restricts_labels():
    classify = optimizers.build_classifier(CLASSES)
    assert set(classify.predict.signature.output_fields) == {"reasoning", "label"}


def test_accuracy_metric():
    gold = dspy.Example(text="x", label="card_arrival")
    assert optimizers.accuracy(gold, dspy.Prediction(label="card_arrival"))
    assert not optimizers.accuracy(gold, dspy.Prediction(label="lost_or_stolen_card"))


def test_main_evaluates_before_and_after(fake_lm, monkeypatch):
    monkeypatch.setattr(dspy, "MIPROv2", StubMIPROv2)
    before, after, _ = optimizers.main(
        lm=fake_lm, trainset=examples(4), devset=examples(3), classes=CLASSES, num_threads=1
    )
    assert before.score == after.score == 100.0
