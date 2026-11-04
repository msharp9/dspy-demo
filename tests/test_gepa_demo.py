import dspy

import gepa_demo


class StubGEPA:
    """Stands in for GEPA so the test checks our wiring, not dspy's optimizer internals."""

    def __init__(self, metric, **kwargs):
        self.metric = metric
        self.kwargs = kwargs

    def compile(self, student, trainset, valset):
        return student


def test_metric_scores_and_explains():
    gold = dspy.Example(question="q", answer="42", gold_reasoning="6 * 7 = 42")
    right = gepa_demo.metric_with_feedback(gold, dspy.Prediction(answer="42"))
    wrong = gepa_demo.metric_with_feedback(gold, dspy.Prediction(answer="41"))
    assert right.score == 1.0
    assert wrong.score == 0.0
    assert "6 * 7 = 42" in wrong.feedback


def test_main_evaluates_before_and_after(fake_lm, gsm8k_trainset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dspy, "GEPA", StubGEPA)
    before, after, _ = gepa_demo.main(
        lm=fake_lm,
        reflection=fake_lm,
        trainset=gsm8k_trainset,
        valset=gsm8k_trainset,
        testset=gsm8k_trainset,
        num_threads=1,
    )
    assert before.score == after.score  # stub optimizer returns the program unchanged
    assert (tmp_path / "dspy_gepa" / "program.json").exists()
