import class_demo
import saving


def test_saving_round_trips_both_formats(fake_lm, gsm8k_trainset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = saving.main(lm=fake_lm, trainset=gsm8k_trainset, out_dir="out")
    assert result.answer == "5"
    assert (tmp_path / "out" / "state.json").exists()
    assert (tmp_path / "out" / "whole").is_dir()


def test_class_demo_saves_and_answers(fake_lm, gsm8k_trainset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = class_demo.main(lm=fake_lm, trainset=gsm8k_trainset, save_dir="out")
    assert len(results) == len(class_demo.QUESTIONS)
    assert (tmp_path / "out" / "program.json").exists()
