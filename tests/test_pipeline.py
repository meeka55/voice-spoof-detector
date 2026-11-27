import json

import pytest

from vsd import evaluate, train
from vsd.detector import Detector

# Synthetic audio has trivially separable artefacts, so these only prove the plumbing works.


@pytest.mark.parametrize("backend", ["gmm", "logreg", "svm"])
def test_train_evaluate_roundtrip(synthetic_root, tmp_path, backend):
    model = tmp_path / f"{backend}.joblib"
    train.main(
        [
            "--data-root", str(synthetic_root),
            "--backend", backend,
            "--n-components", "4",
            "--n-jobs", "1",
            "--out", str(model),
        ]
    )
    assert model.exists()
    out = tmp_path / "res"
    evaluate.main(
        ["--model", str(model), "--data-root", str(synthetic_root), "--split", "eval",
         "--n-jobs", "1", "--out-dir", str(out)]
    )
    metrics = json.loads((out / "metrics.json").read_text())
    assert 0.0 <= metrics["eer"] <= 1.0
    assert (out / "roc.png").exists() and (out / "scores.csv").exists()
    assert set(metrics["per_attack_eer"]) == {"A07", "A08", "A09"}


def test_train_calibrates_an_operating_point(synthetic_root, tmp_path):
    model = tmp_path / "calibrated.joblib"
    train.main(
        [
            "--data-root", str(synthetic_root),
            "--backend", "logreg",
            "--n-jobs", "1",
            "--target-far", "0.1",
            "--target-frr", "0.1",
            "--out", str(model),
        ]
    )
    det = Detector.load(str(model))
    op = det.meta["operating_point"]
    assert op["target_far"] == pytest.approx(0.1)
    assert op["target_frr"] == pytest.approx(0.1)
    assert det.accept_threshold == pytest.approx(op["accept"])
    assert det.reject_threshold == pytest.approx(op["reject"])
    assert det.reject_threshold <= det.accept_threshold

    out = tmp_path / "cal_res"
    evaluate.main(
        [
            "--model", str(model),
            "--data-root", str(synthetic_root),
            "--n-jobs", "1",
            "--ci-boot", "50",
            "--out-dir", str(out),
        ]
    )
    metrics = json.loads((out / "metrics.json").read_text())
    assert len(metrics["eer_ci95"]) == 2
    assert metrics["eer_ci95"][0] <= metrics["eer_ci95"][1]
    report = metrics["operating_point"]
    assert set(report) >= {
        "accept_threshold", "reject_threshold", "false_accept_rate", "false_reject_rate", "review_rate",
    }
    assert 0.0 <= report["review_rate"] <= 1.0
    assert 0.0 <= report["false_accept_rate"] <= 1.0


def test_evaluate_can_override_the_threshold_and_skip_the_ci(synthetic_root, tmp_path):
    model = tmp_path / "m.joblib"
    train.main(
        ["--data-root", str(synthetic_root), "--backend", "logreg", "--n-jobs", "1",
         "--out", str(model)]
    )
    out = tmp_path / "thr_res"
    evaluate.main(
        ["--model", str(model), "--data-root", str(synthetic_root), "--n-jobs", "1",
         "--threshold", "0.25", "--ci-boot", "0", "--out-dir", str(out)]
    )
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["threshold"] == pytest.approx(0.25)
    assert "eer_ci95" not in metrics
    assert "operating_point" not in metrics


def test_train_refuses_half_a_calibration(synthetic_root, tmp_path):
    with pytest.raises(SystemExit):
        train.main(
            ["--data-root", str(synthetic_root), "--n-jobs", "1",
             "--target-far", "0.1", "--out", str(tmp_path / "x.joblib")]
        )


def test_evaluate_requires_a_manifest_for_manifest_datasets(tmp_path):
    with pytest.raises(SystemExit):
        evaluate.main(["--model", str(tmp_path / "missing.joblib"), "--dataset", "manifest"])
