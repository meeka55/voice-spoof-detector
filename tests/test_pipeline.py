import json

import pytest

from vsd import evaluate, train

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
