import pandas as pd

from vsd.data import load_asvspoof2019_la, load_manifest, subsample


def test_asvspoof_protocol_parsing(synthetic_root):
    df = load_asvspoof2019_la(synthetic_root)
    assert set(df.split) == {"train", "dev", "eval"}
    assert set(df.label) == {0, 1}
    assert (df[df.label == 1].attack == "-").all()
    # eval attacks must be disjoint from train attacks: that is the point of the benchmark
    train_attacks = set(df[(df.split == "train") & (df.label == 0)].attack)
    eval_attacks = set(df[(df.split == "eval") & (df.label == 0)].attack)
    assert train_attacks.isdisjoint(eval_attacks)


def test_audio_files_exist(synthetic_root):
    from pathlib import Path

    df = load_asvspoof2019_la(synthetic_root)
    assert all(Path(p).exists() for p in df.path)


def test_manifest_label_mapping_and_split(tmp_path):
    csv = tmp_path / "m.csv"
    pd.DataFrame(
        {"path": [f"a{i}.wav" for i in range(200)], "label": ["bona-fide", "spoof"] * 100}
    ).to_csv(csv, index=False)
    df = load_manifest(csv)
    assert set(df.label) == {0, 1}
    assert set(df.split) <= {"train", "dev", "eval"}


def test_subsample_is_stratified(synthetic_root):
    df = load_asvspoof2019_la(synthetic_root)
    sub = subsample(df, 20, seed=1)
    assert len(sub) <= 22
    assert set(sub.label) == {0, 1}
