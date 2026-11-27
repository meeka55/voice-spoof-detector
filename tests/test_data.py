import pandas as pd
import pytest

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


def test_manifest_label_spellings_all_map_to_the_right_class(tmp_path):
    csv = tmp_path / "m.csv"
    spellings = ["Bona Fide", "REAL", "1", "True", "bona_fide", "spoof", "Fake", "0", "no", "deepfake"]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    pd.DataFrame({"path": [f"a{i}.wav" for i in range(len(spellings))], "label": spellings}).to_csv(
        csv, index=False
    )
    df = load_manifest(csv)
    assert list(df.label) == labels


def test_manifest_unknown_label_is_refused(tmp_path):
    csv = tmp_path / "m.csv"
    pd.DataFrame({"path": ["a.wav", "b.wav"], "label": ["bonafide", "typo-label"]}).to_csv(
        csv, index=False
    )
    with pytest.raises(ValueError, match="unrecognised label"):
        load_manifest(csv)


def test_subsample_none_and_large_keep_everything(synthetic_root):
    df = load_asvspoof2019_la(synthetic_root)
    assert len(subsample(df, None)) == len(df)
    assert len(subsample(df, len(df))) == len(df)


@pytest.mark.parametrize("bad", [0, -5])
def test_subsample_refuses_a_meaningless_size(synthetic_root, bad):
    df = load_asvspoof2019_la(synthetic_root)
    with pytest.raises(ValueError, match="max_utts"):
        subsample(df, bad)
