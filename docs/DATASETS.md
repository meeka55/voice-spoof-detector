# Datasets

## ASVspoof 2019 Logical Access (primary)
Download "LA" from the ASVspoof 2019 page on the University of Edinburgh DataShare
(search "ASVspoof 2019 database"). It is several GB, so budget time and disk. Unpack so that you have:

```
data/LA/
  ASVspoof2019_LA_train/flac/*.flac
  ASVspoof2019_LA_dev/flac/*.flac
  ASVspoof2019_LA_eval/flac/*.flac
  ASVspoof2019_LA_cm_protocols/
    ASVspoof2019.LA.cm.train.trn.txt
    ASVspoof2019.LA.cm.dev.trl.txt
    ASVspoof2019.LA.cm.eval.trl.txt
```

Check the dataset license and cite the ASVspoof 2019 database paper in your README.

## In-the-Wild (generalisation test)
Audio of public figures, real and deepfaked, collected from the internet. Build a manifest and train on ASVspoof,
then evaluate on this set **without retraining** to measure cross-dataset generalisation:

```csv
path,label,attack,split
wavs/0.wav,bona-fide,-,eval
wavs/1.wav,spoof,spoof,eval
```

```bash
python -m vsd.evaluate --model models/lfcc_gmm.joblib --dataset manifest \
    --manifest in_the_wild.csv --split eval --out-dir results/itw
```
Expect a large drop compared with ASVspoof; report it honestly, since that gap is the finding.
