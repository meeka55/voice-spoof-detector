"""Train a detector.

Example (real data):
    python -m vsd.train --data-root data/LA --features lfcc --backend gmm \
        --out models/lfcc_gmm.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import __version__
from .calibrate import calibrate
from .data import load_asvspoof2019_la, load_manifest, subsample
from .detector import BACKENDS, Detector
from .features import FeatureConfig
from .metrics import compute_eer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", choices=["asvspoof2019la", "manifest"], default="asvspoof2019la")
    p.add_argument("--data-root", help="ASVspoof 2019 LA root (folder containing the protocols)")
    p.add_argument("--manifest", help="CSV with columns path,label[,attack,split]")
    p.add_argument("--features", choices=["lfcc", "mfcc"], default="lfcc")
    p.add_argument("--backend", choices=BACKENDS, default="gmm")
    p.add_argument("--n-components", type=int, default=64, help="GMM components")
    p.add_argument("--max-train-utts", type=int, default=None, help="subsample for quick runs")
    p.add_argument("--max-dev-utts", type=int, default=3000)
    p.add_argument(
        "--target-far",
        type=float,
        default=None,
        help="calibrate the accept threshold to this false-accept budget (needs --target-frr)",
    )
    p.add_argument(
        "--target-frr",
        type=float,
        default=None,
        help="calibrate the reject threshold to this false-reject budget (needs --target-far)",
    )
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models/model.joblib")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.target_far is None) != (args.target_frr is None):
        parser.error("--target-far and --target-frr must be given together")
    if args.dataset == "asvspoof2019la":
        if not args.data_root:
            raise SystemExit("--data-root is required for asvspoof2019la")
        df = load_asvspoof2019_la(args.data_root)
    else:
        if not args.manifest:
            raise SystemExit("--manifest is required for dataset=manifest")
        df = load_manifest(args.manifest, seed=args.seed)

    train = subsample(df[df.split == "train"], args.max_train_utts, args.seed)
    dev = subsample(df[df.split == "dev"], args.max_dev_utts, args.seed)
    if len(train) == 0:
        raise SystemExit("no training utterances for split=train; check the dataset layout")
    if len(dev) == 0 or dev.label.nunique() < 2:
        raise SystemExit(
            "the dev split needs both bona fide and spoof utterances to choose a threshold"
        )
    print(f"train utterances: {len(train)}  dev utterances: {len(dev)}")

    det = Detector(
        FeatureConfig(kind=args.features),
        backend=args.backend,
        n_components=args.n_components,
        seed=args.seed,
    )
    t0 = time.time()
    det.fit(train.path.tolist(), train.label.tolist(), n_jobs=args.n_jobs)
    fit_s = time.time() - t0

    dev_scores = det.score_many(dev.path.tolist(), n_jobs=args.n_jobs)
    dev_eer, thr = compute_eer(dev_scores, dev.label.values)
    det.threshold = thr  # operating point chosen on dev, never on eval
    det.meta = {
        "package_version": __version__,
        "backend": args.backend,
        "features": args.features,
        "dataset": args.dataset,
        "n_train": len(train),
        "dev_eer": dev_eer,
        "seed": args.seed,
    }
    extra: dict = {}
    if args.target_far is not None and args.target_frr is not None:
        op = calibrate(dev_scores, dev.label.values, args.target_far, args.target_frr)
        det.accept_threshold = op.accept
        det.reject_threshold = op.reject
        det.meta["operating_point"] = op.as_dict()
        extra["operating_point"] = op.as_dict()
        if op.band_empty:
            print(
                "warning: the review band is empty (reject >= accept at these budgets); "
                "the model has little separation at this operating point",
                file=sys.stderr,
            )
    det.save(args.out)
    print(json.dumps({**det.meta, "threshold": thr, "fit_seconds": round(fit_s, 1), **extra}, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
