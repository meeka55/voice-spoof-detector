"""Evaluate a trained detector and write honest, reproducible artefacts.

Outputs (in --out-dir): metrics.json, scores.csv, roc.png, per_attack_eer.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from .data import load_asvspoof2019_la, load_manifest, subsample
from .detector import Detector
from .metrics import per_attack_eer, roc_points, summarize

matplotlib.use("Agg")  # headless backend; must be set before any figure is created


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", choices=["asvspoof2019la", "manifest"], default="asvspoof2019la")
    p.add_argument("--data-root")
    p.add_argument("--manifest")
    p.add_argument("--split", default="eval", choices=["train", "dev", "eval"])
    p.add_argument("--max-utts", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results/run")
    args = p.parse_args(argv)

    det = Detector.load(args.model)
    df = (
        load_asvspoof2019_la(args.data_root)
        if args.dataset == "asvspoof2019la"
        else load_manifest(args.manifest, seed=args.seed)
    )
    df = subsample(df[df.split == args.split], args.max_utts, args.seed).reset_index(drop=True)
    print(f"evaluating on {len(df)} utterances from split={args.split}")

    df["score"] = det.score_many(df.path.tolist(), n_jobs=args.n_jobs)
    summary = summarize(df.score.values, df.label.values)
    attacks = per_attack_eer(df.score.values, df.label.values, df.attack.values)
    pred_bona = df.score >= det.threshold
    summary.update(
        {
            "split": args.split,
            "dev_threshold": det.threshold,
            "false_accept_rate_at_dev_threshold": float(
                (pred_bona & (df.label == 0)).sum() / max((df.label == 0).sum(), 1)
            ),
            "false_reject_rate_at_dev_threshold": float(
                (~pred_bona & (df.label == 1)).sum() / max((df.label == 1).sum(), 1)
            ),
            "per_attack_eer": attacks,
            "model_meta": det.meta,
        }
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "scores.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    far, frr = roc_points(df.score.values, df.label.values)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(far, frr)
    ax.plot([0, 1], [0, 1], "--", lw=0.8, color="grey")
    ax.set(xlabel="False accept rate (spoof accepted)", ylabel="False reject rate (bona fide rejected)",
           title=f"DET-style curve, {args.split}  (EER {summary['eer']:.2%})")
    fig.tight_layout()
    fig.savefig(out / "roc.png", dpi=150)
    plt.close(fig)

    if attacks:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(list(attacks), [v * 100 for v in attacks.values()])
        ax.set(ylabel="EER (%)", title="Per-attack EER (bona fide vs one attack)")
        fig.tight_layout()
        fig.savefig(out / "per_attack_eer.png", dpi=150)
        plt.close(fig)

    print(json.dumps({k: v for k, v in summary.items() if k != "model_meta"}, indent=2))


if __name__ == "__main__":
    main()
