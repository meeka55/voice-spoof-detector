"""Evaluate a trained detector and write honest, reproducible artefacts.

Outputs (in --out-dir): metrics.json, scores.csv, roc.png, per_attack_eer.png

The report includes a bootstrap confidence interval for the EER, because a single
point estimate on a few thousand utterances says nothing about how much it can move.
When the model carries a calibrated operating point (``train --target-far/--target-frr``)
the realised false-accept, false-reject and review rates are reported as well.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from .data import load_asvspoof2019_la, load_manifest, subsample
from .detector import Detector
from .metrics import eer_ci, per_attack_eer, roc_points, summarize

matplotlib.use("Agg")  # headless backend; must be set before any figure is created


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", choices=["asvspoof2019la", "manifest"], default="asvspoof2019la")
    p.add_argument("--data-root")
    p.add_argument("--manifest")
    p.add_argument("--split", default="eval", choices=["train", "dev", "eval"])
    p.add_argument("--max-utts", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override the decision threshold (disables a calibrated review band for this run)",
    )
    p.add_argument(
        "--ci-boot",
        type=int,
        default=200,
        help="bootstrap replicates for the EER confidence interval; 0 disables it",
    )
    p.add_argument("--out-dir", default="results/run")
    return p


def _operating_point_report(det: Detector, scores, labels) -> dict:
    """Realised rates of the model's calibrated band on this split."""
    decisions = [det.decide(float(s)) for s in scores]
    labels = list(labels)
    n_bona = sum(1 for lab in labels if lab == 1)
    n_spoof = len(labels) - n_bona
    n_review = sum(1 for d in decisions if d == "review")
    accepted = sum(1 for d, lab in zip(decisions, labels) if lab == 0 and d == "bonafide")
    rejected = sum(1 for d, lab in zip(decisions, labels) if lab == 1 and d == "spoof")
    return {
        "accept_threshold": det.accept_threshold,
        "reject_threshold": det.reject_threshold,
        "false_accept_rate": round(accepted / max(n_spoof, 1), 6),
        "false_reject_rate": round(rejected / max(n_bona, 1), 6),
        "review_rate": round(n_review / max(len(decisions), 1), 6),
        "n_review": n_review,
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dataset == "manifest" and not args.manifest:
        parser.error("--dataset manifest requires --manifest")
    if args.dataset == "asvspoof2019la" and not args.data_root:
        parser.error("--dataset asvspoof2019la requires --data-root")
    if args.ci_boot < 0:
        parser.error("--ci-boot must be >= 0")

    det = Detector.load(args.model)
    df = (
        load_asvspoof2019_la(args.data_root)
        if args.dataset == "asvspoof2019la"
        else load_manifest(args.manifest, seed=args.seed)
    )
    df = subsample(df[df.split == args.split], args.max_utts, args.seed).reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(f"no utterances for split={args.split}; nothing to evaluate")
    if df.label.nunique() < 2:
        raise SystemExit(f"split={args.split} has only one class; EER needs both")
    print(f"evaluating on {len(df)} utterances from split={args.split}")

    dev_threshold = det.threshold
    has_band = det.accept_threshold is not None
    if args.threshold is not None:
        det.threshold = float(args.threshold)
        det.accept_threshold = None
        det.reject_threshold = None
        has_band = False

    df["score"] = det.score_many(df.path.tolist(), n_jobs=args.n_jobs)
    summary: dict = summarize(df.score.values, df.label.values)
    attacks = per_attack_eer(df.score.values, df.label.values, df.attack.values)
    pred_bona = df.score >= det.threshold
    summary.update(
        {
            "split": args.split,
            "threshold": det.threshold,
            "dev_threshold": dev_threshold,
            "false_accept_rate_at_threshold": float(
                (pred_bona & (df.label == 0)).sum() / max((df.label == 0).sum(), 1)
            ),
            "false_reject_rate_at_threshold": float(
                (~pred_bona & (df.label == 1)).sum() / max((df.label == 1).sum(), 1)
            ),
            "per_attack_eer": attacks,
            "model_meta": det.meta,
        }
    )
    if args.ci_boot > 0:
        lo, hi = eer_ci(df.score.values, df.label.values, n_boot=args.ci_boot, seed=args.seed)
        summary["eer_ci95"] = [lo, hi]
    if has_band:
        summary["operating_point"] = _operating_point_report(
            det, df.score.values, df.label.values
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
