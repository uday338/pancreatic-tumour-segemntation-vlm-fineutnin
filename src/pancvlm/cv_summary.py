"""Paired 5-fold comparison of nnU-Net experiments.

Reads nnUNet_results/DatasetXXX_*/<trainer>__<plans>__<config>/fold_k/validation/summary.json
for each experiment, reports per-fold tumor (label 2) and pancreas (label 1) Dice, and paired
differences against the baseline with a t-based 95% confidence interval.

    python -m pancvlm.cv_summary --baseline 1 --compare 3 6 --out results/cv_summary.json
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

from .common import load_json, save_json


def fold_dice(results_root: Path, dataset_id: int, label: str, config: str,
              trainer: str, plans: str, folds: list[int]) -> list[float]:
    ds = sorted(results_root.glob(f"Dataset{dataset_id:03d}_*"))
    if len(ds) != 1:
        raise SystemExit(f"Dataset{dataset_id:03d}_* not found (or ambiguous) in {results_root}")
    base = ds[0] / f"{trainer}__{plans}__{config}"
    out = []
    for k in folds:
        summ = load_json(base / f"fold_{k}" / "validation" / "summary.json")
        out.append(float(summ["mean"][label]["Dice"]))
    return out


def paired_stats(a: list[float], b: list[float]) -> dict:
    """Stats of d = b - a over folds."""
    d = np.asarray(b) - np.asarray(a)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    try:
        from scipy import stats
        tcrit = float(stats.t.ppf(0.975, n - 1))
    except Exception:  # pragma: no cover
        tcrit = {4: 2.776}.get(n - 1, 1.96)
    half = tcrit * sd / math.sqrt(n) if n > 1 else float("nan")
    return {
        "per_fold_diff": d.round(4).tolist(),
        "mean_diff": round(mean, 4),
        "sd_diff": round(sd, 4),
        "ci95": [round(mean - half, 4), round(mean + half, 4)],
        "wins": int((d > 0).sum()),
        "losses": int((d < 0).sum()),
        "ci_includes_zero": bool(mean - half <= 0 <= mean + half),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=int, required=True)
    ap.add_argument("--compare", type=int, nargs="+", required=True)
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--trainer", default="nnUNetTrainer")
    ap.add_argument("--plans", default="nnUNetPlans")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=Path("results/cv_summary.json"))
    a = ap.parse_args()

    root = Path(os.environ["nnUNet_results"])
    report: dict = {"folds": a.folds, "experiments": {}}
    ids = [a.baseline, *a.compare]
    for label, lname in (("2", "tumor"), ("1", "pancreas")):
        for d in ids:
            vals = fold_dice(root, d, label, a.config, a.trainer, a.plans, a.folds)
            report["experiments"].setdefault(str(d), {})[lname] = {
                "per_fold": [round(v, 4) for v in vals], "mean": round(float(np.mean(vals)), 4)}

    print(f"{'fold':>6} " + " ".join(f"{'D' + str(d):>10}" for d in ids))
    tum = {d: report["experiments"][str(d)]["tumor"]["per_fold"] for d in ids}
    for i, k in enumerate(a.folds):
        print(f"{k:>6} " + " ".join(f"{tum[d][i]:>10.4f}" for d in ids))
    print(f"{'mean':>6} " + " ".join(f"{np.mean(tum[d]):>10.4f}" for d in ids))

    report["paired_vs_baseline"] = {}
    for d in a.compare:
        s = paired_stats(tum[a.baseline], tum[d])
        report["paired_vs_baseline"][str(d)] = s
        print(f"D{d} - D{a.baseline}: mean {s['mean_diff']:+.4f}, sd {s['sd_diff']:.4f}, "
              f"95% CI {s['ci95']}, wins/losses {s['wins']}/{s['losses']}")
    save_json(report, a.out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
