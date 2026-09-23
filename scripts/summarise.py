#!/usr/bin/env python3
"""Turn the band counts into the numbers reported in the paper.

  python scripts/summarise.py results/band-counts

Prints, per species and model, mean and standard deviation over seeds of
global mIoU and J-IoU for every band variant and radius, per-class IoU,
in-band accuracy, band size, point spacing, and the Welch comparison of the
two models (difference, 95% confidence interval, Cohen's d, p-value).

Scores follow the paper: IoU is computed for stem and leaf only, a class
missing from both labels and predictions inside a mask is left out of the
mean, each scan is scored first, scans are averaged within a run, and runs are
summarised over seeds.
"""
import glob, os, sys

import numpy as np
import pandas as pd
from scipy import stats

RADII = [1, 2, 5, 10]
VARIANTS = ["O16", "A16", "I16", "O8", "O32"]


def load(folder):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(folder, "*.csv")))],
                   ignore_index=True)
    for c in (1, 2):
        tp = df[f"c{c}{c}"]
        fp = sum(df[f"c{g}{c}"] for g in range(3) if g != c)
        fn = sum(df[f"c{c}{p}"] for p in range(3) if p != c)
        den = tp + fp + fn
        present = df[{1: "n_stem", 2: "n_leaf"}[c]] > 0
        df[f"iou{c}"] = np.where(present & (den > 0), tp / den.where(den > 0, 1), np.nan)
    df["miou"] = df[["iou1", "iou2"]].mean(axis=1, skipna=True)
    diag = df["c00"] + df["c11"] + df["c22"]
    df["acc"] = np.where(df["npts"] > 0, diag / df["npts"].where(df["npts"] > 0, 1), np.nan)
    df["species"] = np.where(df.plant.str.startswith("Maize"), "Maize", "Tomato")
    df["arch"] = np.where(df.model.str.contains("ptv3"), "PTv3", "SparseUNet")
    return df


def per_seed(df, sp, arch, mask, col="miou"):
    d = df[(df.species == sp) & (df.arch == arch) & (df["mask"] == mask)]
    return d.groupby("seed")[col].mean().sort_index().to_numpy()


def ms(v):
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def welch(a, b):
    """Difference b - a with a Welch 95% CI, Cohen's d (pooled sd) and p."""
    diff = b.mean() - a.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = stats.t.ppf(0.975, dof)
    p = stats.ttest_ind(b, a, equal_var=False).pvalue
    d = diff / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return diff, diff - t * se, diff + t * se, d, p


def main(folder):
    df = load(folder)
    archs = ("SparseUNet", "PTv3")
    for sp in sorted(df.species.unique()):
        print(f"\n===== {sp}")
        for a in archs:
            print(f"\n{a}: global {ms(per_seed(df, sp, a, 'global'))}")
            for v in VARIANTS:
                print(f"  {v:4s}", "  ".join(f"r{r}: {ms(per_seed(df, sp, a, f'{v}_r{r}'))}" for r in RADII))
            for c, nm in ((1, "stem"), (2, "leaf")):
                print(f"  {nm} IoU  global {ms(per_seed(df, sp, a, 'global', f'iou{c}'))}  ",
                      "  ".join(f"r{r}: {ms(per_seed(df, sp, a, f'O16_r{r}', f'iou{c}'))}" for r in RADII))
            print("  accuracy  global", ms(per_seed(df, sp, a, "global", "acc")), " ",
                  "  ".join(f"r{r}: {ms(per_seed(df, sp, a, f'O16_r{r}', 'acc'))}" for r in RADII))
        one = df[(df.species == sp) & (df.seed == df.seed.min()) & (df.arch == archs[0])]
        print("\nband size, organ-only k = 16 (median points per scan, mean share of scan)")
        for r in RADII:
            y = one[one["mask"] == f"O16_r{r}"]
            print(f"  r{r}: {y.npts.median():.0f} points, {100 * (y.npts / y.n).mean():.2f}%")
        g = one[one["mask"] == "global"]
        print(f"point spacing on the plant: median {g.nn_plant.median():.3f} mm "
              f"(range {g.nn_plant.min():.3f} to {g.nn_plant.max():.3f})")
        print(f"\n{archs[1]} minus {archs[0]}, Welch")
        for m in ["global"] + [f"O16_r{r}" for r in RADII]:
            diff, lo, hi, d, p = welch(per_seed(df, sp, archs[0], m), per_seed(df, sp, archs[1], m))
            print(f"  {m:8s} {diff:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  d {d:+.2f}  p {p:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/band-counts")
