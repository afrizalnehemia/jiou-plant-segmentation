#!/usr/bin/env python3
"""Per-scan confusion counts for every band variant, for all runs at once.

The band depends only on the annotation, so it is computed once per scan and
reused for every run that predicted that scan. For each scan, run and mask the
script stores the full 3x3 confusion matrix (soil, stem, leaf), so J-IoU,
per-class IoU and in-band accuracy can all be derived later by summarise.py
without touching the point clouds again.

Masks written per scan:
  global          all points
  O16_r{1,2,5,10} organ-only band, k = 16 (the paper's default)
  O8_r*, O32_r*   organ-only band with k = 8 and k = 32
  A16_r*          all-boundary band (soil contact counts as a boundary)
  I16_r*          instance-aware organ-only band: seeds also where two
                  different leaves touch

  python scripts/band_counts.py --raw data/Pheno4D --pred predictions/maize \\
      --out results/band-counts [--published-labels] [Plant-scan ...]

Labels are corrected with jiou/labels.py unless --published-labels is given.
Scans that already have an output file are skipped, so the script can be
stopped and restarted.
"""
import argparse, glob, os, sys, time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jiou import load_pheno4d_txt, boundary_seeds

RADII = [1, 2, 5, 10]


def band_distance(xyz, labels, k, organ_only):
    """Distance to the nearest boundary seed, and the number of seeds."""
    if organ_only:
        keep = np.flatnonzero(labels != 0)
        if len(keep) < 2:
            return np.full(len(xyz), np.inf), 0
        seeds = np.zeros(len(xyz), bool)
        seeds[keep[boundary_seeds(xyz[keep], labels[keep], k=min(k, len(keep) - 1))]] = True
    else:
        seeds = boundary_seeds(xyz, labels, k=k)
    if not seeds.any():
        return np.full(len(xyz), np.inf), 0
    d = np.empty(len(xyz))
    tree, step = cKDTree(xyz[seeds]), 500_000
    for s in range(0, len(xyz), step):
        d[s:s + step] = tree.query(xyz[s:s + step], k=1, workers=-1)[0]
    return d, int(seeds.sum())


def nn_spacing(xyz, mask, n=200_000, seed=0):
    """Median distance from a point to its nearest neighbour (subsampled)."""
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return float("nan")
    q = idx if len(idx) <= n else np.random.default_rng(seed).choice(idx, n, replace=False)
    return float(np.median(cKDTree(xyz[idx]).query(xyz[q], k=2, workers=-1)[0][:, 1]))


def count_scan(raw_root, pred_root, plant, scan, out_file, corrected):
    xyz, sem, ids = load_pheno4d_txt(os.path.join(raw_root, plant, scan + ".txt"),
                                     corrected=corrected, return_ids=True)
    variants = {"A16": band_distance(xyz, sem, 16, False),
                "O8": band_distance(xyz, sem, 8, True),
                "O16": band_distance(xyz, sem, 16, True),
                "O32": band_distance(xyz, sem, 32, True),
                "I16": band_distance(xyz, ids, 16, True)}
    masks = {"global": (np.ones(len(xyz), bool), -1)}
    for v, (d, nseed) in variants.items():
        for r in RADII:
            masks[f"{v}_r{r}"] = (d <= r, nseed)
    meta = {"plant": plant, "scan": scan, "n": len(xyz),
            "nn_all": nn_spacing(xyz, np.ones(len(xyz), bool)),
            "nn_plant": nn_spacing(xyz, sem != 0),
            "n_soil": int((sem == 0).sum()), "n_stem": int((sem == 1).sum()),
            "n_leaf": int((sem == 2).sum()), "n_leaves": int(len(np.unique(ids[ids >= 2])))}
    rows = []
    for rd in sorted(d for d in glob.glob(os.path.join(pred_root, "*")) if os.path.isdir(d)):
        f = os.path.join(rd, f"{plant}-{scan}.npy")
        if not os.path.exists(f):
            continue
        model, _, seed = os.path.basename(rd).rpartition("-seed")
        pred = np.load(f).astype(np.int64)
        assert len(pred) == len(sem), (rd, plant, scan)
        code = sem * 3 + pred
        for name, (m, nseed) in masks.items():
            cm = np.bincount(code[m], minlength=9).reshape(3, 3)
            rows.append({"model": model, "seed": seed, "mask": name, "nseed": nseed,
                         "npts": int(m.sum()),
                         **{f"c{g}{p}": int(cm[g, p]) for g in range(3) for p in range(3)}})
    df = pd.DataFrame(rows)
    for k, v in meta.items():
        df[k] = v
    df.to_csv(out_file + ".tmp", index=False)
    os.replace(out_file + ".tmp", out_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder with Maize01 ... Tomato07")
    ap.add_argument("--pred", required=True, help="folder of <model>-seed<N> run folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--published-labels", action="store_true",
                    help="use the labels exactly as published")
    ap.add_argument("only", nargs="*", help="restrict to these <Plant>-<scan> names")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    first = sorted(d for d in glob.glob(os.path.join(a.pred, "*")) if os.path.isdir(d))[0]
    for f in sorted(glob.glob(os.path.join(first, "*.npy"))):
        name = os.path.basename(f)[:-4]
        if a.only and name not in a.only:
            continue
        out_file = os.path.join(a.out, name + ".csv")
        if os.path.exists(out_file):
            continue
        plant, scan = name.split("-", 1)
        t0 = time.time()
        count_scan(a.raw, a.pred, plant, scan, out_file, not a.published_labels)
        print(f"{name}  {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
