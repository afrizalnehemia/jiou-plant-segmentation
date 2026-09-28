#!/usr/bin/env python3
"""Junction gap on matched scans, and the maize global comparison with and without
the seedling scans that have no junction.

The gap is computed per scan (global mIoU minus J-IoU of the same scan) and then
averaged like the scores, so scans with an undefined J-IoU (the first scan of
maize plants 3 to 6, which has no leaf label) are left out of it.

  python scripts/matched_gap.py   ->  results/revision/matched-gap.txt
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarise import load, welch
SK = os.path.join(HERE, "..", "results", "revision")
lines = []
def p(x): print(x); lines.append(x)

rows = []
for sp in ("maize", "tomato"):
    for f in (0, 1, 2):
        d = load(os.path.join(SK, f"band-counts-{sp}-f{f}"))
        d = d[d.model.isin([f"spunet-{sp}-f{f}", f"ptv3-{sp}-f{f}"])]
        g = d[d["mask"] == "global"].set_index(["model", "seed", "scan"]).miou
        for r in (1, 2, 5, 10):
            j = d[d["mask"] == f"O16_r{r}"].set_index(["model", "seed", "scan"]).miou
            t = pd.DataFrame({"g": g, "j": j}).reset_index(); t["sp"], t["fold"], t["r"] = sp, f, r
            rows.append(t)
D = pd.concat(rows); D["arch"] = np.where(D.model.str.contains("ptv3"), "PTv3", "SparseUNet")

def per_seed(x):
    m = x[x.j.notna()]
    return pd.DataFrame({"g_all": x.groupby("seed").g.mean(), "g_matched": m.groupby("seed").g.mean(),
                         "j": m.groupby("seed").j.mean(), "gap": (m.g - m.j).groupby(m.seed).mean()})

p("===== gap at r = 1 mm, matched scans (mean over seeds; gap / sd of J-IoU)")
for (sp, f, a), x in D[D.r == 1].groupby(["sp", "fold", "arch"]):
    s = per_seed(x)
    p(f"{sp:6s} f{f} {a:10s} global all {s.g_all.mean():.3f}  matched {s.g_matched.mean():.3f}  "
      f"J {s.j.mean():.3f}  gap {s.gap.mean():.3f}  gap/sd {s.gap.mean() / s.j.std(ddof=1):.1f}")
for r in (1, 2, 5, 10):
    p(f"\n===== pooled over folds, seeds 0-2, r = {r} mm")
    for (sp, a), x in D[(D.r == r) & (D.seed <= 2)].groupby(["sp", "arch"]):
        s = per_seed(x)
        p(f"{sp:6s} {a:10s} global all {s.g_all.mean():.3f}  matched {s.g_matched.mean():.3f}  "
          f"J {s.j.mean():.3f}  gap {s.gap.mean():.3f}")

p("\n===== maize, PTv3 minus SparseUNet on global mIoU, all scans vs matched scans")
for f in (0, 1, 2):
    x = D[(D.sp == "maize") & (D.fold == f) & (D.r == 1)]
    for basis, q in (("all", x), ("matched", x[x.j.notna()])):
        a = q[q.arch == "SparseUNet"].groupby("seed").g.mean().to_numpy()
        b = q[q.arch == "PTv3"].groupby("seed").g.mean().to_numpy()
        diff, lo, hi, d, pv = welch(a, b)
        p(f"fold {f} {basis:8s} {diff:+.4f}  95% CI [{lo:+.3f}, {hi:+.3f}]  d {d:+.2f}  p {pv:.3f}")
    un = x[x.j.isna()]
    for (scan, a), v in un.groupby(["scan", "arch"]).g:
        p(f"   seedling scan {scan} {a:10s} global per seed " + ", ".join(f"{y:.3f}" for y in v))
open(os.path.join(SK, "matched-gap.txt"), "w").write("\n".join(lines) + "\n")
