"""Score every exported run with both protocols and emit one tidy CSV.

Produces, per (model, seed, scan): global mIoU and junction-restricted mIoU at
several band radii. Downstream aggregation then gives mean +/- std across seeds,
which is the number the literature never reports.
"""
import argparse, csv, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jiou import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--gt_root", required=True)
ap.add_argument("--results_root", required=True, help="folder of <config>-seedN dirs")
ap.add_argument("--out_csv", required=True)
ap.add_argument("--radii", type=float, nargs="+", default=[1, 2, 5, 10])
a = ap.parse_args()

rows = []
for run_dir in sorted(d for d in glob.glob(os.path.join(a.results_root, "*")) if os.path.isdir(d)):
    run = os.path.basename(run_dir)
    model, seed = (run.rsplit("-seed", 1) + ["?"])[:2]
    for pf in sorted(glob.glob(os.path.join(run_dir, "*.npy"))):
        name = os.path.splitext(os.path.basename(pf))[0]
        plant, scan = name.split("-", 1)
        gt_dir = os.path.join(a.gt_root, plant, scan)
        coord = np.load(os.path.join(gt_dir, "coord.npy"))
        gt = np.load(os.path.join(gt_dir, "segment.npy")).astype(np.int64)
        pred = np.load(pf).astype(np.int64)
        if len(pred) != len(gt):
            print(f"!! {run}/{name}: {len(pred)} vs {len(gt)} -- run export_predictions.py first")
            continue
        # Both band definitions, computed side by side so they can be compared
        # rather than chosen up front:
        #   J -- every label boundary, soil-plant contact included
        #   O -- organ-organ junctions only (soil does not count as a boundary)
        # Soil contact is the easiest transition in a scan. Left in, it dominates
        # the band on soil-heavy scans, and the metric then quietly measures
        # ground separation instead of the leaf collar.
        r = evaluate(coord, gt, pred, a.radii)
        o = evaluate(coord, gt, pred, a.radii, seed_ignore=(0,))
        row = {"model": model, "seed": seed, "plant": plant, "scan": scan,
               "n_points": r["n_points"], "mIoU_global": r["mIoU_global"]}
        for c, v in r["IoU_global"].items():
            row[f"IoU_c{c}_global"] = v
        for pre, res in (("J", r), ("O", o)):
            for b in res["bands"]:
                t = f"{b['radius']:g}"
                row[f"{pre}_mIoU_r{t}"] = b["J_mIoU"]
                row[f"{pre}_band_share_r{t}"] = b["band_point_share"]
                row[f"{pre}_hidden_gap_r{t}"] = b["hidden_gap"]
                for c, v in b["J_IoU"].items():
                    row[f"{pre}_IoU_c{c}_r{t}"] = v
        rows.append(row)
        rr = f"{a.radii[len(a.radii)//2]:g}"
        print(f"{run:30s} {name:22s} global {row['mIoU_global']:.4f}"
              f"  J(r{rr}) {row[f'J_mIoU_r{rr}']:.4f}"
              f"  O(r{rr}) {row[f'O_mIoU_r{rr}']:.4f}")

if not rows:
    sys.exit("no scored runs found")
keys = sorted({k for r in rows for k in r})
order = ["model", "seed", "plant", "scan", "n_points", "mIoU_global"]
keys = order + [k for k in keys if k not in order]
with open(a.out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(f"\nwrote {a.out_csv}  ({len(rows)} rows)")
