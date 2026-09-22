"""Map model predictions back onto the FULL-RESOLUTION point cloud.

Why this step exists
--------------------
Every model in this comparison voxelises or subsamples before inference. The
junction band we score on is only a few millimetres wide, so a coarse grid can
leave almost no points inside it -- meaning the junction metric would be computed
on whatever survived the grid rather than on the real geometry, and each model
would be scored on a different set of points. That is not a comparison.

So: every prediction is propagated back to the original points by nearest
neighbour before any metric is computed. All models are then scored on the exact
same point set, at the resolution the ground truth was annotated at.

Inputs
  --gt_root    converted Pheno4D root (Plant/scan/{coord,segment}.npy)
  --pred_root  model output, one .npy of predicted labels per scan, named
               <Plant>-<scan>.npy, in the model's own (possibly reduced) order
  --pred_coord optional: matching coords for the predictions. Omit when the model
               already writes one label per original point (Pointcept test mode
               does this), in which case the file is used as-is after a length check.
"""
import argparse, glob, os
import numpy as np
from scipy.spatial import cKDTree


def propagate(full_coord, pred_coord, pred_label):
    """Nearest-neighbour transfer from the reduced cloud to every original point."""
    tree = cKDTree(pred_coord)
    dist, idx = tree.query(full_coord, k=1, workers=-1)
    return pred_label[idx].astype(np.int16), dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--pred_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--pred_coord_root", default=None)
    a = ap.parse_args()

    os.makedirs(a.out_root, exist_ok=True)
    n_ok = n_prop = 0
    for pred_file in sorted(glob.glob(os.path.join(a.pred_root, "*.npy"))):
        name = os.path.splitext(os.path.basename(pred_file))[0]
        # Pointcept menulis "<Plant>-<scan>_pred.npy"; buang akhiran itu supaya
        # nama scan-nya cocok dengan folder ground truth.
        if name.endswith("_pred"):
            name = name[: -len("_pred")]
        if "-" not in name:
            print(f"lewati {name}: nama tidak berbentuk <Plant>-<scan>"); continue
        plant, scan = name.split("-", 1)
        gt_dir = os.path.join(a.gt_root, plant, scan)
        if not os.path.isdir(gt_dir):
            print(f"skip {name}: no ground truth at {gt_dir}"); continue

        full_coord = np.load(os.path.join(gt_dir, "coord.npy"))
        pred = np.load(pred_file).reshape(-1)

        if len(pred) == len(full_coord):
            out, maxd = pred.astype(np.int16), 0.0
            n_ok += 1
        else:
            if not a.pred_coord_root:
                raise SystemExit(
                    f"{name}: {len(pred)} predictions vs {len(full_coord)} points. "
                    "Reduced predictions need --pred_coord_root.")
            pc = np.load(os.path.join(a.pred_coord_root, name + ".npy"))
            out, dist = propagate(full_coord, pc, pred)
            maxd = float(dist.max()); n_prop += 1

        np.save(os.path.join(a.out_root, name + ".npy"), out)
        print(f"{name}: {len(out):>9,} labels"
              + (f"  (propagated, max NN dist {maxd:.2f} mm)" if maxd else "  (already full-res)"))

    print(f"\n{n_ok} already full-resolution, {n_prop} propagated -> {a.out_root}")


if __name__ == "__main__":
    main()
