"""
Junction-restricted evaluation for 3D plant point cloud segmentation.

Adapts the 2D trimap and Boundary IoU idea to plant point clouds. Instead of
scoring every point, the score is restricted to a narrow band around the
ground-truth boundaries between organs.
"""
import numpy as np
from scipy.spatial import cKDTree


# ------------------------------------------------------- junction band (trimap)

def boundary_seeds(xyz, labels, k=16, tree=None, chunk=200_000):
    """Points whose k nearest neighbours include a different label.

    The query runs in chunks so that scans of several million points fit in
    a few GB of memory.
    """
    tree = tree or cKDTree(xyz)
    out = np.zeros(len(xyz), bool)
    for s in range(0, len(xyz), chunk):
        _, idx = tree.query(xyz[s:s + chunk], k=k + 1, workers=-1)
        out[s:s + chunk] = (labels[idx[:, 1:]] != labels[s:s + chunk, None]).any(axis=1)
    return out


def distance_to_boundary(xyz, labels, k=16, tree=None, seed_ignore=()):
    """Distance from every point to the nearest label boundary.

    The distance is computed once, and every band radius is then a threshold
    on the same array. A ball query per seed would give the same band but is
    much slower on a full sweep.

    seed_ignore removes those labels before the boundaries are located. With
    soil ignored (seed_ignore=(0,)), soil contact no longer counts as a
    boundary. With semantic labels only stem-leaf transitions then produce
    seeds, because all leaves share one label. Pass leaf-instance ids as
    `labels` to also seed where two different leaves touch. Distances are
    still measured for every point.
    """
    tree = tree or cKDTree(xyz)
    if len(seed_ignore):
        keep = ~np.isin(labels, list(seed_ignore))
        if keep.sum() < 2:
            return np.full(len(xyz), np.inf), np.zeros(len(xyz), bool)
        sub = np.flatnonzero(keep)
        sub_seeds = boundary_seeds(xyz[keep], labels[keep], k=min(k, keep.sum() - 1))
        seeds = np.zeros(len(xyz), bool)
        seeds[sub[sub_seeds]] = True
    else:
        seeds = boundary_seeds(xyz, labels, k=k, tree=tree)
    if not seeds.any():
        return np.full(len(xyz), np.inf), seeds
    dist, _ = cKDTree(xyz[seeds]).query(xyz, k=1, workers=-1)
    return dist, seeds


def junction_band(xyz, labels, radius, k=16, tree=None, dist=None):
    """Boolean mask: points within `radius` of a ground-truth label boundary."""
    if dist is None:
        dist, _ = distance_to_boundary(xyz, labels, k=k, tree=tree)
    return dist <= radius


# ------------------------------------------------------------------- metrics

def iou_per_class(gt, pred, classes):
    out = {}
    for c in classes:
        g, p = gt == c, pred == c
        union = (g | p).sum()
        out[c] = float((g & p).sum() / union) if union else float("nan")
    return out


def miou(gt, pred, classes):
    v = [x for x in iou_per_class(gt, pred, classes).values() if not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def evaluate(xyz, gt, pred, radii, classes=None, k=16, exclude_soil=True,
             seed_ignore=(), seed_labels=None):
    """Global mIoU and junction-restricted mIoU over a sweep of band radii.

    seed_ignore=(0,) gives the organ-only band used in the paper, in which
    soil contact is not a boundary. Soil contact is the easiest transition in
    a scan, and on scans with a lot of soil it would otherwise supply most of
    the band.

    seed_labels, if given, are used to locate the boundaries instead of gt,
    for example leaf-instance ids for the instance-aware band. Scoring always
    uses gt and pred.
    """
    classes = classes if classes is not None else sorted(set(gt.tolist()))
    scored = [c for c in classes if not (exclude_soil and c == 0)]
    tree = cKDTree(xyz)
    seeds_from = gt if seed_labels is None else seed_labels
    dist, _ = distance_to_boundary(xyz, seeds_from, k=k, tree=tree, seed_ignore=seed_ignore)

    res = {"n_points": int(len(xyz)),
           "mIoU_global": miou(gt, pred, scored),
           "IoU_global": iou_per_class(gt, pred, scored),
           "bands": []}

    for r in radii:
        m = dist <= r
        share = float(m.mean())
        row = {"radius": float(r),
               "band_point_share": share,
               "J_mIoU": miou(gt[m], pred[m], scored) if m.any() else float("nan"),
               "J_IoU": iou_per_class(gt[m], pred[m], scored) if m.any() else {}}
        row["hidden_gap"] = res["mIoU_global"] - row["J_mIoU"]
        res["bands"].append(row)
    return res
