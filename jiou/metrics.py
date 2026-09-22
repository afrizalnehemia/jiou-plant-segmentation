"""
Junction-aware evaluation for 3D plant point cloud segmentation.

Adapts the 2D trimap / Boundary-IoU evaluation protocol to plant point clouds:
instead of scoring every point equally, restrict the score to a narrow band
around ground-truth organ junctions, where the literature consistently reports
failure but never measures it.

Author: Afrizal Nehemia Toscany
"""
import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- data loading

def load_pheno4d(path, scheme="collar"):
    """Pheno4D maize scan -> (xyz, semantic, instance).

    Column 4 = leaf-collar scheme : 0 soil, 1 stem, >=2 leaf instances
    Column 5 = leaf-tip scheme    : 0 soil, >=1 leaf instances (no stem class)
    Tomato scans carry a single label column with the collar-style convention.
    """
    raw = np.loadtxt(path)
    xyz = raw[:, :3].astype(np.float64)
    if raw.shape[1] >= 5:
        lab = raw[:, 3 if scheme == "collar" else 4].astype(np.int64)
    else:
        lab = raw[:, 3].astype(np.int64)

    inst = lab.copy()
    if scheme == "collar":
        sem = np.where(lab == 0, 0, np.where(lab == 1, 1, 2))   # soil/stem/leaf
    else:
        sem = np.where(lab == 0, 0, 2)                          # soil/leaf
    return xyz, sem, inst


# ------------------------------------------------------- junction band (trimap)

def boundary_seeds(xyz, labels, k=16, tree=None):
    """Points whose k-NN neighbourhood contains another label."""
    tree = tree or cKDTree(xyz)
    _, idx = tree.query(xyz, k=k + 1, workers=-1)
    return (labels[idx[:, 1:]] != labels[:, None]).any(axis=1)


def distance_to_boundary(xyz, labels, k=16, tree=None, seed_ignore=()):
    """Distance from every point to the nearest label boundary.

    Computed once; every band radius is then a threshold on this array. The
    obvious implementation -- a ball query per seed, unioned in a Python loop --
    is orders of magnitude slower and does not scale to a full dataset sweep.

    seed_ignore drops those labels BEFORE boundaries are located, which changes
    what the band means. With soil ignored, the soil-plant contact stops being a
    boundary and only organ-organ transitions -- stem/leaf and leaf/leaf -- seed
    the band. Distances are still measured for every point, so nothing is lost
    from the scored set; only the definition of "junction" narrows.
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
             seed_ignore=()):
    """Global mIoU vs junction-restricted mIoU across a sweep of band radii.

    seed_ignore=(0,) restricts the band to organ-organ junctions by refusing to
    treat the soil-plant contact as a boundary. That contact is the easiest
    transition in the scan and, left in, it dominates the band on soil-heavy
    scans -- so the metric ends up measuring ground separation rather than the
    leaf collar it is named after.
    """
    classes = classes if classes is not None else sorted(set(gt.tolist()))
    scored = [c for c in classes if not (exclude_soil and c == 0)]
    tree = cKDTree(xyz)
    dist, _ = distance_to_boundary(xyz, gt, k=k, tree=tree, seed_ignore=seed_ignore)

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
