"""Loading Pheno4D scans and model predictions."""
import os

import numpy as np

from .labels import apply_corrections

try:
    import pandas as pd
    _HAVE_PANDAS = True
except ImportError:                                     # pragma: no cover
    _HAVE_PANDAS = False


def read_raw(path):
    """Raw Pheno4D text file as a float array (x y z label [label])."""
    if _HAVE_PANDAS:
        return pd.read_csv(path, sep=r"\s+", header=None, engine="c").to_numpy()
    return np.loadtxt(path)                             # pragma: no cover


def load_pheno4d_txt(path, scheme="collar", corrected=False, return_ids=False):
    """Read a raw Pheno4D scan and return (xyz, semantic labels).

    Semantic labels are 0 soil, 1 stem, 2 leaf. For maize, column 4 holds the
    leaf-collar annotation (0 soil, 1 stem, 2 and up one id per leaf) and
    column 5 the leaf-tip annotation (0 soil, 1 and up one id per leaf, no stem
    class). Tomato files have one label column with the collar convention.

    corrected=True applies the swaps in jiou/labels.py. It only affects the
    collar column and the tomato labels. The file name must follow the
    dataset layout, <Plant>/<scan>.txt, for the scan to be recognised.

    return_ids=True also returns the raw ids after correction, which the
    instance-aware band needs.
    """
    raw = read_raw(path)
    xyz = raw[:, :3].astype(np.float64)
    if raw.shape[1] >= 5:
        ids = raw[:, 3 if scheme == "collar" else 4].astype(np.int64)
    else:
        ids = raw[:, 3].astype(np.int64)

    if corrected and scheme == "collar":
        plant = os.path.basename(os.path.dirname(os.path.abspath(path)))
        scan = os.path.splitext(os.path.basename(path))[0]
        ids = apply_corrections(plant, scan, ids)

    if scheme == "collar":
        sem = np.where(ids == 0, 0, np.where(ids == 1, 1, 2))
    else:
        sem = np.where(ids == 0, 0, 2)
    return (xyz, sem, ids) if return_ids else (xyz, sem)


def load_predictions(path):
    """Per-point predicted labels, one per point of the original cloud.

    Predictions must already be mapped back to full resolution. Scoring at
    voxel resolution measures an easier problem than the one faced in trait
    extraction.
    """
    return np.load(path).astype(np.int64)
