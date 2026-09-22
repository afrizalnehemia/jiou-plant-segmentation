"""Loading Pheno4D scans and model predictions."""
import numpy as np

try:
    import pandas as pd
    _HAVE_PANDAS = True
except ImportError:                                     # pragma: no cover
    _HAVE_PANDAS = False


def load_pheno4d_txt(path, scheme="collar"):
    """Read a raw Pheno4D scan -> (xyz float64, semantic labels int64).

    Semantic scheme: 0 soil, 1 stem, 2 leaf. Column 4 of the raw file carries
    the leaf-collar annotation (0 soil, 1 stem, >=2 individual leaves) and
    column 5, where present, the leaf-tip annotation.

    pandas' C parser is used when available; np.loadtxt is far too slow for
    files of several million rows.
    """
    if _HAVE_PANDAS:
        raw = pd.read_csv(path, sep=r"\s+", header=None, engine="c").to_numpy()
    else:                                               # pragma: no cover
        raw = np.loadtxt(path)

    xyz = raw[:, :3].astype(np.float64)
    if raw.shape[1] >= 5:
        lab = raw[:, 3 if scheme == "collar" else 4].astype(np.int64)
    else:
        lab = raw[:, 3].astype(np.int64)

    if scheme == "collar":
        sem = np.where(lab == 0, 0, np.where(lab == 1, 1, 2))
    else:
        sem = np.where(lab == 0, 0, 2)
    return xyz, sem


def load_predictions(path):
    """Per-point predicted labels, one per point of the ORIGINAL cloud.

    Predictions must already be propagated back to full resolution; evaluating
    at voxel resolution measures an easier problem than the one trait
    extraction faces (see README).
    """
    return np.load(path).astype(np.int64)
