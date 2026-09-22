"""Geometric label-integrity audit for annotated plant point clouds.

The test uses only the labelled coordinates -- no model, no predictions. In a
correctly labelled scan the soil class is a thin planar sheet at the base of
the scene and the stem class is an upright structure with a far greater
vertical extent. A scan whose soil class is *thicker* than its stem class is
flagged.

Thickness is the 5th-95th percentile spread of z, not the full range: scans
routinely contain stray noise points below the soil plane, and a raw range
reports a normal soil plane as tens of millimetres thick, producing false
positives.
"""
import glob
import os

import numpy as np

from .io import load_pheno4d_txt


def robust_thickness(z, lo=5, hi=95):
    p_lo, p_hi = np.percentile(z, [lo, hi])
    return float(p_hi - p_lo)


def audit_scan(path, soil=0, stem=1):
    """-> dict with per-class robust thickness and a boolean `flagged`."""
    xyz, gt = load_pheno4d_txt(path)
    t = {}
    for c in np.unique(gt):
        t[int(c)] = robust_thickness(xyz[gt == c, 2])
    flagged = soil in t and stem in t and t[soil] > t[stem]
    return {"path": path, "thickness": t, "flagged": bool(flagged)}


def audit_directory(root, pattern="*/*_a.txt", verbose=True):
    """Audit every scan under `root`; returns the list of flagged results."""
    flagged = []
    for path in sorted(glob.glob(os.path.join(root, pattern))):
        r = audit_scan(path)
        if r["flagged"]:
            flagged.append(r)
        if verbose and r["flagged"]:
            t = r["thickness"]
            print(f"FLAGGED {os.path.relpath(path, root)}  "
                  f"soil={t.get(0, float('nan')):.1f}mm  "
                  f"stem={t.get(1, float('nan')):.1f}mm")
    return flagged
