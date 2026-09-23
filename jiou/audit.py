"""Geometric label checks for annotated plant point clouds.

Both tests use only the labelled coordinates, no model output. They compare
the vertical extent of labels within a scan, measured as the spread between
the 5th and 95th percentile of z. The full range is not used because scans
often contain a few noise points below the soil, which would make a normal
soil plane look tens of millimetres thick.

Test 1 (soil vs stem): the soil should be a thin sheet and the stem an upright
structure, so a scan is flagged when the soil is taller than the stem.

Test 2 (stem vs leaf ids): in the raw collar labels the stem (id 1) runs from
the base to the youngest leaves, so a scan is flagged when any single leaf id
is taller than the stem.
"""
import glob
import os

import numpy as np

from .io import read_raw


def robust_extent(z, lo=5, hi=95):
    """Spread between two percentiles of z (default 5th to 95th)."""
    p_lo, p_hi = np.percentile(z, [lo, hi])
    return float(p_hi - p_lo)


def audit_scan(path, min_points=20):
    """Run both tests on one raw scan (collar column for maize).

    Returns a dict with the extent of every id, the tallest leaf id, and one
    boolean per test.
    """
    raw = read_raw(path)
    z = raw[:, 2]
    ids = raw[:, 3].astype(np.int64)
    ext = {int(u): robust_extent(z[ids == u]) for u in np.unique(ids)
           if (ids == u).sum() >= min_points}
    leaves = {u: e for u, e in ext.items() if u >= 2}
    tallest = max(leaves, key=leaves.get) if leaves else None
    soil, stem = ext.get(0), ext.get(1)
    t1 = soil is not None and stem is not None and soil > stem
    t2 = stem is not None and tallest is not None and leaves[tallest] > stem
    return {"path": path, "extent": ext, "tallest_leaf": tallest,
            "soil_vs_stem": bool(t1), "stem_vs_leaf": bool(t2),
            "flagged": bool(t1 or t2)}


def audit_directory(root, pattern="*/*_a.txt", verbose=True):
    """Audit every annotated scan under root and return the flagged ones."""
    flagged = []
    for path in sorted(glob.glob(os.path.join(root, pattern))):
        r = audit_scan(path)
        if not r["flagged"]:
            continue
        flagged.append(r)
        if verbose:
            e = r["extent"]
            why = []
            if r["soil_vs_stem"]:
                why.append(f"soil {e.get(0, float('nan')):.1f} mm > stem {e.get(1, float('nan')):.1f} mm")
            if r["stem_vs_leaf"]:
                why.append(f"leaf id {r['tallest_leaf']} {e[r['tallest_leaf']]:.1f} mm "
                           f"> stem {e.get(1, float('nan')):.1f} mm")
            print(f"FLAGGED {os.path.relpath(path, root)}  " + ", ".join(why))
    return flagged
