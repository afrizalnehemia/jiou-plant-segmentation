"""Known label errors in Pheno4D and the corrections applied in the paper.

Raw label ids (leaf-collar column for maize, the only label column for tomato):
0 soil, 1 stem, 2 and up one id per leaf. Each entry below swaps two raw ids.

The ten scans were found with the two tests in jiou/audit.py. See
docs/label-errors.md for the evidence. The leaf-tip column of the maize files
is not affected and is never modified here.
"""
import numpy as np

CORRECTIONS = {
    # soil and stem interchanged
    ("Tomato02", "T02_0325_a"): [(0, 1)],
    # stem carries id 2, first leaf carries id 1 (all seven scans of this plant)
    **{("Maize02", f"M02_{d}_a"): [(1, 2)]
       for d in ("0313", "0315", "0317", "0319", "0321", "0324", "0325")},
    # stem carries a leaf id in two training scans
    ("Maize03", "M03_0321_a"): [(1, 3)],
    ("Maize03", "M03_0324_a"): [(1, 4)],
}


def apply_corrections(plant, scan, ids):
    """Return a copy of the raw id array with the listed swaps applied.

    plant and scan are names such as "Maize02" and "M02_0313_a". Scans without
    an entry are returned unchanged.
    """
    ids = np.asarray(ids).copy()
    for a, b in CORRECTIONS.get((plant, scan), []):
        ma, mb = ids == a, ids == b
        ids[ma], ids[mb] = b, a
    return ids
