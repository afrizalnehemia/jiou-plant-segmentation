"""Plant-level splits for Pheno4D.

The split is over PLANTS, never over scans: two scans of the same plant on
consecutive days are nearly identical, so a scan-level split leaks the test set
into training. Pheno4D has 7 maize and 7 tomato plants; the dataset paper uses
5 train / 2 test. We keep that ratio and expose all folds so the whole thing can
be cross-validated instead of resting on one lucky partition.

The split is FIXED across training seeds. Seeds vary weight init and data order,
not which plants are held out. Otherwise seed variance and split variance get
confounded and the error bars mean nothing.
"""

MAIZE  = [f"Maize{i:02d}"  for i in range(1, 8)]
TOMATO = [f"Tomato{i:02d}" for i in range(1, 8)]


def folds(plants, n_test=2):
    """Contiguous, non-overlapping test blocks -> deterministic fold list."""
    out = []
    for start in range(0, len(plants), n_test):
        test = plants[start:start + n_test]
        if len(test) < n_test:                 # last remainder joins previous fold
            break
        out.append({"train": [p for p in plants if p not in test], "test": test})
    return out


SPLITS = {
    "maize":  folds(MAIZE),
    "tomato": folds(TOMATO),
}

DEFAULT_FOLD = 0   # fold used for the headline table; others for cross-validation

if __name__ == "__main__":
    for k, v in SPLITS.items():
        print(f"{k}: {len(v)} folds")
        for i, f in enumerate(v):
            print(f"  fold {i}: test={f['test']}  train={len(f['train'])} plants")
