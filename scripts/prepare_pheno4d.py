"""Convert raw Pheno4D scans into the per-scan npy layout the training code reads.

Raw format (verified against the data, not guessed):
  maize  : 5 columns  x y z collar tip
             collar -> 0 soil, 1 stem, >=2 leaf instances
             tip    -> 0 soil, >=1 leaf instances (no stem class at all)
  tomato : 4 columns  x y z label
             label  -> 0 soil, 1 stem, >=2 leaf instances

Output, one directory per scan:
  coord.npy      float32 (N,3)  millimetres, centred on the plant
  segment.npy    int16   (N,)   0 soil, 1 stem, 2 leaf
  instance.npy   int32   (N,)   -1 for soil and stem, 0..k-1 per leaf
  meta.json                     counts, extent, provenance

Units are MILLIMETRES and stay that way end to end. The junction radius is a
physical distance, so switching to metres halfway through would silently change
what the metric measures. Grid sizes in the model configs are in mm to match.
"""
import argparse, json, os, sys
import numpy as np

SEM_SOIL, SEM_STEM, SEM_LEAF = 0, 1, 2


def read_points(path):
    """Read an ASCII scan into a float array.

    np.loadtxt parses in pure Python and needs roughly a minute per 100 MB. The
    labelled half of Pheno4D is several gigabytes, so that alone would cost hours
    before a single epoch runs. pandas' C parser reads the same files 20-50x
    faster; numpy stays as the fallback so the script never hard-depends on it.
    """
    try:
        import pandas as pd
        return pd.read_csv(path, sep=r"\s+", header=None,
                           dtype=np.float64, engine="c").to_numpy()
    except ImportError:
        return np.loadtxt(path)


def parse_scan(path, scheme="collar"):
    raw = read_points(path)
    if raw.ndim != 2 or raw.shape[1] < 4:
        raise ValueError(f"{path}: expected >=4 columns, got shape {raw.shape}")
    xyz = raw[:, :3].astype(np.float32)

    if raw.shape[1] >= 5:                       # maize
        lab = raw[:, 3 if scheme == "collar" else 4].astype(np.int64)
        tip_scheme = (scheme == "tip")
    else:                                       # tomato
        lab = raw[:, 3].astype(np.int64)
        tip_scheme = False

    if tip_scheme:
        seg = np.where(lab == 0, SEM_SOIL, SEM_LEAF).astype(np.int16)
        leaf_src = lab                          # every non-soil label is a leaf id
        leaf_mask = lab > 0
    else:
        seg = np.where(lab == 0, SEM_SOIL,
              np.where(lab == 1, SEM_STEM, SEM_LEAF)).astype(np.int16)
        leaf_src = lab
        leaf_mask = lab >= 2

    inst = np.full(len(lab), -1, dtype=np.int32)
    if leaf_mask.any():
        ids = np.unique(leaf_src[leaf_mask])
        remap = {int(v): i for i, v in enumerate(ids)}
        inst[leaf_mask] = np.array([remap[int(v)] for v in leaf_src[leaf_mask]], np.int32)
    return xyz, seg, inst


def write_scan(out_dir, xyz, seg, inst, src):
    os.makedirs(out_dir, exist_ok=True)
    plant = seg != SEM_SOIL
    centre = xyz[plant].mean(0) if plant.any() else xyz.mean(0)
    coord = (xyz - centre).astype(np.float32)

    np.save(os.path.join(out_dir, "coord.npy"), coord)
    np.save(os.path.join(out_dir, "segment.npy"), seg)
    np.save(os.path.join(out_dir, "instance.npy"), inst)
    meta = {
        "source": os.path.basename(src),
        "n_points": int(len(coord)),
        "extent_mm": np.ptp(coord, axis=0).round(2).tolist(),
        "n_soil": int((seg == SEM_SOIL).sum()),
        "n_stem": int((seg == SEM_STEM).sum()),
        "n_leaf": int((seg == SEM_LEAF).sum()),
        "n_leaf_instances": int(inst.max() + 1) if (inst >= 0).any() else 0,
        "units": "mm",
    }
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_root", help="folder containing Maize01..Tomato07")
    ap.add_argument("out_root")
    ap.add_argument("--scheme", default="collar", choices=["collar", "tip"],
                    help="maize labelling scheme; tomato ignores this")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N scans")
    a = ap.parse_args()

    plants = sorted(d for d in os.listdir(a.raw_root)
                    if os.path.isdir(os.path.join(a.raw_root, d)))
    if not plants:
        sys.exit(f"no plant folders under {a.raw_root}")

    total, skipped = 0, 0
    for plant in plants:
        pdir = os.path.join(a.raw_root, plant)
        scans = sorted(f for f in os.listdir(pdir) if f.endswith("_a.txt"))
        for s in scans:                     # only *_a.txt carry labels
            src = os.path.join(pdir, s)
            dst = os.path.join(a.out_root, plant, os.path.splitext(s)[0])
            if os.path.exists(os.path.join(dst, "meta.json")):
                skipped += 1; continue
            xyz, seg, inst = parse_scan(src, a.scheme)
            m = write_scan(dst, xyz, seg, inst, src)
            total += 1
            print(f"{plant}/{s}: {m['n_points']:>9,} pts  "
                  f"soil {m['n_soil']/m['n_points']:5.1%}  "
                  f"stem {m['n_stem']/m['n_points']:5.1%}  "
                  f"leaf {m['n_leaf']/m['n_points']:5.1%}  "
                  f"{m['n_leaf_instances']} leaves")
            if a.limit and total >= a.limit:
                print("limit reached"); return
    print(f"\nconverted {total} scans, skipped {skipped} already present")


if __name__ == "__main__":
    main()
