#!/usr/bin/env python3
"""Write one Pointcept config per run of the revision batch, and the job list.

All runs use the corrected labels (data prepared with prepare_data.py
--corrected, placed at Pointcept/data/pheno4d-fixed) and are tested with the
checkpoint of the last epoch.

Each config is a full copy of its base config in configs/ with four lines
replaced (train_plants, test_plants, data_root, grid_size) and the seed added.
A copy is used instead of `_base_` inheritance because the base configs build
their `data` dict from these variables when the file is read, so overriding
them in a child config would not reach the dict.

Run names: rev-<model>-<species>-f<fold>-g<grid>-s<seed>, with the grid
written as 0p5 for 0.5 mm. Folds: 0 tests plants 1-2, 1 tests plants 3-4,
2 tests plants 5-6; plant 7 is always in training.

The job list is in priority order:
  P1  fold 0 maize, 5 seeds
  P2  folds 1-2 maize, 3 seeds
  P3  folds 1-2 tomato, 3 seeds
  P4  grid size 2, 1, 0.5, 0.25 mm, tomato SparseUNet fold 0, 3 seeds
  P5  fold 0 tomato, 5 seeds

  python scripts/write_configs.py   ->  configs/revision/*.py, scripts/jobs.txt
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "configs")
OUT = os.path.join(HERE, "..", "configs", "revision")
DATA_ROOT = "data/pheno4d-fixed"

PLANTS = {"maize": [f"Maize{i:02d}" for i in range(1, 8)],
          "tomato": [f"Tomato{i:02d}" for i in range(1, 8)]}
FOLD_TEST = {0: (0, 1), 1: (2, 3), 2: (4, 5)}      # plants 1-2, 3-4, 5-6


def fmt_grid(g):
    # no dots in names: 0.5 -> 0p5, so config and experiment names stay plain
    return f"{g:g}".replace(".", "p")


def job_name(model, species, fold, grid, seed):
    return f"rev-{model}-{species}-f{fold}-g{fmt_grid(grid)}-s{seed}"


def make(model, species, fold, grid, seed):
    base = os.path.join(BASE, f"semseg-{model}-pheno4d-{species}.py")
    s = open(base, encoding="utf-8").read()
    plants = PLANTS[species]
    test = [plants[i] for i in FOLD_TEST[fold]]
    train = [p for p in plants if p not in test]
    subs = [(r"^train_plants\s*=.*$", f"train_plants = {train!r}"),
            (r"^test_plants\s*=.*$", f"test_plants  = {test!r}"),
            (r"^data_root\s*=.*$", f'data_root = "{DATA_ROOT}"'),
            (r"^grid_size\s*=.*$", f"grid_size = {grid}")]
    for pat, rep in subs:
        s, n = re.subn(pat, rep, s, count=1, flags=re.M)
        assert n == 1, (base, pat)
    name = job_name(model, species, fold, grid, seed)
    header = (f"# Written by scripts/write_configs.py, do not edit by hand.\n"
              f"# {name}: fold {fold} (test {test}), grid {grid} mm, seed {seed},\n"
              f"# corrected labels, evaluated with model_last.\n")
    s = header + s.rstrip() + f"\n\nseed = {seed}\n"
    return name, s


def check(text, train, test, grid, seed):
    ns = {"dict": dict}
    exec(text.replace("_base_", "_unused_base_"), ns)
    assert ns["data"]["train"]["split"] == train, "train split"
    assert ns["data"]["val"]["split"] == test and ns["data"]["test"]["split"] == test, "test split"
    for part in ("train", "val", "test"):
        assert ns["data"][part]["data_root"] == DATA_ROOT
    gs = [t["grid_size"] for t in ns["data"]["train"]["transform"] if t["type"] == "GridSample"]
    assert gs == [grid], gs
    assert ns["data"]["test"]["test_cfg"]["voxelize"]["grid_size"] == grid
    assert ns["seed"] == seed


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs, groups = [], []
    def add(tag, model, species, fold, grid, seed):
        name, text = make(model, species, fold, grid, seed)
        plants = PLANTS[species]; test = [plants[i] for i in FOLD_TEST[fold]]
        check(text, [p for p in plants if p not in test], test, grid, seed)
        if name not in jobs:
            open(os.path.join(OUT, name + ".py"), "w", encoding="utf-8").write(text)
            jobs.append(name); groups.append(tag)

    for m in ("spunet", "ptv3"):
        for s in range(5): add("P1 fold0 maize", m, "maize", 0, 0.5, s)
    for f in (1, 2):
        for m in ("spunet", "ptv3"):
            for s in range(3): add("P2 folds maize", m, "maize", f, 0.5, s)
    for f in (1, 2):
        for m in ("spunet", "ptv3"):
            for s in range(3): add("P3 folds tomato", m, "tomato", f, 0.5, s)
    for g in (2.0, 1.0, 0.5, 0.25):
        for s in range(3): add("P4 grid tomato", "spunet", "tomato", 0, g, s)
    for m in ("spunet", "ptv3"):
        for s in range(5): add("P5 fold0 tomato", m, "tomato", 0, 0.5, s)

    with open(os.path.join(HERE, "jobs.txt"), "w") as f:
        f.write("# Revision batch, priority order. One job per line; '#' lines are skipped.\n")
        last = None
        for j, g in zip(jobs, groups):
            if g != last:
                f.write(f"# --- {g}\n"); last = g
            f.write(j + "\n")
    from collections import Counter
    print(len(jobs), "jobs", dict(Counter(groups)))


if __name__ == "__main__":
    main()
