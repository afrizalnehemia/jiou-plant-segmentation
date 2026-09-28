# Changelog

## 1.2.0 (2026-09-28)

- Results of the 53 runs used in the paper: three plant-level folds, five
  seeds on fold 0 and three on folds 1 and 2, a grid-size sweep, all trained
  and tested on corrected labels and evaluated with the final-epoch weights.
  Per-scan band counts are in `results/revision/`.
- `scripts/paper_stats.py` and `scripts/matched_gap.py` recompute every number
  in the paper from those counts.
- `docs/label-errors.md` now lists how to run each step with the published or
  the corrected labels, and gives the effect of the correction on the new runs.
- `results/label-audit-extents.csv`: soil, stem and tallest-leaf extents for
  all 126 scans.
- The first batch of runs moved to `results/batch1/`.

## 1.1.0 (2026-09-23)

- Second label check (stem vs leaf ids). Together with the first check it
  finds ten Pheno4D scans with interchanged labels. Corrections are in
  `jiou/labels.py` and documented in `docs/label-errors.md`.
- `load_pheno4d_txt` takes `corrected=True` and `return_ids=True`;
  `prepare_data.py` takes `--corrected`.
- `evaluate` takes `seed_labels`, for the instance-aware band.
- New `scripts/band_counts.py` and `scripts/summarise.py`: per-scan confusion
  counts for every band variant (k = 8, 16, 32, all-boundary, instance-aware),
  in-band accuracy, point spacing, and 95% confidence intervals.
- New `scripts/write_configs.py` and `scripts/queue.sh` for the second batch
  of runs (extra plant folds, grid-size sweep).
- Maize results rescored with corrected labels. Old scores kept as
  `results/scores-maize-raw.csv`.
- Scripts renamed: `prepare_data.py`, `train.sh`, `test.sh`, `score.py`,
  `figures.py`. `rescore_corrected_scan.py` removed, since the correction is
  now part of data preparation.
- README corrected: with semantic labels only stem-leaf transitions seed the
  band (leaf-leaf contact needs the instance-aware band).

## 1.0.0

First release.
