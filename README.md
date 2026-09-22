# J-IoU: junction-restricted evaluation for 3D plant organ segmentation

Organ-level segmentation of plant point clouds is almost always reported as a
single global mIoU. That number is dominated by the interiors of large organs.
The traits people actually want — leaf insertion angle, leaf count, internode
length — are determined at the **leaf collar**, where the leaf meets the stem,
and that region is tiny: points within 1 mm of an organ–organ boundary are
**0.19 % of a maize scan and 0.73 % of a tomato scan** in Pheno4D.

A model can therefore fail at every leaf collar in the plant and still post a
global score that looks close to solved.

**J-IoU** restricts the score to a narrow band around ground-truth organ
boundaries, adapting the trimap / Boundary-IoU idea from 2D image segmentation
to irregularly sampled 3D point clouds. On Pheno4D, across two architectures
and five random seeds each:

| Species | Model | Global mIoU | J-IoU (r = 1 mm) | gap |
|---|---|---|---|---|
| Maize  | SparseUNet | 0.511 ± 0.039 | 0.333 ± 0.040 | 0.178 |
| Maize  | PTv3       | 0.520 ± 0.026 | 0.319 ± 0.034 | 0.201 |
| Tomato | SparseUNet | 0.925 ± 0.003 | 0.569 ± 0.010 | 0.356 |
| Tomato | PTv3       | 0.930 ± 0.001 | 0.564 ± 0.010 | 0.366 |

On tomato the gap is 37× the between-seed standard deviation. The difference
*between the two architectures*, by contrast, is not resolvable at five seeds —
which is worth keeping in mind when reading single-run comparisons.

## Install

```bash
git clone https://github.com/afrizalnehemia/jiou-plant-segmentation.git
cd jiou-plant-segmentation
pip install -r requirements.txt
```

The metric itself needs only `numpy` and `scipy`. `pandas` makes reading raw
Pheno4D text files roughly two orders of magnitude faster, and `matplotlib` is
needed only for the figure scripts.

## Using the metric on your own data

```python
from jiou import evaluate, load_pheno4d_txt

xyz, gt = load_pheno4d_txt("data/Pheno4D/Maize01/M01_0325_a.txt")
pred = my_model(xyz)                      # per-point labels, same order as xyz

res = evaluate(xyz, gt, pred, radii=[1, 2, 5, 10], seed_ignore=(0,))
print("global mIoU:", res["mIoU_global"])
for b in res["bands"]:
    print(f"  r={b['radius']:g}mm  J-IoU={b['J_mIoU']:.4f}"
          f"  band={b['band_point_share']*100:.2f}% of points"
          f"  gap={b['hidden_gap']:.4f}")
```

`xyz` and `pred` must be at the **original full resolution** of the cloud. If
your network works on voxels, propagate predictions back to the original points
first (`scripts/export_predictions.py` does this by nearest neighbour).
Evaluating at voxel resolution scores an easier problem, because voxelisation
discards the most information exactly where sampling is densest — at the
junction.

### The two band definitions

Boundary seeds are points whose *k* nearest neighbours include a different
ground-truth label. `seed_ignore` controls which labels may seed a band:

* `seed_ignore=()` — **all-boundary band**. Every label transition seeds,
  including soil–plant contact.
* `seed_ignore=(0,)` — **organ-only band**. Soil is dropped before boundaries
  are located, so only stem–leaf and leaf–leaf transitions seed.

This matters more than it looks. Soil contact is the easiest transition in a
scan; on soil-dominated scans it supplies most of the band, and the metric then
silently measures ground separation rather than the leaf collar it is named
for. Report both, or state which you used.

Bands are computed from the **ground truth only**, never from predictions, so
the same band applies to every model compared on a scan.

## Reproducing the paper

1. **Get the data.** Download Pheno4D from
   <https://www.ipb.uni-bonn.de/html/projects/Pheno4D/> and unpack it to
   `data/Pheno4D/` (so that `data/Pheno4D/Maize01/M01_0313_a.txt` exists).
2. **Audit the labels** (optional but recommended — it is how the anomaly in
   `docs/label-anomaly.md` was found):
   ```bash
   python -c "from jiou import audit_directory; audit_directory('data/Pheno4D')"
   ```
3. **Convert to the training format:** `python scripts/prepare_pheno4d.py`
4. **Train.** `scripts/train_seeds.sh` runs five seeds per configuration using
   [Pointcept](https://github.com/Pointcept/Pointcept); the configs in
   `configs/` are the ones used in the paper. Copy `configs/pheno4d_dataset.py`
   into Pointcept's `pointcept/datasets/` as `pheno4d.py`.
5. **Export predictions at full resolution:** `scripts/test_and_export.sh`
6. **Score:**
   ```bash
   python scripts/score_runs.py \
       --gt_root data/pheno4d --results_root predictions/maize \
       --out_csv results/scores-maize.csv
   ```
7. **Figures:** `python scripts/make_paper_figures.py`

Steps 1–6 need a GPU and take several hours. **If you only want to check the
numbers**, `results/*.csv` contains the per-scan scores for all 20 runs
(2 species × 2 architectures × 5 seeds), and steps 7 and the aggregation work
directly from those.

## The label anomaly

The geometric audit flagged one scan out of 126 in Pheno4D —
`Tomato02/T02_0325_a` — in which the soil and stem labels are interchanged.
Scored against the published labels, that scan yields a global mIoU of 0.496 in
every one of ten independent training runs; with the labels corrected it yields
0.950. `results/scores-tomato-raw.csv` and
`results/scores-tomato-corrected.csv` contain both. Details and the evidence
are in [`docs/label-anomaly.md`](docs/label-anomaly.md).

## Repository layout

```
jiou/                  the metric, the loaders and the label audit (importable)
scripts/               data preparation, training, scoring, figures
configs/               Pointcept configs used in the paper + the dataset class
results/               per-scan scores for all 20 runs
docs/                  the label anomaly write-up
```

## Citing

If you use this, please cite the paper (see `CITATION.cff`). The evaluation
code is released so that paired reporting of global and junction-restricted
scores costs nothing to adopt.

## License

MIT — see `LICENSE`.
