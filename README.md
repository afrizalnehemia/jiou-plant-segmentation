# J-IoU: junction-restricted evaluation for 3D plant organ segmentation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22893259.svg)](https://doi.org/10.5281/zenodo.22893259)

Organ segmentation of plant point clouds is usually reported as one global
mIoU. Each class-wise IoU in that number is accumulated over the whole extent
of the class, and only a small part of it lies near the junctions between
organs. In Pheno4D, points within 1 mm of a stem-leaf boundary are 0.29 % of a
maize scan and 0.73 % of a tomato scan. Traits such as leaf insertion angle,
leaf count and internode length are measured at exactly these junctions.

J-IoU scores a segmentation only inside a band around the ground-truth
boundaries between organs. It adapts the trimap and Boundary IoU evaluation
from 2D image segmentation to 3D point clouds.

Results on Pheno4D, two architectures, five seeds each, mean ± sd over seeds,
corrected labels (see [Label errors](#label-errors-in-pheno4d)):

| Species | Model | Global mIoU | J-IoU (r = 1 mm) | Gap |
|---|---|---|---|---|
| Maize  | SparseUNet | 0.782 ± 0.094 | 0.484 ± 0.052 | 0.297 |
| Maize  | PTv3       | 0.682 ± 0.109 | 0.439 ± 0.035 | 0.243 |
| Tomato | SparseUNet | 0.925 ± 0.003 | 0.569 ± 0.010 | 0.356 |
| Tomato | PTv3       | 0.930 ± 0.001 | 0.564 ± 0.010 | 0.366 |

On tomato the gap is 37 times the standard deviation between seeds. The
difference between the two architectures at the junction is not resolved with
five seeds.

## Install

```bash
git clone https://github.com/afrizalnehemia/jiou-plant-segmentation.git
cd jiou-plant-segmentation
pip install -r requirements.txt
```

The metric needs only `numpy` and `scipy`. `pandas` reads the raw Pheno4D text
files much faster, and `matplotlib` is only needed for the figures.

## Using the metric

```python
from jiou import evaluate, load_pheno4d_txt

xyz, gt = load_pheno4d_txt("data/Pheno4D/Maize01/M01_0325_a.txt", corrected=True)
pred = my_model(xyz)                      # one label per point, same order as xyz

res = evaluate(xyz, gt, pred, radii=[1, 2, 5, 10], seed_ignore=(0,))
print("global mIoU:", res["mIoU_global"])
for b in res["bands"]:
    print(f"r = {b['radius']:g} mm  J-IoU = {b['J_mIoU']:.4f}  "
          f"band = {100 * b['band_point_share']:.2f} % of points")
```

`xyz` and `pred` must be at the original resolution of the scan. If your
network works on voxels, map the predictions back to the original points first
(`scripts/export_predictions.py` does this by nearest neighbour). Scoring at
voxel resolution measures an easier problem, because voxelisation removes most
detail where the sampling is densest, which is near the junctions.

### Band variants

Boundary seeds are points whose k nearest neighbours (k = 16 by default)
include a different ground-truth label. Every point within distance r of a seed
is in the band. The band depends only on the ground truth, so every model is
scored on the same points.

- `seed_ignore=(0,)`: **organ-only band**, used in the paper. Soil is removed
  before the seeds are found, so only stem-leaf transitions produce seeds.
- `seed_ignore=()`: **all-boundary band**. Soil contact also produces seeds.
  On scans with a lot of soil this contact makes up most of the band, so the
  score then mostly reflects the separation of plant and ground.
- `seed_labels=ids` with `seed_ignore=(0,)`: **instance-aware band**. Seeds are
  found on the leaf-instance ids, so contact between two different leaves also
  counts. Load the ids with `load_pheno4d_txt(..., return_ids=True)`.

With the semantic labels alone, contact between two leaves never produces a
seed, since all leaves share one label.

## Reproducing the paper

1. **Data.** Download Pheno4D from <https://www.ipb.uni-bonn.de/data/pheno4d/>
   and unpack it so that `data/Pheno4D/Maize01/M01_0313_a.txt` exists.
2. **Check the labels** (optional):
   `python -c "from jiou import audit_directory; audit_directory('data/Pheno4D')"`
3. **Convert:** `python scripts/prepare_data.py data/Pheno4D data/pheno4d --corrected`
4. **Train** with [Pointcept](https://github.com/Pointcept/Pointcept). Copy
   `configs/pheno4d_dataset.py` to `pointcept/datasets/pheno4d.py`, register it
   in `pointcept/datasets/__init__.py`, copy the configs to
   `configs/pheno4d/`, then run `scripts/train.sh <config>` for five seeds.
5. **Test and export** full-resolution predictions: `scripts/test.sh <config>`
6. **Score:** `python scripts/band_counts.py --raw data/Pheno4D --pred predictions/maize --out results/band-counts`,
   then `python scripts/summarise.py results/band-counts`. For the plain
   per-scan table use `scripts/score.py`.
7. **Figures:** `python scripts/figures.py`

Steps 4 and 5 need a GPU and take several hours per run. To check the numbers
without training, use the files in `results/`:

| File | Content |
|---|---|
| `scores-*-corrected.csv` | per-scan scores for all 20 runs, corrected labels |
| `scores-*-raw.csv` | the same runs scored against the labels as published |
| `band-counts/` | per-scan confusion counts for every band variant (k = 8, 16, 32, all-boundary, instance-aware) |
| `label-audit.csv` | the label check for all 126 scans |

`scripts/write_configs.py` and `scripts/queue.sh` set up and run the second
batch of the paper (two further plant folds and a grid-size sweep) on several
GPUs.

## Label errors in Pheno4D

The label check found ten scans with interchanged labels: soil and stem in
Tomato02/T02_0325_a, stem and first leaf in all seven scans of Maize02, and the
stem carrying a leaf id in Maize03/M03_0321_a and M03_0324_a. Only the
leaf-collar labels are affected. Details, evidence and the effect on the scores
are in [`docs/label-errors.md`](docs/label-errors.md).

## Layout

```
jiou/        the metric, data loading, label corrections and label check
scripts/     data preparation, training, scoring, summaries, figures
configs/     Pointcept configs and dataset class used in the paper
results/     per-scan scores and band counts for all runs
docs/        the label errors
```

## Citing

Please cite the paper and this software (see `CITATION.cff`).

## License

MIT, see `LICENSE`.
