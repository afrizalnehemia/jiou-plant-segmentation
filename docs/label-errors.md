# Label errors in Pheno4D

Two geometric checks on the 126 annotated Pheno4D scans found interchanged
labels in ten files. The corrections are listed in `jiou/labels.py` and are
applied by `prepare_data.py --corrected` and `load_pheno4d_txt(..., corrected=True)`.
All results in the paper and in `results/*-corrected.csv` use them.

To our knowledge these errors had not been reported before. We reported them
to the dataset maintainers at the University of Bonn, who checked the ten scans
and confirmed on 24 September 2026 that the labelling is not consistent in
these cases. Until a corrected release of the dataset is available, use the
corrections in this repository.

## How they were found

Both checks use only the labelled coordinates. The vertical extent of a label
is the spread between the 5th and 95th percentile of z, which ignores the few
noise points that usually sit below the soil.

1. **Soil vs stem.** Soil should be a thin sheet at the bottom of the scene and
   the stem an upright structure. A scan is flagged when soil is taller than
   the stem.
2. **Stem vs leaf ids.** In the raw leaf-collar labels the stem (id 1) runs from
   the base to the youngest leaves, so it should be taller than any single leaf.
   A scan is flagged when some leaf id is taller than id 1.

```python
from jiou import audit_directory
audit_directory("data/Pheno4D")
```

`results/label-audit.csv` lists the stem extent and the tallest leaf for all
126 scans.

## The ten scans

| Plant | Scan(s) | Problem | Correction |
|---|---|---|---|
| Tomato02 | T02_0325_a | soil and stem interchanged | swap 0 and 1 |
| Maize02 | all seven annotated scans | stem has id 2, first leaf has id 1 | swap 1 and 2 |
| Maize03 | M03_0321_a | stem has id 3 | swap 1 and 3 |
| Maize03 | M03_0324_a | stem has id 4 | swap 1 and 4 |

**Tomato02, T02_0325_a.** The points labelled stem form a flat sheet with a
vertical extent of 2.2 mm over an area of 129 x 131 mm. The points labelled
soil form an upright structure 71.6 mm tall. With 0 and 1 swapped, geometry
and point counts match the scan of the same plant one day earlier.

**Maize02.** In every scan id 1 is a small organ near the base, while id 2
spans the plant:

| Scan | id 1 extent (mm) | id 2 extent (mm) |
|---|---|---|
| M02_0313_a | 23.8 | 77.8 |
| M02_0315_a | 19.7 | 96.7 |
| M02_0317_a | 19.0 | 150.6 |
| M02_0319_a | 15.7 | 218.4 |
| M02_0321_a | 15.1 | 274.9 |
| M02_0324_a | 6.9 | 276.5 |
| M02_0325_a | 10.0 | 293.8 |

In the other maize plants it is the other way round: id 1 is the tall stem and
id 2 a small first leaf.

**Maize03.** In M03_0321_a id 1 has an extent of 14.3 mm and id 3 of 223.3 mm.
In M03_0324_a id 1 has 13.1 mm and id 4 has 338.8 mm. In the scans of the same
plant before and after (M03_0319_a and M03_0325_a), id 1 is the stem.

**Leaf-tip labels.** The fifth column of the maize files (leaf-tip scheme, no
stem class) shows no such pattern, so work based on the leaf-tip labels is not
affected. Note that some redistributions of Pheno4D, for example the copy on
Hugging Face, use the leaf-collar labels as their default.

## Effect on the scores

The predictions are the same in both columns, only the reference labels
change. Values are from the fold-0 runs of the paper (5 seeds per
architecture, labels corrected during training).

| | published labels | corrected |
|---|---|---|
| T02_0325_a, global mIoU (mean of 10 runs) | 0.496 | 0.950 |
| T02_0325_a, J-IoU at 5 mm | undefined (no seeds) | 0.844 |
| M02_0313_a, SparseUNet seed 0, global mIoU | 0.001 | 0.996 |
| Maize02, 7 scans, SparseUNet, global mIoU | 0.115 | 0.818 |
| Maize02, 7 scans, PTv3, global mIoU | 0.122 | 0.818 |

The per-scan counts behind this table are in
`results/revision/band-counts-published-*-f0/` (published labels) and
`results/revision/band-counts-*-f0/` (corrected labels).

## Reproducing with the published or the corrected labels

The corrections are a fixed list of id swaps, so every step can be run either
way.

| Step | Published labels | Corrected labels |
|---|---|---|
| Load one scan | `load_pheno4d_txt(path)` | `load_pheno4d_txt(path, corrected=True)` |
| Prepare training data | `python scripts/prepare_data.py data/Pheno4D data/pheno4d` | `python scripts/prepare_data.py data/Pheno4D data/pheno4d-fixed --corrected` |
| Score predictions | `python scripts/band_counts.py ... --published-labels` | `python scripts/band_counts.py ...` (default) |

`prepare_data.py --corrected` records the swaps it applied in each scan's
`meta.json`, so a prepared dataset shows which version it is. Running the
label check on the raw files, `audit_directory("data/Pheno4D")`, lists the
same ten scans, and `results/label-audit-extents.csv` gives the vertical
extents behind both tests for all 126 scans.

All runs of the paper (release 1.2.0) were trained and tested on the
corrected labels. The first batch of runs (release 1.0.0, kept in
`results/batch1/`) was trained with the two Maize03 scans as published.
