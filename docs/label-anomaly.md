# A label anomaly in Pheno4D: `Tomato02/T02_0325_a`

**Summary.** In one scan out of the 126 annotated scans in Pheno4D, the soil
(class 0) and stem (class 1) labels are interchanged. This document records the
evidence, the effect on scores, and how the scan is handled in our results.

## How it surfaced

Scored against the published labels, this scan produced a global mIoU of
**0.496** in all ten of our independent training runs — two architectures ×
five random seeds — agreeing to three decimal places. That uniformity was the
first signal: a genuine model failure would vary across seeds and across
architectures.

The organ-only J-IoU was simultaneously **undefined** at every band radius,
with a band point share of exactly zero.

## The evidence

Robust vertical extent per class (5th–95th percentile of *z*), compared with
the scan of the same plant taken one day earlier:

| Scan | soil (class 0) | stem (class 1) |
|---|---|---|
| `T02_0324_a` (normal) | 16.1 mm thick, 128 × 135 mm — a plane | 95.8 mm tall — upright |
| **`T02_0325_a`** | **71.6 mm tall — upright** | **2.2 mm thick, 129 × 131 mm — a plane** |

Point counts match the interchange as well: the 292,119 points labelled stem in
`T02_0325_a` correspond in both count and geometry to the soil of the adjacent
scan, and the 217,357 labelled soil to its stem. The leaf class is unaffected.

## Why the score was exactly 0.496

Global mIoU here averages over the stem and leaf classes (soil is excluded).
Because the points labelled *stem* are in fact soil, the models predict them as
soil — anatomically correct, but scored as wrong. That gives a stem IoU of
0.0001 against a leaf IoU of 0.989, and a mean of 0.495. The arithmetic
reproduces the observed value exactly, which is what confirmed the diagnosis.

## Why J-IoU was undefined

The organ-only band excludes soil from seeding. With the labels swapped, the
points excluded are the *real stem*, leaving the genuine soil plane and the
leaves as the two remaining classes. Those two never touch — the nearest
approach is 3.43 mm, and the densely sampled soil plane means a *k* = 16
neighbourhood never crosses between them. No seed is found, the band is empty,
and J-IoU is undefined rather than merely low.

## Effect of correcting it

Swapping the two labels back and rescoring the **unchanged** predictions —
nothing is retrained, only the ground truth is repaired:

| | published labels | corrected labels |
|---|---|---|
| global mIoU (mean of 10 runs) | 0.496 | **0.950** |
| J-IoU at r = 5 mm | undefined (0 seeds) | **0.844** (3353 seeds) |

The junction gap that remains after correction, 0.106, is in line with the
other tomato scans.

## How we handle it

Our reported tomato results use the **corrected** labels and retain the scan,
giving 22 of 22 test scans. Excluding it instead shifts every tomato cell by
less than 0.005, so no conclusion depends on the choice. The uncorrected
values should not be used under either convention: they measure an annotation
error, not model performance.

`results/scores-tomato-raw.csv` and `results/scores-tomato-corrected.csv`
contain both versions so the difference can be inspected directly.

## Reproducing the audit

```python
from jiou import audit_directory
audit_directory("data/Pheno4D")      # flags T02_0325_a, and nothing else
```

The test uses only the labelled coordinates. Note that it measures thickness
with a 5th–95th percentile spread rather than the full range: stray noise
points below the soil plane make a raw range report a normal soil plane as tens
of millimetres thick, which produced a false positive on `T03_0307_a` in an
earlier version of the check.
