"""J-IoU: junction-restricted evaluation for 3D plant organ segmentation.

Quick start::

    from jiou import evaluate, load_pheno4d_txt

    xyz, gt = load_pheno4d_txt("Maize01/M01_0325_a.txt", corrected=True)
    pred = ...                      # per-point labels from your model, same order
    res = evaluate(xyz, gt, pred, radii=[1, 2, 5, 10], seed_ignore=(0,))
    print(res["mIoU_global"], res["bands"][0]["J_mIoU"])

seed_ignore=(0,) gives the organ-only band used in the paper, in which soil
contact is not a boundary. See README.md.
"""
from .metrics import (boundary_seeds, distance_to_boundary, junction_band,
                      iou_per_class, miou, evaluate)
from .io import read_raw, load_pheno4d_txt, load_predictions
from .labels import CORRECTIONS, apply_corrections
from .audit import robust_extent, audit_scan, audit_directory

__version__ = "1.1.0"
__all__ = ["boundary_seeds", "distance_to_boundary", "junction_band",
           "iou_per_class", "miou", "evaluate", "read_raw", "load_pheno4d_txt",
           "load_predictions", "CORRECTIONS", "apply_corrections",
           "robust_extent", "audit_scan", "audit_directory"]
