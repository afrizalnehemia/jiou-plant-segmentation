"""J-IoU: junction-restricted evaluation for 3D plant organ segmentation.

Quick start::

    from jiou import evaluate, load_pheno4d_txt

    xyz, gt = load_pheno4d_txt("Maize01/M01_0325_a.txt")
    pred = ...                      # your model's per-point labels, same order
    res = evaluate(xyz, gt, pred, radii=[1, 2, 5, 10], seed_ignore=(0,))
    print(res["mIoU_global"], res["bands"][0]["J_mIoU"])

`seed_ignore=(0,)` restricts the band to organ-organ junctions by refusing to
treat the soil-plant contact as a boundary; omit it to seed on every label
transition. See README.md for what that distinction changes.
"""
from .metrics import (boundary_seeds, distance_to_boundary, junction_band,
                      iou_per_class, miou, evaluate)
from .io import load_pheno4d_txt, load_predictions
from .audit import robust_thickness, audit_scan, audit_directory

__version__ = "1.0.0"
__all__ = ["boundary_seeds", "distance_to_boundary", "junction_band",
           "iou_per_class", "miou", "evaluate", "load_pheno4d_txt",
           "load_predictions", "robust_thickness", "audit_scan",
           "audit_directory"]
