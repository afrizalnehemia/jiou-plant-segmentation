# Written by scripts/write_configs.py, do not edit by hand.
# rev-spunet-tomato-f2-g0p5-s0: fold 2 (test ['Tomato05', 'Tomato06']), grid 0.5 mm, seed 0,
# corrected labels, evaluated with model_last.
"""SparseUNet (MinkUNet-style) semantic segmentation on Pheno4D tomato (soil / stem / leaf).

Copy to: Pointcept/configs/pheno4d/semseg-spunet-pheno4d-tomato.py

COORDINATES ARE IN MILLIMETRES. Every spatial hyperparameter below is therefore
in mm, including grid_size. This matters more than usual here: grid_size sets how
much geometry survives voxelisation, and the junction band we score on is only a
few millimetres wide. A coarse grid quietly deletes the very region the
evaluation is about, so grid_size is treated as a reported experimental factor,
not a detail.
"""
_base_ = ["../_base_/default_runtime.py"]

# ---- split: edit test_plants per fold (see scripts/splits.py) ----------------
train_plants = ['Tomato01', 'Tomato02', 'Tomato03', 'Tomato04', 'Tomato07']
test_plants  = ['Tomato05', 'Tomato06']
# ----------------------------------------------------------------- TOMATO NOTES
# Fold 0 split for tomato, following splits.py: 5 plants to train on, 2 to test.
#
# Two things differ from maize and are worth knowing before reading the results:
#
# 1. The label scheme is the SAME (0 soil, 1 stem, 2 leaf). Tomato only ever has
#    one collar-style label column. num_classes and names are unchanged, so both
#    species are scored against identical class definitions.
#
# 2. Class composition shifts hard over development. On T07, soil falls from 93.3%
#    (0305) to 8.5% (0325) while leaf climbs from 5.8% to 84.0%, with leaf count
#    going 3 -> 26. Tomato scans are also bigger: up to 4.2M points against 1.7M
#    for maize. Watch VRAM during the smoke test before launching five seeds --
#    batch 4 will not necessarily fit the way it does on maize.


data_root = "data/pheno4d-fixed"
grid_size = 0.5

# GRID NOTE
# Point spacing on the plant itself in Pheno4D is 0.03-0.08 mm (measured, not
# assumed). A 2 mm grid is 25-60x coarser than the data, and the junction band
# being scored is only a few millimetres wide, so the model gets marked down
# partly for being blinded by preprocessing rather than for its architecture.
# The compute cost is low (0.5 mm -> ~80k voxels per scan, 0.25 mm -> ~325k), so a
# fine grid is affordable. grid_size is treated as a reported experimental factor:
# run 2.0 / 1.0 / 0.5 / 0.25 mm and report the curve.

num_classes = 3          # 0 soil, 1 stem, 2 leaf
names = ["soil", "stem", "leaf"]

batch_size = 2          # TOMATO: not 4 as on maize, see BATCH NOTE
# BATCH NOTE
# Tomato scans reach 4.2M points (maize: 1.7M). PTv3, which peaks at 12.1 GB on
# maize, asks for 21.6 GB here and OOMs a 24 GB card. Batch 2 is used for BOTH
# tomato models, not just PTv3: the comparison the paper claims is between models
# WITHIN a species, so the two have to share a batch size. Maize stays at 4.
# Absolute values are not compared across species anyway, and the difference is
# reported.
num_worker = 8
mix_prob = 0.0           # off: mixing two plants makes junction labels meaningless
empty_cache = False
enable_amp = True

model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=96,
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=3,
        num_classes=0,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

epoch = 200
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(type="OneCycleLR", max_lr=[0.002, 0.0002],
                 pct_start=0.05, anneal_strategy="cos",
                 div_factor=10.0, final_div_factor=1000.0)
param_dicts = [dict(keyword="block", lr=0.0002)]

dataset_type = "Pheno4DDataset"
data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=names,
    train=dict(
        type=dataset_type, split=train_plants, data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.05, clip=0.2),   # mm
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv",
                 mode="train", return_grid_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"),
                 feat_keys=("coord",)),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type, split=test_plants, data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv",
                 mode="train", return_grid_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"),
                 feat_keys=("coord",)),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type, split=test_plants, data_root=data_root,
        transform=[dict(type="CenterShift", apply_z=True)],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample", grid_size=grid_size, hash_type="fnv",
                          mode="test", return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "grid_coord", "index"),
                     feat_keys=("coord",)),
            ],
            aug_transform=[[dict(type="RandomScale", scale=[1, 1])]],
        ),
    ),
)

seed = 0
