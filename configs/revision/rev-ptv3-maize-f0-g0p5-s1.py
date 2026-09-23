# Written by scripts/write_configs.py, do not edit by hand.
# rev-ptv3-maize-f0-g0p5-s1: fold 0 (test ['Maize01', 'Maize02']), grid 0.5 mm, seed 1,
# corrected labels, evaluated with model_last.
"""PTv3 semantic segmentation on Pheno4D maize (soil / stem / leaf).

Copy to: Pointcept/configs/pheno4d/semseg-ptv3-pheno4d-maize.py

COORDINATES ARE IN MILLIMETRES. Every spatial hyperparameter below is therefore
in mm, including grid_size. This matters more than usual here: grid_size sets how
much geometry survives voxelisation, and the junction band we score on is only a
few millimetres wide. A coarse grid quietly deletes the very region the
evaluation is about, so grid_size is treated as a reported experimental factor,
not a detail.
"""
_base_ = ["../_base_/default_runtime.py"]

# ---- split: edit test_plants per fold (see scripts/splits.py) ----------------
train_plants = ['Maize03', 'Maize04', 'Maize05', 'Maize06', 'Maize07']
test_plants  = ['Maize01', 'Maize02']

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

batch_size = 4
num_worker = 8
mix_prob = 0.0           # off: mixing two plants makes junction labels meaningless
empty_cache = False
enable_amp = True

model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,                       # xyz only, Pheno4D carries no colour
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4, qkv_bias=True, drop_path=0.3,
        shuffle_orders=True, pre_norm=True,
        enable_rpe=False, enable_flash=True,
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

seed = 1
