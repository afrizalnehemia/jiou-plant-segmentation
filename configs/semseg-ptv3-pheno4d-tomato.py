"""PTv3 semantic segmentation on Pheno4D tomato (soil / stem / leaf).

Copy to: Pointcept/configs/pheno4d/semseg-ptv3-pheno4d-tomato.py

COORDINATES ARE IN MILLIMETRES. Every spatial hyperparameter below is therefore
in mm, including grid_size. This matters more than usual here: grid_size sets how
much geometry survives voxelisation, and the junction band we score on is only a
few millimetres wide. A coarse grid quietly deletes the very region the
evaluation is about, so grid_size is treated as a reported experimental factor,
not a detail.
"""
_base_ = ["../_base_/default_runtime.py"]

# ---- split: edit test_plants per fold (see 01-DATA/splits.py) ----------------
train_plants = ["Tomato03", "Tomato04", "Tomato05", "Tomato06", "Tomato07"]
test_plants  = ["Tomato01", "Tomato02"]
# ---------------------------------------------------------------- CATATAN TOMAT
# Split fold 0 untuk tomat, mengikuti splits.py: 5 tanaman latih, 2 tanaman uji.
#
# Dua hal yang berbeda dari jagung dan perlu disadari saat membaca hasilnya:
#
# 1. Skema labelnya SAMA (0 soil, 1 stem, 2 leaf) -- tomat memang hanya punya satu
#    kolom label bergaya collar. Jadi num_classes dan names tidak berubah, dan
#    kedua spesies dinilai dengan definisi kelas yang identik.
#
# 2. Komposisi kelasnya bergeser ekstrem sepanjang pertumbuhan. Pada T07, soil
#    turun dari 93,3% (0305) ke 8,5% (0325) sementara leaf naik dari 5,8% ke 84,0%,
#    dengan jumlah daun 3 -> 26. Scan tomat juga lebih besar: sampai 4,2 juta titik
#    versus 1,7 juta pada jagung. Awasi VRAM pada smoke test sebelum melepas lima
#    seed -- batch 4 belum tentu muat seperti pada jagung.


data_root = "data/pheno4d"
grid_size = 0.5          # mm -- lihat CATATAN GRID di bawah

# CATATAN GRID
# Jarak antar titik asli di area tanaman Pheno4D adalah 0,03-0,08 mm (diukur, bukan
# diperkirakan). Grid 2 mm berarti 25-60x lebih kasar daripada datanya, dan pita
# persimpangan yang dinilai lebarnya hanya beberapa milimeter -- model jadi dinilai
# gagal sebagian karena dibutakan preprocessing, bukan karena arsitekturnya.
# Beban komputasinya ringan (0,5 mm -> ~80 ribu voxel per scan, 0,25 mm -> ~325 ribu),
# jadi grid halus terjangkau. grid_size diperlakukan sebagai faktor eksperimen yang
# dilaporkan: jalankan 2,0 / 1,0 / 0,5 / 0,25 mm dan laporkan kurvanya.

num_classes = 3          # 0 soil, 1 stem, 2 leaf
names = ["soil", "stem", "leaf"]

batch_size = 2          # TOMAT: bukan 4 seperti jagung -- lihat CATATAN BATCH
# CATATAN BATCH
# Scan tomat mencapai 4,2 juta titik (jagung: 1,7 juta). PTv3 yang memuncak di
# 12,1 GB pada jagung meminta 21,6 GB di sini dan OOM pada kartu 24 GB.
# batch 2 dipakai untuk KEDUA model tomat, bukan hanya PTv3: perbandingan yang
# menjadi klaim makalah adalah antar model DI DALAM satu spesies, jadi keduanya
# harus berbagi batch size. Jagung tetap di 4; angka absolut lintas spesies
# memang tidak dibandingkan, dan perbedaan ini dilaporkan.
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
