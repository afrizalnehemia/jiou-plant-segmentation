#!/usr/bin/env python3
"""Agregat tomat DENGAN scan anomali dikoreksi (bukan dikeluarkan).

Label 0<->1 pada Tomato02/T02_0325_a ditukar balik, lalu scan itu diskor ulang
dengan prediksi yang SAMA -- prediksi tidak perlu dihitung ulang karena hanya
ground truth-nya yang salah. Hasilnya menggantikan baris rusak di CSV, dan
agregat per model dihitung atas 22 dari 22 scan uji.
"""
import csv, glob, os, sys
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from jiou import load_pheno4d_txt as load_raw, distance_to_boundary, miou

RAW  = os.path.join(HERE, "..", "data", "Pheno4D", "Tomato02", "T02_0325_a.txt")
RUNS = os.path.join(HERE, "..", "predictions", "tomato")
CSV  = os.path.join(RUNS, "scores-tomato.csv")
RADII, SCORED = [1, 2, 5, 10], [1, 2]

xyz, gt_raw = load_raw(RAW)
gt = gt_raw.copy(); gt[gt_raw == 0] = 1; gt[gt_raw == 1] = 0      # koreksi

d_J, _ = distance_to_boundary(xyz, gt)
d_O, _ = distance_to_boundary(xyz, gt, seed_ignore=(0,))
mJ = {r: d_J <= r for r in RADII}
mO = {r: d_O <= r for r in RADII}

fixed = {}
for d in sorted(glob.glob(os.path.join(RUNS, "*tomato-seed*"))):
    run = os.path.basename(d)
    model, seed = run.rsplit("-seed", 1)
    pred = np.load(os.path.join(d, "Tomato02-T02_0325_a.npy")).astype(np.int64)
    row = {"mIoU_global": miou(gt, pred, SCORED)}
    for r in RADII:
        row[f"J_mIoU_r{r}"] = miou(gt[mJ[r]], pred[mJ[r]], SCORED)
        row[f"O_mIoU_r{r}"] = miou(gt[mO[r]], pred[mO[r]], SCORED)
        row[f"O_band_share_r{r}"] = float(mO[r].mean())
    fixed[(model, seed)] = row

print("scan terkoreksi -- pita organ r=5mm mencakup %.2f%% titik\n" % (mO[5].mean()*100))

rows = list(csv.DictReader(open(CSV)))
store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in rows:
    key = (r["model"], r["seed"])
    src = fixed[key] if (r["plant"], r["scan"]) == ("Tomato02", "T02_0325_a") else r
    for c in ["mIoU_global"] + [f"O_mIoU_r{t}" for t in RADII]:
        store[r["model"]][r["seed"]][c].append(float(src[c]))

cols = ["mIoU_global"] + [f"O_mIoU_r{t}" for t in RADII]
print(f"{'model':<32}{'metrik':<16}{'mean':>8}{'sd':>8}{'n_scan':>8}")
for model in sorted(store):
    for c in cols:
        per_seed = np.array([np.mean(store[model][s][c]) for s in sorted(store[model])])
        n = len(store[model][sorted(store[model])[0]][c])
        print(f"{model:<32}{c:<16}{per_seed.mean():>8.3f}{per_seed.std(ddof=1):>8.3f}{n:>8}")
    print()

# tulis CSV terkoreksi sebagai sumber tunggal untuk tabel dan gambar hilir
OUT = os.path.join(RUNS, "scores-tomato-corrected.csv")
with open(CSV) as f:
    allrows = list(csv.DictReader(f)); fields = allrows[0].keys()
for r in allrows:
    if (r["plant"], r["scan"]) == ("Tomato02", "T02_0325_a"):
        r.update({k: v for k, v in fixed[(r["model"], r["seed"])].items() if k in r})
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(fields)); w.writeheader(); w.writerows(allrows)
print("ditulis:", OUT, len(allrows), "baris")
