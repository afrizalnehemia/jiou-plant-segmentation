#!/usr/bin/env python3
"""All numbers in the paper, from the 53 training runs.

Reads results/revision/band-counts-<species>-f<fold>/ and writes
results/revision/paper-numbers.txt and paper-stats.json.

Runs: fold 0 has 5 seeds, folds 1-2 have 3 seeds. The cross-validated (CV)
score pools the test scans of the three folds (plants 1-6, each tested once)
for seeds 0-2, which all folds share.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarise import load, welch

SK = os.path.join(HERE, "..", "results", "revision")
RADII = [1, 2, 5, 10]
SPECIES = ["maize", "tomato"]
ARCH = {"spunet": "SparseUNet", "ptv3": "PTv3"}
out_lines, J = [], {}

def p(*a):
    s = " ".join(str(x) for x in a); print(s); out_lines.append(s)

def ms(v): return f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"

frames = []
for sp in SPECIES:
    for f in (0, 1, 2):
        d = load(os.path.join(SK, f"band-counts-{sp}-f{f}"))
        d["fold"] = f
        d["base"] = d.model.str.replace(r"-f\d$", "", regex=True)   # spunet-tomato, spunet-g1-tomato ...
        d["sp"] = sp
        frames.append(d)
df = pd.concat(frames, ignore_index=True)
df["seed"] = df.seed.astype(int)
main = df[df.base.isin([f"{a}-{s}" for a in ARCH for s in SPECIES])]

def per_seed(d, mask, col="miou"):
    x = d[d["mask"] == mask]
    return x.groupby("seed")[col].mean().sort_index()

def row(d):
    r = {"n": int(d.seed.nunique()), "global": per_seed(d, "global").tolist(),
         "acc_global": per_seed(d, "global", "acc").tolist()}
    for rr in RADII:
        r[f"r{rr}"] = per_seed(d, f"O16_r{rr}").tolist()
        r[f"acc_r{rr}"] = per_seed(d, f"O16_r{rr}", "acc").tolist()
        for c, nm in ((1, "stem"), (2, "leaf")):
            r[f"{nm}_r{rr}"] = per_seed(d, f"O16_r{rr}", f"iou{c}").tolist()
    for c, nm in ((1, "stem"), (2, "leaf")):
        r[f"{nm}_global"] = per_seed(d, "global", f"iou{c}").tolist()
    for v in ("O8", "O32", "A16", "I16"):
        r[f"{v}_r1"] = per_seed(d, f"{v}_r1").tolist()
    return r

# ---------------------------------------------------------------- per fold
p("===== 1. Per fold (organ-only band, k=16), mean ± sd over seeds")
J["fold"] = {}
for sp in SPECIES:
    for f in (0, 1, 2):
        for a in ARCH:
            d = main[(main.sp == sp) & (main.fold == f) & (main.base == f"{a}-{sp}")]
            r = row(d); J["fold"][f"{sp}|{f}|{a}"] = r
            gap = np.mean(r["global"]) - np.mean(r["r1"])
            p(f"{sp:6s} f{f} {ARCH[a]:10s} n={r['n']} global {ms(r['global'])}  r1 {ms(r['r1'])}  r2 {ms(r['r2'])}"
              f"  r5 {ms(r['r5'])}  r10 {ms(r['r10'])}  gap1 {gap:.3f}  gap/sd {gap/np.std(r['r1'], ddof=1):.1f}"
              f"  acc {np.mean(r['acc_global']):.3f} acc1 {np.mean(r['acc_r1']):.3f}")

# ---------------------------------------------------------------- CV pooled
p("\n===== 2. Cross-validated, test scans of folds 0-2 pooled, seeds 0-2")
J["cv"] = {}
cv = main[main.seed <= 2]
for sp in SPECIES:
    for a in ARCH:
        d = cv[(cv.sp == sp) & (cv.base == f"{a}-{sp}")]
        r = row(d); J["cv"][f"{sp}|{a}"] = r
        nscan = d[d["mask"] == "global"].groupby("seed").size().iloc[0]
        gap = np.mean(r["global"]) - np.mean(r["r1"])
        p(f"{sp:6s} {ARCH[a]:10s} scans={nscan} global {ms(r['global'])}  r1 {ms(r['r1'])}  r2 {ms(r['r2'])}"
          f"  r5 {ms(r['r5'])}  r10 {ms(r['r10'])}  gap1 {gap:.3f} gap/sd {gap/np.std(r['r1'], ddof=1):.1f}")
        p(f"      acc global {ms(r['acc_global'])}  " + "  ".join(f"acc r{x} {ms(r[f'acc_r{x}'])}" for x in RADII))
        for nm in ("stem", "leaf"):
            p(f"      {nm} global {ms(r[f'{nm}_global'])}  " + "  ".join(f"r{x} {ms(r[f'{nm}_r{x}'])}" for x in RADII))
        p("      sens r1  O8 " + ms(r["O8_r1"]) + "  O16 " + ms(r["r1"]) + "  O32 " + ms(r["O32_r1"])
          + "  A16 " + ms(r["A16_r1"]) + "  I16 " + ms(r["I16_r1"]))
    # sensitivity: gap range and architecture p under every variant
    ps = []
    for v in ("O8_r1", "O16_r1", "O32_r1", "A16_r1", "I16_r1"):
        a_ = per_seed(cv[(cv.sp == sp) & (cv.base == f"spunet-{sp}")], v).to_numpy()
        b_ = per_seed(cv[(cv.sp == sp) & (cv.base == f"ptv3-{sp}")], v).to_numpy()
        ps.append(welch(a_, b_)[4])
    p(f"      sensitivity Welch p (O8,O16,O32,A16,I16) at r1: " + ", ".join(f"{x:.2f}" for x in ps))

# band share per class (fold-pooled, one run: band is annotation only)
p("\n===== 3. Band share (organ-only k=16), mean over all test scans of the 3 folds")
J["share"] = {}
one = main[(main.seed == 0) & (main.base.str.startswith("spunet"))]
for sp in SPECIES:
    o = one[one.sp == sp]
    for rr in RADII:
        m = o[o["mask"] == f"O16_r{rr}"]
        share = (m.npts / m.n).mean()
        stem = (sum(m[f"c1{q}"] for q in range(3)) / m.n_stem).mean()
        leaf = (sum(m[f"c2{q}"] for q in range(3)) / m.n_leaf.where(m.n_leaf > 0)).mean()
        J["share"][f"{sp}|r{rr}"] = [share, stem, leaf]
        p(f"{sp:6s} r{rr}: scan {100*share:.2f}%  stem {100*stem:.1f}%  leaf {100*leaf:.1f}%  median pts {m.npts.median():.0f}")
    g = o[o["mask"] == "global"]
    p(f"{sp:6s} nn spacing on plant median {g.nn_plant.median():.3f} mm")

# ---------------------------------------------------------------- architecture
p("\n===== 4. PTv3 minus SparseUNet, Welch per fold and CV, random-effects pooled over folds")
J["arch"] = {}
def dl_pool(diffs, vars_):
    w = 1 / vars_; fe = np.sum(w * diffs) / np.sum(w)
    q = np.sum(w * (diffs - fe) ** 2); k = len(diffs)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c)
    ws = 1 / (vars_ + tau2); est = np.sum(ws * diffs) / np.sum(ws); se = np.sqrt(1 / np.sum(ws))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    z = est / se
    return est, est - 1.96 * se, est + 1.96 * se, 2 * stats.norm.sf(abs(z)), i2, q, stats.chi2.sf(q, k - 1)
for sp in SPECIES:
    for m in ["global", "O16_r1", "O16_r2", "O16_r5", "O16_r10"]:
        diffs, vs = [], []
        for f in (0, 1, 2):
            a_ = per_seed(main[(main.sp == sp) & (main.fold == f) & (main.base == f"spunet-{sp}")], m).to_numpy()
            b_ = per_seed(main[(main.sp == sp) & (main.fold == f) & (main.base == f"ptv3-{sp}")], m).to_numpy()
            diff, lo, hi, d, pv = welch(a_, b_)
            J["arch"][f"{sp}|f{f}|{m}"] = [diff, lo, hi, d, pv]
            diffs.append(diff); vs.append(a_.var(ddof=1) / len(a_) + b_.var(ddof=1) / len(b_))
            p(f"{sp:6s} f{f} {m:8s} {diff:+.4f} [{lo:+.4f}, {hi:+.4f}] d {d:+.2f} p {pv:.3f}")
        a_ = per_seed(cv[(cv.sp == sp) & (cv.base == f"spunet-{sp}")], m).to_numpy()
        b_ = per_seed(cv[(cv.sp == sp) & (cv.base == f"ptv3-{sp}")], m).to_numpy()
        diff, lo, hi, d, pv = welch(a_, b_)
        J["arch"][f"{sp}|cv|{m}"] = [diff, lo, hi, d, pv]
        p(f"{sp:6s} CV {m:8s} {diff:+.4f} [{lo:+.4f}, {hi:+.4f}] d {d:+.2f} p {pv:.3f}")
        est, lo, hi, pv, i2, q, pq = dl_pool(np.array(diffs), np.array(vs))
        J["arch"][f"{sp}|re|{m}"] = [est, lo, hi, pv, i2, q, pq]
        p(f"{sp:6s} RE {m:8s} {est:+.4f} [{lo:+.4f}, {hi:+.4f}] p {pv:.3f}  I2 {100*i2:.0f}%  Q {q:.2f} pQ {pq:.3f}")

# ---------------------------------------------------------------- grid sweep
p("\n===== 5. Grid sweep, tomato SparseUNet fold 0, seeds 0-2")
J["grid"] = {}
grids = [("g0p25", 0.25), ("g0p5", 0.5), ("g1", 1.0), ("g2", 2.0)]
t0 = df[(df.sp == "tomato") & (df.fold == 0) & (df.seed <= 2)]
rows = []
for tag, g in grids:
    base = "spunet-tomato" if tag == "g0p5" else f"spunet-{tag}-tomato"
    d = t0[t0.base == base]
    r = row(d); J["grid"][str(g)] = r
    gap = np.mean(r["global"]) - np.mean(r["r1"])
    p(f"grid {g:4} mm n={r['n']} global {ms(r['global'])}  r1 {ms(r['r1'])}  r2 {ms(r['r2'])}  r5 {ms(r['r5'])}  r10 {ms(r['r10'])}"
      f"  stem g {ms(r['stem_global'])} stem r1 {ms(r['stem_r1'])}  leaf r1 {ms(r['leaf_r1'])}  gap1 {gap:.3f}  acc1 {ms(r['acc_r1'])}")
    for s_, (gv, jv) in enumerate(zip(r["global"], r["r1"])):
        rows.append({"g": g, "global": gv, "r1": jv, "r2": r["r2"][s_], "r5": r["r5"][s_], "stem_r1": r["stem_r1"][s_],
                     "stem_global": r["stem_global"][s_]})
gr = pd.DataFrame(rows); x = np.log2(gr.g)
for col in ["global", "r1", "r2", "r5", "stem_global", "stem_r1"]:
    res = stats.linregress(x, gr[col])
    J["grid"][f"slope_{col}"] = [res.slope, res.stderr, res.pvalue, res.rvalue ** 2]
    p(f"slope per doubling of grid, {col:12s}: {res.slope:+.4f} ± {res.stderr:.4f}  p {res.pvalue:.2g}  R2 {res.rvalue**2:.2f}")
lo_, hi_ = J["grid"]["0.25"], J["grid"]["2.0"]
for col in ["global", "r1", "stem_r1"]:
    diff, l, h, d, pv = welch(np.array(lo_[col]), np.array(hi_[col]))
    p(f"2 mm minus 0.25 mm {col:8s}: {diff:+.4f} [{l:+.4f}, {h:+.4f}] d {d:+.2f} p {pv:.4f}")

# ---------------------------------------------------------------- per-scan correlation
p("\n===== 6. Per-scan relation between global mIoU and J-IoU (r1), scans of all folds, mean over seeds 0-2")
J["corr"] = {}
for sp in SPECIES:
    for a in ARCH:
        d = cv[(cv.sp == sp) & (cv.base == f"{a}-{sp}")]
        g = d[d["mask"] == "global"].groupby(["plant", "scan"]).miou.mean()
        j = d[d["mask"] == "O16_r1"].groupby(["plant", "scan"]).miou.mean()
        both = pd.concat([g, j], axis=1, keys=["g", "j"]).dropna()
        rho, pr = stats.spearmanr(both.g, both.j)
        J["corr"][f"{sp}|{a}"] = {"rho": rho, "p": pr, "g": both.g.tolist(), "j": both.j.tolist(),
                                  "plant": [x[0] for x in both.index]}
        p(f"{sp:6s} {ARCH[a]:10s} scans {len(both)}  Spearman rho {rho:+.2f}  p {pr:.3g}"
          f"  global range {both.g.min():.3f}-{both.g.max():.3f}  J1 range {both.j.min():.3f}-{both.j.max():.3f}")

# ---------------------------------------------------------------- tomato fold 1 soil-as-leaf scans
p("\n===== 7. Scans where soil was predicted as leaf (global acc < 0.9), mean over the runs of that fold")
g = main[main["mask"] == "global"].copy()
g["soil_as_leaf"] = g.c02 / g.n_soil
bad = g.groupby(["sp", "fold", "plant", "scan"]).agg(acc=("acc", "mean"), sal=("soil_as_leaf", "mean"),
                                                     leaf=("iou2", "mean"), miou=("miou", "mean")).reset_index()
for _, b in bad[bad.acc < 0.9].iterrows():
    j1 = main[(main["mask"] == "O16_r1") & (main.scan == b.scan)].miou.mean()
    p(f"{b.sp} f{b.fold} {b.scan}: acc {b.acc:.3f}  soil predicted leaf {100*b.sal:.1f}%  leaf IoU {b.leaf:.3f}  mIoU {b.miou:.3f}  J1 {j1:.3f}")

# ---------------------------------------------------------------- published labels
p("\n===== 8. Fold-0 runs scored against the labels as published")
for sp, folder, sel in (("tomato", "band-counts-published-tomato-f0", "T02_0325_a"),
                        ("maize", "band-counts-published-maize-f0", None)):
    pub = load(os.path.join(SK, folder)); cor = main[(main.sp == sp) & (main.fold == 0)]
    for a in ARCH:
        pb = pub[(pub.model == f"{a}-{sp}-f0") & (pub["mask"] == "global")]
        cr = cor[(cor.base == f"{a}-{sp}") & (cor["mask"] == "global") & cor.scan.isin(pb.scan.unique())]
        p(f"{sp} {ARCH[a]:10s} scans {pb.scan.nunique()}  global published {pb.groupby('seed').miou.mean().mean():.3f}"
          f" (per seed {', '.join(f'{v:.3f}' for v in pb.groupby('seed').miou.mean())})"
          f"  corrected {cr.groupby('seed').miou.mean().mean():.3f}")
        if sp == "tomato":
            p(f"      stem IoU published {pb.iou1.mean():.4f}  leaf {pb.iou2.mean():.3f}; seeds O16 published {pub[(pub.model==f'{a}-{sp}-f0')&(pub['mask']=='O16_r5')].nseed.iloc[0]}"
              f"  corrected r5 J {cor[(cor.base==f'{a}-{sp}')&(cor.scan==sel)&(cor['mask']=='O16_r5')].miou.mean():.3f}"
              f" nseed {cor[(cor.base==f'{a}-{sp}')&(cor.scan==sel)&(cor['mask']=='O16_r5')].nseed.iloc[0]}"
              f"  corrected global {cor[(cor.base==f'{a}-{sp}')&(cor.scan==sel)&(cor['mask']=='global')].miou.mean():.3f}")
        else:
            m0 = pb[(pb.scan == "M02_0313_a") & (pb.seed == 0)].miou.iloc[0]
            c0 = cr[(cr.scan == "M02_0313_a") & (cr.seed == 0)].miou.iloc[0]
            p(f"      M02_0313_a seed 0: published {m0:.3f}  corrected {c0:.3f}")

open(os.path.join(SK, "paper-numbers.txt"), "w").write("\n".join(out_lines) + "\n")
json.dump(J, open(os.path.join(SK, "paper-stats.json"), "w"), indent=1, default=float)
