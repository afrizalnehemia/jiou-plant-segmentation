#!/usr/bin/env python3
"""Regenerate the figures of the paper.

  fig1  construction of the junction band on a maize leaf collar
  fig2  J-IoU against band radius, with the global mIoU for reference
  fig3  misclassified points and the 5 mm band in six test scans
  fig4  per-seed scores of the two architectures
  fig5  the interchanged soil and stem labels in Tomato02/T02_0325_a

The two model colours (#0072B2, #D55E00) remain distinguishable under
simulated protanopia (delta-E 21.9), and the series also differ in marker and
line style so the figures still read in greyscale.

Expects the raw data in data/Pheno4D/, full-resolution predictions in
predictions/<species>/<run>/<Plant>-<scan>.npy, and the per-scan scores in
results/. Output goes to figures/.
"""
import argparse, csv, os, sys
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import ttest_ind
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from jiou import load_pheno4d_txt, distance_to_boundary

RAW = os.path.join(HERE, "..", "data", "Pheno4D")
PRED_M = os.path.join(HERE, "..", "predictions", "maize")
PRED_T = os.path.join(HERE, "..", "predictions", "tomato")
RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "figures")


def load_raw(path):
    """Labels exactly as published (used where the published labels are the point)."""
    return load_pheno4d_txt(path, corrected=False)


CLS_COL = {0: "#c9b896", 1: "#8a5a2b", 2: "#3f9142"}
CLS_EN  = {0: "soil", 1: "stem", 2: "leaf"}
MODEL_COL = {"spunet": "#0072B2", "ptv3": "#D55E00"}
MODEL_EN  = {"spunet": "SparseUNet", "ptv3": "PTv3"}
MODEL_MK  = {"spunet": "o", "ptv3": "s"}
MODEL_LS  = {"spunet": "-", "ptv3": "--"}


def view_sort(xyz, elev, azim):
    er, ar = np.radians(elev), np.radians(azim)
    return np.argsort(xyz @ np.array([np.cos(er)*np.cos(ar), np.cos(er)*np.sin(ar), np.sin(er)]))


# --------------------------------------------------------------- fig 1
def fig1(out):
    xyz, gt = load_raw(os.path.join(RAW, "Maize01", "M01_0325_a.txt"))
    keep = gt != 0
    sx, sl = xyz[keep], gt[keep]
    _, idx = cKDTree(sx).query(sx, k=17, workers=-1)
    seed = (sl[idx[:, 1:]] != sl[:, None]).any(axis=1)
    sxy = sx[seed]
    st = cKDTree(sxy)
    centre = sxy[int(np.argmax([len(st.query_ball_point(p, 6.0)) for p in sxy]))]

    box = np.all(np.abs(sx - centre) <= 28.0, axis=1)
    c_xyz, c_lab, c_seed = sx[box], sl[box], seed[box]
    X = c_xyz - c_xyz.mean(0)
    P = X @ np.linalg.svd(X, full_matrices=False)[2][:2].T
    d = cKDTree(c_xyz[c_seed]).query(c_xyz, k=1, workers=-1)[0]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor="white")
    S = 2.4
    for cl in (1, 2):
        m = c_lab == cl
        axes[0].scatter(P[m, 0], P[m, 1], s=S, c=CLS_COL[cl], linewidths=0, label=CLS_EN[cl])
    axes[0].set_title("(a) ground-truth organ labels", fontsize=11)
    axes[0].legend(loc="upper right", fontsize=9, frameon=False, markerscale=4)

    axes[1].scatter(P[~c_seed, 0], P[~c_seed, 1], s=S, c="#d9d9d9", linewidths=0)
    axes[1].scatter(P[c_seed, 0], P[c_seed, 1], s=7, c="#e8112d", linewidths=0,
                    label="boundary seeds ($k$=16)")
    axes[1].set_title("(b) boundary seeds", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=9, frameon=False, markerscale=3)

    axes[2].scatter(P[:, 0], P[:, 1], s=S, c="#e8e8e8", linewidths=0)
    for r, col in zip([5, 2, 1], ["#8fbcd4", "#4a7fa5", "#2a4d69"]):
        m = d <= r
        axes[2].scatter(P[m, 0], P[m, 1], s=S + 1.2, c=col, linewidths=0, label=f"$B({r}\\,$mm$)$")
    axes[2].set_title("(c) junction band $B(r)$", fontsize=11)
    h, l = axes[2].get_legend_handles_labels()
    axes[2].legend(h[::-1], l[::-1], loc="upper right", fontsize=9, frameon=False, markerscale=4)

    for ax in axes:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#cccccc")
    fig.tight_layout(); fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    print("fig1 ->", out)


# --------------------------------------------------------------- fig 2
def agg(path, species):
    store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in csv.DictReader(open(path)):
        key = "ptv3" if "ptv3" in r["model"] else "spunet"
        for c in ["mIoU_global"] + [f"O_mIoU_r{t}" for t in (1, 2, 5, 10)]:
            store[key][r["seed"]][c].append(float(r[c]))
    out = {}
    for mk in store:
        out[mk] = {}
        for c in ["mIoU_global"] + [f"O_mIoU_r{t}" for t in (1, 2, 5, 10)]:
            v = np.array([np.mean(store[mk][s][c]) for s in sorted(store[mk])])
            out[mk][c] = (v.mean(), v.std(ddof=1))
    return out


def fig2(out):
    data = {"Maize": agg(os.path.join(RESULTS, "scores-maize-corrected.csv"), "maize"),
            "Tomato": agg(os.path.join(RESULTS, "scores-tomato-corrected.csv"), "tomato")}
    radii = [1, 2, 5, 10]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="white", sharey=True)
    for ax, (sp, d) in zip(axes, data.items()):
        for mk in ("spunet", "ptv3"):
            m = np.array([d[mk][f"O_mIoU_r{t}"][0] for t in radii])
            s = np.array([d[mk][f"O_mIoU_r{t}"][1] for t in radii])
            ax.plot(radii, m, MODEL_LS[mk], color=MODEL_COL[mk], lw=2,
                    marker=MODEL_MK[mk], ms=6, label=f"{MODEL_EN[mk]}  J-IoU$(r)$", zorder=3)
            ax.fill_between(radii, m - s, m + s, color=MODEL_COL[mk], alpha=0.16, lw=0, zorder=2)
            g = d[mk]["mIoU_global"][0]
            ax.axhline(g, color=MODEL_COL[mk], ls=":", lw=1.4, alpha=0.85, zorder=1)
            # offset the two labels vertically so they do not overlap
            dy = 5 if mk == "ptv3" else -11
            ax.annotate(f"{MODEL_EN[mk]} global {g:.3f}", xy=(1, g), xytext=(2, dy),
                        textcoords="offset points", ha="left", fontsize=8.5,
                        color=MODEL_COL[mk])
        ax.set_title(sp, fontsize=12)
        ax.set_xlabel("band radius $r$ (mm)")
        ax.set_xscale("log"); ax.set_xticks(radii)
        ax.set_xticklabels([str(t) for t in radii])
        ax.grid(axis="y", color="#e8e8e8", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp_ in ax.spines.values(): sp_.set_color("#cccccc")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("mIoU")
    axes[0].legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    print("fig2 ->", out)


# --------------------------------------------------------------- fig 3
CAT_COL = {"out_ok": "#d9d9d9", "out_bad": "#f2a154", "band_ok": "#7fb3d5", "band_bad": "#e8112d"}
CAT_EN = {"out_ok": "correct, outside band", "out_bad": "incorrect, outside band",
          "band_ok": "correct, inside band", "band_bad": "incorrect, inside band"}


def panel_error(ax, raw, pred_npy, radius, elev, azim, title):
    xyz, gt = load_pheno4d_txt(raw, corrected=True)
    pred = np.load(pred_npy).astype(np.int64)
    dist, _ = distance_to_boundary(xyz, gt, seed_ignore=(0,))
    inb, ok = dist <= radius, gt == pred
    cat = np.empty(len(xyz), dtype=object)
    cat[(~inb) & ok] = "out_ok"; cat[(~inb) & ~ok] = "out_bad"
    cat[inb & ok] = "band_ok";  cat[inb & ~ok] = "band_bad"
    rng = np.random.default_rng(0)
    bi, oi = np.flatnonzero(inb), np.flatnonzero(~inb)
    if len(oi) > 110000: oi = rng.choice(oi, 110000, replace=False)
    # dense late-stage scans: thin the correct in-band points, keep every error
    bi_ok, bi_bad = bi[ok[bi]], bi[~ok[bi]]
    if len(bi_ok) > 70000: bi_ok = rng.choice(bi_ok, 70000, replace=False)
    bi = np.concatenate([bi_ok, bi_bad])
    sel = np.concatenate([oi, bi])
    xs, cs = xyz[sel], cat[sel]
    o = view_sort(xs, elev, azim); xs, cs = xs[o], cs[o]
    for c in ["out_ok", "out_bad", "band_ok", "band_bad"]:
        m = cs == c
        if not m.any(): continue
        sz = 10 if c == "band_bad" else (4 if "band" in c else 1.5)
        al = 1.0 if c == "band_bad" else (0.85 if "band" in c else 0.32)
        ax.scatter(xs[m, 0], xs[m, 1], xs[m, 2], c=CAT_COL[c], s=sz, alpha=al,
                   linewidths=0, depthshade=False, label=CAT_EN[c])
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([np.ptp(xs[:, 0]), np.ptp(xs[:, 1]), np.ptp(xs[:, 2])], zoom=1.45)
    ax.set_axis_off(); ax.set_title(title, fontsize=11)
    return float((inb & ~ok).sum()) / max(int(inb.sum()), 1)


def fig3(out):
    """Six test scans, three per species, from both test plants and from early,
    middle and late growth stages. SparseUNet seed 0 throughout, corrected
    labels."""
    cases = [
        ("Maize",  "Maize01",  "M01_0317_a", "(a) maize, plant 1, early"),
        ("Maize",  "Maize01",  "M01_0325_a", "(b) maize, plant 1, late"),
        ("Maize",  "Maize02",  "M02_0321_a", "(c) maize, plant 2, late"),
        ("Tomato", "Tomato01", "T01_0311_a", "(d) tomato, plant 1, early"),
        ("Tomato", "Tomato02", "T02_0317_a", "(e) tomato, plant 2, middle"),
        ("Tomato", "Tomato01", "T01_0324_a", "(f) tomato, plant 1, late"),
    ]
    runs = {"Maize": os.path.join(PRED_M, "semseg-spunet-pheno4d-maize-seed0"),
            "Tomato": os.path.join(PRED_T, "semseg-spunet-pheno4d-tomato-seed0")}
    fig = plt.figure(figsize=(13.5, 9.6), facecolor="white")
    errs = []
    for i, (sp, plant, scan, ttl) in enumerate(cases):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        e = panel_error(ax, os.path.join(RAW, plant, scan + ".txt"),
                        os.path.join(runs[sp], f"{plant}-{scan}.npy"),
                        5, 15, 60, ttl)
        ax.set_title(f"{ttl}\nin-band error {e*100:.1f}%", fontsize=11, pad=2)
        errs.append((scan, e))
        if i == 0:
            h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=10, markerscale=3)
    fig.tight_layout(rect=[0, 0.04, 1, 1], h_pad=1.0, w_pad=0.5)
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    print("fig3 ->", out, "  ", ", ".join(f"{s} {e*100:.1f}%" for s, e in errs))


# --------------------------------------------------------------- fig 5 (label anomaly)
def fig5(out):
    xyz, gt0 = load_raw(os.path.join(RAW, "Tomato02", "T02_0325_a.txt"))
    gt1 = gt0.copy(); gt1[gt0 == 0] = 1; gt1[gt0 == 1] = 0
    fig = plt.figure(figsize=(12.5, 6.4), facecolor="white")
    for i, (gt, ttl, note) in enumerate([
            (gt0, "(a) labels as published",
             "no organ-organ seeds; J-IoU undefined"),
            (gt1, "(b) labels with classes 0 and 1 swapped back",
             "3353 seeds; global mIoU 0.496 $\\rightarrow$ 0.950")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        keep = gt != 0
        sx, sl = xyz[keep], gt[keep]
        _, idx = cKDTree(sx).query(sx, k=17, workers=-1)
        seed = (sl[idx[:, 1:]] != sl[:, None]).any(axis=1)
        rng = np.random.default_rng(0)
        ch = seed.copy()
        rest = np.flatnonzero(~ch)
        ch[rng.choice(rest, min(120000 - ch.sum(), len(rest)), replace=False)] = True
        s_i = np.flatnonzero(ch)
        xs, ls_, sd = sx[s_i], sl[s_i], seed[s_i]
        o = view_sort(xs, 15, 60); xs, ls_, sd = xs[o], ls_[o], sd[o]
        ax.scatter(xs[~sd, 0], xs[~sd, 1], xs[~sd, 2],
                   c=[CLS_COL[int(c)] for c in ls_[~sd]], s=1.0, alpha=0.35,
                   linewidths=0, depthshade=False)
        if sd.any():
            ax.scatter(xs[sd, 0], xs[sd, 1], xs[sd, 2], c="#e8112d", s=12,
                       linewidths=0, depthshade=False)
        ax.view_init(elev=15, azim=60)
        ax.set_box_aspect([np.ptp(xs[:, 0]), np.ptp(xs[:, 1]), np.ptp(xs[:, 2])], zoom=1.4)
        ax.set_axis_off(); ax.set_title(ttl, fontsize=11, pad=0)
        ax.text2D(0.5, -0.02, note, transform=ax.transAxes, ha="center",
                  va="top", fontsize=9.5, color="#333")
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=CLS_COL[1],
                          markersize=8, label="labelled stem (class 1)"),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=CLS_COL[2],
                          markersize=8, label="labelled leaf (class 2)"),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#e8112d",
                          markersize=9, label="boundary seed")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    print("fig5 ->", out)


# --------------------------------------------------------------- fig 4 (seed spread)
def per_seed(path, col):
    """-> {model_key: [one value per seed]}, averaged over that seed's scans."""
    from collections import defaultdict
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        mk = "ptv3" if "ptv3" in r["model"] else "spunet"
        acc[mk][r["seed"]].append(float(r[col]))
    return {mk: [np.mean(acc[mk][s]) for s in sorted(acc[mk])] for mk in acc}


def fig4(out):
    """Score of every seed for both models. Each panel has its own y range,
    because global mIoU and J-IoU differ by about 0.4 on tomato and a shared
    axis would hide the spread between seeds."""
    sources = {"Maize": os.path.join(RESULTS, "scores-maize-corrected.csv"),
               "Tomato": os.path.join(RESULTS, "scores-tomato-corrected.csv")}
    metrics = [("mIoU_global", "global mIoU"), ("O_mIoU_r1", "J-IoU ($r{=}1$ mm)")]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.6), facecolor="white")
    rng = np.random.default_rng(1)
    for i, (sp, path) in enumerate(sources.items()):
        for j, (col, label) in enumerate(metrics):
            ax = axes[i][j]
            d = per_seed(path, col)
            for k, mk in enumerate(("spunet", "ptv3")):
                v = np.array(d[mk])
                x = k + (rng.random(len(v)) - 0.5) * 0.22
                ax.scatter(x, v, s=42, color=MODEL_COL[mk], marker=MODEL_MK[mk],
                           zorder=3, linewidths=0, alpha=0.9)
                ax.hlines(v.mean(), k - 0.26, k + 0.26, color=MODEL_COL[mk],
                          lw=2.2, zorder=4)
            a, b = np.array(d["spunet"]), np.array(d["ptv3"])
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            t_stat, p_val = ttest_ind(a, b, equal_var=False)   # Welch
            cohen = (b.mean() - a.mean()) / pooled
            ax.set_title(f"{sp}, {label}", fontsize=11, pad=22)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["SparseUNet", "PTv3"])
            ax.set_xlim(-0.6, 1.6)
            ax.grid(axis="y", color="#ececec", lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            for s_ in ("left", "bottom"):
                ax.spines[s_].set_color("#cccccc")
            # statistics go above the axes so they never cover data points
            mark = "*" if p_val < 0.05 else "n.s."
            ax.annotate(f"$d$ = {cohen:+.2f}   $p$ = {p_val:.3f}   {mark}",
                        xy=(0.5, 1.015), xycoords="axes fraction", ha="center",
                        va="bottom", fontsize=9.5,
                        color="#b3301c" if p_val < 0.05 else "#666")
            ax.set_ylabel("mIoU" if j == 0 else "J-IoU")
    fig.tight_layout()
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    print("fig4 ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["1", "2", "3", "4", "5"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if "1" in a.only: fig1(os.path.join(OUT, "fig1_band_definition.png"))
    if "2" in a.only: fig2(os.path.join(OUT, "fig2_radius_sweep.png"))
    if "3" in a.only: fig3(os.path.join(OUT, "fig3_junction_errors.png"))
    if "4" in a.only: fig4(os.path.join(OUT, "fig4_seed_spread.png"))
    if "5" in a.only: fig5(os.path.join(OUT, "fig5_label_anomaly.png"))
