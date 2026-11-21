"""figures.py - summary figures across groups. Reads finished runs (summary.json present, status done).

    python analysis/figures.py [--runs runs] [--out results/figures]   -> writes fig*.png
Unfinished runs are ignored.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--runs", default="runs")
_ap.add_argument("--out", default="results/figures")
_args = _ap.parse_args()
RUNS = Path(_args.runs)
OUT = Path(_args.out)
OUT.mkdir(parents=True, exist_ok=True)

# palette (light mode)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2,
})


def finished(exp):
    out = []
    for sd in sorted((RUNS / exp).glob("seed*")):
        s = sd / "summary.json"
        if not s.exists():
            continue
        summ = json.loads(s.read_text())
        if summ.get("status") != "done":
            continue
        evals = [json.loads(l) for l in (sd / "eval.jsonl").read_text().splitlines() if l.strip()]
        out.append({"seed": int(sd.name[4:]), "summary": summ, "evals": evals})
    return out


def mean_E(run):
    return run["summary"].get("avg_experts_mean")


def curve(runs, key):
    """Seed-mean curve over intermediate evals (262k-token subset; the final full-val eval is dropped)."""
    per = []
    for r in runs:
        pts = {}
        for e in r["evals"]:
            if e.get("final") or e["step"] == 0:
                continue
            pts[e["tokens"]] = key(e)
        per.append(pts)
    common = sorted(set.intersection(*[set(p) for p in per]))
    return np.array(common) / 1e6, np.array([np.mean([p[t] for p in per]) for t in common])


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- figs 1 and 6: loss vs inference / training compute
LABEL_OFFSETS = {"P50": (8, -3), "P50-noent": (-58, 2), "P50-renorm": (-30, -16), "P60": (-10, -16),
                 "K2-ent": (8, 2), "K2-renorm": (8, -8), "K2": (8, 4), "L-P60": (-10, -16), "L-P65": (-40, -14),
                 "L-K2-renorm": (8, 6), "L-P70": (6, -13)}
FLOPS_LABEL_OFFSETS = {"P50": (-12, -17), "K2-ent": (10, 0), "L-K2-renorm": (8, -12), "L-P70": (6, 6)}
PANELS = [
    ("10M tokens", ["K1", "K2", "K3"], ["P30", "P50", "P60", "P70"],
     ["K2-ent", "P50-noent", "K2-renorm", "P50-renorm"], "DENSE"),
    ("30M tokens", ["L-K1", "L-K2", "L-K3"], ["L-P60", "L-P65", "L-P70"], ["L-K2-renorm"], "L-DENSE"),
]


def topk_curve(Es, Ls, n=200):
    """Top-k loss between k = 1 and 3, linear in log E: the interpolation used for the compute-matched gaps."""
    grid = np.exp(np.linspace(np.log(Es[0]), np.log(Es[-1]), n))
    return grid, np.interp(np.log(grid), np.log(Es), Ls)


def loss_vs_compute(training, fname):
    """training=False: x = experts per token at the end of training (inference compute).
    training=True: x = total effective training FLOPs, accumulated from each step's measured experts per token."""
    xval = (lambda r: r["summary"]["cum_flops_effective"] / 1e15) if training else mean_E
    offsets = {**LABEL_OFFSETS, **FLOPS_LABEL_OFFSETS} if training else LABEL_OFFSETS
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (title, ks, ps, ctrls, dense) in zip(axes, PANELS):
        d = finished(dense)
        dl = [r["summary"]["val_loss"] for r in d]
        ax.axhspan(min(dl), max(dl), color=INK2, alpha=0.10, lw=0)
        ax.axhline(np.mean(dl), color=INK2, ls="--", lw=1.2)
        if training:
            ax.scatter([np.mean([xval(r) for r in d])], [np.mean(dl)], marker="x", s=40, color=INK2, zorder=3)
        ax.text(0.99, max(dl), "dense (mean, seed range) ", color=INK2, fontsize=8,
                ha="right", va="bottom", transform=ax.get_yaxis_transform())
        kE, kL, kX = [], [], []
        for group, color, marker in [(ks, BLUE, "o"), (ps, ORANGE, "o"), (ctrls, AQUA, "D")]:
            for exp in group:
                runs = finished(exp)
                if not runs:
                    continue
                ex = [xval(r) for r in runs]
                ey = [r["summary"]["val_loss"] for r in runs]
                ax.scatter(ex, ey, s=22, color=color, alpha=0.45, marker=marker, lw=0)
                mx, my = np.mean(ex), np.mean(ey)
                ax.scatter([mx], [my], s=70, color=color, marker=marker, edgecolor=SURF, lw=1.5, zorder=3)
                ax.annotate(exp, (mx, my), xytext=offsets.get(exp, (5, 5)), textcoords="offset points",
                            fontsize=8, color=INK)
                if group is ks:
                    kE.append(np.mean([mean_E(r) for r in runs])); kL.append(my); kX.append(mx)
        if len(kE) == 3:
            gE, gL = topk_curve(kE, kL)
            ax.plot(np.interp(gE, kE, kX), gL, color=BLUE, lw=1.5, zorder=1)  # x is affine in E for top-k
        ax.set_title(title, loc="left")
        ax.set_xlabel("total training FLOPs (PFLOP, effective, forward + backward)" if training
                      else "average active experts per token at the end of training")
        ax.set_ylabel("final validation loss (nats/token)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BLUE, marker="o", lw=1.5, label="top-k (curve: k = 1-3, log-E interpolation)"),
               Line2D([], [], color=ORANGE, marker="o", lw=0, label="top-p"),
               Line2D([], [], color=AQUA, marker="D", lw=0, label="controls (entropy, gate norm)"),
               Line2D([], [], color=INK2, ls="--", lw=1.2, label="dense, same active params")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), fontsize=9)
    note = "Large markers: seed mean. Small markers: individual seeds."
    if training:
        note += " Training FLOPs accumulate each step's measured experts per token (3x forward FLOPs)."
    fig.text(0.5, -0.11, note, ha="center", fontsize=8, color=INK2)
    save(fig, fname)


def fig_frontier():
    loss_vs_compute(False, "fig1_loss_vs_experts.png")


def fig_train_flops():
    loss_vs_compute(True, "fig6_loss_vs_train_flops.png")


# ---------------------------------------------------------------- fig 2: dense minus top-2 over training
def fig_dense_gap():
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for dense, k2, label, color in [("DENSE", "K2", "10M-token schedule", ORANGE),
                                    ("L-DENSE", "L-K2", "30M-token schedule", BLUE)]:
        td, ld = curve(finished(dense), lambda e: e["val_loss"])
        tk, lk = curve(finished(k2), lambda e: e["val_loss"])
        common = np.intersect1d(td, tk)
        gap = np.array([ld[td == t][0] - lk[tk == t][0] for t in common])
        ax.plot(common, gap, color=color, label=label)
    ax.axhline(0, color=INK2, lw=1)
    ax.text(0.2, 0.004, "above 0: top-2 MoE better", color=INK2, fontsize=8, va="bottom")
    ax.text(0.2, -0.004, "below 0: dense better", color=INK2, fontsize=8, va="top")
    ax.set_ylim(-0.08, 0.04)
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("dense loss minus top-2 loss")
    ax.set_title("Dense minus top-2 MoE during training", loc="left")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "fig2_dense_minus_moe.png")


# ---------------------------------------------------------------- fig 3: experts per token over training
def fig_E_drift():
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for exp, label, color, ls in [("P60", "p = 0.6, 10M", BLUE, "--"), ("L-P60", "p = 0.6, 30M", BLUE, "-"),
                                  ("L-P65", "p = 0.65, 30M", AQUA, "-"),
                                  ("P70", "p = 0.7, 10M", ORANGE, "--"), ("L-P70", "p = 0.7, 30M", ORANGE, "-")]:
        t, E = curve(finished(exp), lambda e: float(np.mean(e["routing"]["avg_experts"])))
        ax.plot(t, E, color=color, ls=ls, label=label)
    ax.axhline(2, color=INK2, lw=1, ls=":")
    ax.text(29.5, 2.02, "top-2", color=INK2, fontsize=8, ha="right", va="bottom")
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("average active experts per token")
    ax.set_title("A fixed p uses fewer experts the longer it trains", loc="left")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    save(fig, "fig3_experts_over_training.png")


# ---------------------------------------------------------------- fig 4: experts per layer
def fig_layers():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    panels = [("10M, raw gates", [("P30", BLUE), ("P50", ORANGE), ("P60", AQUA), ("P70", YELLOW)]),
              ("10M, p = 0.5 controls", [("P50", ORANGE), ("P50-noent", BLUE), ("P50-renorm", AQUA)]),
              ("30M, raw gates", [("L-P60", BLUE), ("L-P65", ORANGE), ("L-P70", AQUA)])]
    x = np.arange(4)
    for ax, (title, series) in zip(axes, panels):
        for exp, color in series:
            arr = np.array([r["summary"]["avg_experts_by_layer"] for r in finished(exp)])
            m = arr.mean(0)
            ax.plot(x, m, color=color, marker="o", ms=6, label=f"{exp} (n={len(arr)})")
            if len(arr) > 1:
                ax.fill_between(x, arr.min(0), arr.max(0), color=color, alpha=0.12, lw=0)
        ax.set_xticks(x, ["L0", "L1", "L2", "L3"])
        ax.set_title(title, loc="left")
        ax.set_xlabel("MoE layer (bottom to top)")
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("average active experts per token")
    fig.text(0.5, -0.04, "Line: seed mean. Band: seed min to max.", ha="center", fontsize=8, color=INK2)
    save(fig, "fig4_experts_by_layer.png")


# ---------------------------------------------------------------- fig 5: entropy-weight sweep (top-p 0.5)
ENT_SWEEP = [("P50-noent", "β = 0"), ("P50", "β = 1e-4"), ("P50-ent1e-3", "β = 1e-3"), ("P50-ent1e-2", "β = 1e-2")]
ENT_COLORS = ["#a9c9ef", "#6aa3e3", "#2a78d6", "#123f78"]  # light to dark: ordinal in β
def fig_entropy():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0))
    kx, ky = [], []
    for k in ("K1", "K2", "K3"):
        runs = finished(k)
        kx.append(np.mean([mean_E(r) for r in runs])); ky.append(np.mean([r["summary"]["val_loss"] for r in runs]))
    a1.plot(kx, ky, color=INK2, lw=1.5, marker="s", ms=5, zorder=1)
    for x, y, k in zip(kx, ky, ("K1", "K2", "K3")):
        a1.annotate(k, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK2)
    for (exp, label), color in zip(ENT_SWEEP, ENT_COLORS):
        runs = finished(exp)
        ex = [mean_E(r) for r in runs]; ey = [r["summary"]["val_loss"] for r in runs]
        a1.scatter(ex, ey, s=22, color=color, alpha=0.6, lw=0)
        a1.scatter([np.mean(ex)], [np.mean(ey)], s=70, color=color, edgecolor=SURF, lw=1.5, zorder=3, label=label)
        t, E = curve(runs, lambda e: float(np.mean(e["routing"]["avg_experts"])))
        a2.plot(t, E, color=color, label=label)
    a1.set_xlim(0.9, 3.1)
    a1.set_title("Loss against compute (gray: top-k curve)", loc="left")
    a1.set_xlabel("average active experts per token")
    a1.set_ylabel("final validation loss (nats/token)")
    a1.legend(fontsize=8, loc="upper right", title="top-p 0.5", title_fontsize=8)
    a2.axhline(1, color=INK2, lw=1, ls=":")
    a2.set_title("Experts per token during training", loc="left")
    a2.set_xlabel("training tokens (M)")
    a2.set_ylabel("average active experts per token")
    a2.legend(fontsize=8, loc="upper right")
    fig.text(0.5, -0.04, "10M tokens, 2 seeds per β. Large markers: seed mean. Small markers: individual seeds.",
             ha="center", fontsize=8, color=INK2)
    save(fig, "fig5_entropy_sweep.png")


if __name__ == "__main__":
    fig_frontier()
    fig_dense_gap()
    fig_E_drift()
    fig_layers()
    fig_entropy()
    fig_train_flops()
