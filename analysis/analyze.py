#!/usr/bin/env python3
"""analyze.py - results tables and figures for the finished runs in runs/.

    python analysis/analyze.py --only main  --out results/analyze_main    # 10M runs
    python analysis/analyze.py --only scale --out results/analyze_scale   # 30M runs, and 10M vs 30M

Prints a results table, the top-p vs top-k gap at matched experts per token, the inference-time routing sweep and
health flags, and writes PNG figures into --out. Only completed runs (those with a summary.json) are analysed.
The 30M-token runs (main-30m group, L-<ID>) are kept out of the 10M tables; --only scale analyses only them, with every
table and figure, as if they were a study of their own, plus a comparison with the 10M runs of the same model.
Without --only, the 10M analysis is followed by the 10M-vs-30M comparison.

frontier.png plots final validation loss against average experts per token.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["K1", "K2", "K3", "P30", "P50", "P60", "P70", "K2-ent", "P50-noent", "DENSE", "K2-renorm", "P50-renorm"]
COLORS = {"K1": "#9ecae1", "K2": "#3182bd", "K3": "#08519c", "K2-ent": "#6baed6", "K2-renorm": "#17becf",
          "P30": "#fdae6b", "P50": "#e6550d", "P60": "#cb181d", "P70": "#a63603", "P50-noent": "#fd8d3c", "P50-renorm": "#d62728",
          "DENSE": "#555555"}


def jl(path):
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return out


def load(runs_dir):
    runs = defaultdict(dict)  # id -> seed -> run
    for rd in sorted(Path(runs_dir).glob("*/seed*")):
        ev, mt = jl(rd / "eval.jsonl"), jl(rd / "metrics.jsonl")
        if not ev or not (rd / "summary.json").exists():  # only completed runs are analysed
            continue
        done = True
        summ = json.loads((rd / "summary.json").read_text()) if done else None
        sweep = json.loads((rd / "sweep_eval.json").read_text())["rows"] if (rd / "sweep_eval.json").exists() else None
        last = ev[-1]
        cfg = rd / "config.json"
        data = json.loads(cfg.read_text())["train"]["data_dir"] if cfg.exists() else "data/tinystories"
        runs[rd.parent.name][int(rd.name[4:])] = dict(
            done=done, ev=ev, mt=mt, sweep=sweep, data=Path(data).name, loss=last["val_loss"], ppl=last["val_ppl"], bpb=last.get("val_bpb"),
            E=last.get("avg_experts_mean"), flops=last["cum_flops_effective"], step=last["step"],
            tps=(summ or {}).get("mean_tokens_per_s"), routing=last.get("routing"))
    return runs


SCALE_PREFIXES = ("L-", "S30-")  # ids of the 30M-token runs (main-30m group)


def base_id(i):
    """Experiment id without the L- (30M tokens) prefix."""
    return i.removeprefix("L-").removeprefix("S30-")


def is_scale(i, seeds):
    return i.startswith(SCALE_PREFIXES) or any("30m" in r["data"].lower() for r in seeds.values())


def ordered_ids(runs):
    return [i for i in ORDER if i in runs] + sorted(i for i in runs if i not in ORDER and not i.startswith(("LR-", "AL-")))


def interp_topk(runs, seed, x, log=False):
    """Piecewise-linear top-k loss at x experts per token, from K1..K3 (this seed if all exist, else the mean)."""
    pts = []
    for kid, k in (("K1", 1), ("K2", 2), ("K3", 3)):
        if kid not in runs:
            continue
        if seed in runs[kid]:
            pts.append((runs[kid][seed]["E"], runs[kid][seed]["loss"]))
        else:
            pts.append((np.mean([r["E"] for r in runs[kid].values()]), np.mean([r["loss"] for r in runs[kid].values()])))
    pts.sort()
    if len(pts) < 2 or not (pts[0][0] - 1e-6 <= x <= pts[-1][0] + 1e-6):
        return None
    f = np.log if log else (lambda v: v)  # log=True: interpolate linearly in log(experts per token)
    return float(np.interp(f(x), [f(p[0]) for p in pts], [p[1] for p in pts]))


def table(runs):
    print("\n== final validation loss (lower is better) ==")
    print(f"{'run':<12}{'seeds':<7}{'val_loss (mean [min,max])':<28}{'ppl':<8}{'bpb':<8}{'experts/tok':<13}{'eff fwd FLOPs/tok':<19}{'status'}")
    for i in ordered_ids(runs):
        rs = list(runs[i].values())
        L = [r["loss"] for r in rs]
        spread = f"{np.mean(L):.4f} [{min(L):.4f},{max(L):.4f}]" if len(L) > 1 else f"{L[0]:.4f}"
        E = [r["E"] for r in rs if r["E"] is not None]
        gf = f"{np.mean([r['ev'][-1]['flops_per_token_effective'] for r in rs]) / 1e6:.2f}M"
        status = "done" if all(r["done"] for r in rs) else f"running (step {max(r['step'] for r in rs)})"
        print(f"{i:<12}{len(rs):<7}{spread:<28}{np.mean([r['ppl'] for r in rs]):<8.2f}"
              f"{(np.mean([r['bpb'] for r in rs]) if rs[0]['bpb'] else float('nan')):<8.3f}"
              f"{(np.mean(E) if E else float('nan')):<13.2f}{gf:<19}{status}")


def gaps(runs):
    print("\n== top-p vs top-k at matched compute (gap = top-k loss at the same experts/token minus top-p loss; > 0 means top-p better) ==")
    spread = None
    if "K2" in runs and len(runs["K2"]) > 1:
        v = [r["loss"] for r in runs["K2"].values()]
        spread = max(v) - min(v)
        print(f"seed-to-seed spread of K2: {spread:.4f}  <- a gap smaller than this is not evidence of a difference")
    if not any(k in runs for k in ("K1", "K3")):
        print("(need K1/K2/K3 to interpolate the top-k curve)")
    for i in ordered_ids(runs):
        if not i.startswith("P") or i.endswith(("noent", "renorm")):
            continue
        per = []
        for s, r in runs[i].items():
            t, tl = interp_topk(runs, s, r["E"]), interp_topk(runs, s, r["E"], log=True)
            if t is not None:
                per.append((s, r["E"], t - r["loss"], tl - r["loss"]))
        if per:
            print(f"{i:<8} experts/tok {np.mean([x[1] for x in per]):.2f}  gap linear {np.mean([x[2] for x in per]):+.4f}"
                  f"   gap log-spaced {np.mean([x[3] for x in per]):+.4f}   per seed (log): "
                  + ", ".join(f"s{s}={d:+.4f}" for s, _, _, d in per))
        else:
            print(f"{i:<8} experts/tok outside the K1..K3 range, cannot interpolate")
    print("(the log-spaced gap is the more conservative of the two; trust a result only if both clear the seed spread)")


def health(runs):
    print("\n== health flags ==")
    bad = False
    for i in ordered_ids(runs):
        for s, r in sorted(runs[i].items()):
            flags = []
            mt = r["mt"]
            if mt:
                if np.mean([m["clip_frac"] for m in mt]) > 0.5:
                    flags.append("gradient clipped in over half of steps (lr too high?)")
                if r["E"] is not None and "avg_experts" in mt[-1]:
                    load = np.array(mt[-1]["expert_load"])
                    if load.max() > 0.35:
                        flags.append(f"one expert takes {load.max():.0%} of assignments (collapse?)")
                    ae = np.array(mt[-1]["avg_experts"])
                    if base_id(i).startswith("P") and (ae.min() < 1.05 or ae.max() > 3.95) and "renorm" not in i:
                        flags.append(f"a layer uses almost exactly 1 expert per token or sits at the maximum ({ae.round(2).tolist()})")
            per = [e for e in r["ev"] if not e.get("final")]  # periodic evals share one validation subset; the final eval does not
            if len(per) > 2 and per[-1]["val_loss"] > per[len(per) // 2]["val_loss"] and r["done"]:
                flags.append("validation loss rose in the second half")
            if flags:
                bad = True
                print(f"{i}/seed{s}: " + "; ".join(flags))
    print("none" if not bad else "")


def fig_frontier(runs, out):
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ks = [(i, k) for i, k in (("K1", 1), ("K2", 2), ("K3", 3)) if i in runs]
    if len(ks) > 1:
        ax.plot([np.mean([r["E"] for r in runs[i].values()]) for i, _ in ks],
                [np.mean([r["loss"] for r in runs[i].values()]) for i, _ in ks], "-", color="#3182bd", alpha=.5, zorder=1, label="top-k frontier")
    for i in ordered_ids(runs):
        rs = list(runs[i].values())
        if rs[0]["E"] is None:
            continue
        c = COLORS.get(i, "k")
        ax.scatter([r["E"] for r in rs], [r["loss"] for r in rs], s=14, color=c, alpha=.6, zorder=2)
        ax.scatter([np.mean([r["E"] for r in rs])], [np.mean([r["loss"] for r in rs])], s=70, color=c, edgecolor="k", zorder=3)
        ax.annotate(i, (np.mean([r["E"] for r in rs]), np.mean([r["loss"] for r in rs])), textcoords="offset points", xytext=(6, 4), fontsize=8)
    if "DENSE" in runs:
        d = np.mean([r["loss"] for r in runs["DENSE"].values()])
        ax.axhline(d, color="#555", ls="--", lw=1)
        ax.text(ax.get_xlim()[1], d, " dense", va="bottom", ha="right", fontsize=8, color="#555")
    ax.set_xlabel("average experts per token (compute)"); ax.set_ylabel("final validation loss")
    ax.set_title("Loss vs compute: lower and further left is better"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "frontier.png", dpi=150); plt.close(fig)


def fig_curves(runs, out):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    allv = []
    for i in ordered_ids(runs):
        c = COLORS.get(i, "k")
        for j, (s, r) in enumerate(sorted(runs[i].items())):
            ev = [e for e in r["ev"] if e["step"] > 0 and not e.get("final")]  # periodic evals: same 262k-token subset
            fin = [e for e in r["ev"] if e.get("final")]  # final eval uses the WHOLE validation file: a different set
            lab = i if j == 0 else None
            axs[0].plot([e["tokens"] / 1e6 for e in ev], [e["val_loss"] for e in ev], color=c, alpha=.8, label=lab, lw=1.2)
            axs[1].plot([e["cum_flops_effective"] for e in ev], [e["val_loss"] for e in ev], color=c, alpha=.8, label=lab, lw=1.2)
            if fin:
                axs[0].scatter([fin[-1]["tokens"] / 1e6], [fin[-1]["val_loss"]], marker="x", s=18, color=c, alpha=.8)
                axs[1].scatter([fin[-1]["cum_flops_effective"]], [fin[-1]["val_loss"]], marker="x", s=18, color=c, alpha=.8)
            if r["mt"]:
                axs[2].plot([m["tokens"] / 1e6 for m in r["mt"]], [m["lm_loss"] for m in r["mt"]], color=c, alpha=.6, label=lab, lw=.9)
            allv += [e["val_loss"] for e in ev]
    if allv:
        lo, hi = min(allv), np.percentile(allv, 50)  # zoom on the converged half; early losses are huge
        for a in axs[:2]:
            a.set_ylim(lo * 0.99, hi)
        axs[2].set_ylim(lo * 0.97, hi)
    for a, (xl, t) in zip(axs, (("tokens (M)", "validation loss vs tokens (line: 262k-token subset; x: final, whole val file)"), ("cumulative effective training FLOPs", "validation loss vs compute (x: final, whole val file)"),
                                ("tokens (M)", "training LM loss (10-step means)"))):
        a.set_xlabel(xl); a.set_title(t); a.grid(alpha=.3)
    axs[0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "curves.png", dpi=150); plt.close(fig)


def fig_layers(runs, out):
    ids = [i for i in ordered_ids(runs) if i.startswith("P") or i in ("K2",)]
    ids = [i for i in ids if any(r["mt"] and "avg_experts" in r["mt"][0] for r in runs[i].values())]
    if not ids:
        return
    fig, axs = plt.subplots(1, len(ids), figsize=(3.6 * len(ids), 3.6), sharey=True, squeeze=False)
    for a, i in zip(axs[0], ids):
        r = runs[i][min(runs[i])]
        ae = np.array([m["avg_experts"] for m in r["mt"]])
        for l in range(ae.shape[1]):
            a.plot([m["tokens"] / 1e6 for m in r["mt"]], ae[:, l], label=f"layer {l}")
        a.set_title(f"{i}: experts per token by layer"); a.set_xlabel("tokens (M)"); a.grid(alpha=.3)
    axs[0][0].set_ylabel("average experts per token"); axs[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "experts_by_layer.png", dpi=150); plt.close(fig)


def sweep_mean(runs, i):
    """Inference-time sweep rows of run i, averaged over its seeds (same label order in every seed)."""
    per = [r["sweep"] for r in runs[i].values() if r["sweep"]]
    if not per:
        return None
    out = []
    for rows in zip(*per):
        out.append({"label": rows[0]["label"], "val_loss": float(np.mean([x["val_loss"] for x in rows])),
                    "E": float(np.mean([x["avg_experts_mean"] for x in rows])), "n": len(rows)})
    return out


def sweep_table(runs):
    ids = [i for i in ("K2", "P30", "P50", "P60", "P70") if i in runs and sweep_mean(runs, i)]
    if not ids:
        return
    print("\n== inference-time sweep (trained model re-evaluated with another routing setting; val loss [experts/token]) ==")
    labels = [r["label"] for r in sweep_mean(runs, ids[0])]
    print(f"{'setting':<12}" + "".join(f"{'trained '+i:<20}" for i in ids))
    for j, lab in enumerate(labels):
        cells = []
        for i in ids:
            r = sweep_mean(runs, i)[j]
            cells.append(f"{r['val_loss']:.4f} [{r['E']:.2f}]")
        print(f"{lab:<12}" + "".join(f"{c:<20}" for c in cells))
    print("(first 131k validation tokens, mean over seeds; compare columns only with care, rows within a column are paired)")


def fig_sweep(runs, out):
    ids = [i for i in ("K2", "P30", "P50", "P60", "P70") if i in runs and sweep_mean(runs, i)]
    if not ids:
        return
    fig, axs = plt.subplots(1, len(ids), figsize=(3.6 * len(ids), 3.8), sharey=True, squeeze=False)
    for a, i in zip(axs[0], ids):
        rows = sweep_mean(runs, i)
        for pre, mk, lab in (("topp", "o-", "inference top-p"), ("topk", "s--", "inference top-k")):
            sel = sorted((r for r in rows if r["label"].startswith(pre)), key=lambda r: r["E"])
            a.plot([r["E"] for r in sel], [r["val_loss"] for r in sel], mk, label=lab, ms=4)
        t = next(r for r in rows if r["label"] == "as_trained")
        a.scatter([t["E"]], [t["val_loss"]], s=120, marker="*", color="k", zorder=5, label="as trained")
        a.set_title(f"trained with {i} ({rows[0]['n']} seeds)", fontsize=9); a.set_xlabel("experts per token at inference"); a.grid(alpha=.3)
    axs[0][0].set_ylabel("validation loss (131k-token subset)"); axs[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "inference_sweep.png", dpi=150); plt.close(fig)


def fig_health(runs, out):
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    for i in ordered_ids(runs):
        r = runs[i][min(runs[i])]
        if not r["mt"]:
            continue
        x = [m["tokens"] / 1e6 for m in r["mt"]]
        c = COLORS.get(i, "k")
        axs[0].plot(x, [m["grad_norm"] for m in r["mt"]], color=c, label=i, lw=1)
        if "expert_load" in r["mt"][0]:
            axs[1].plot(x, [np.max(m["expert_load"]) for m in r["mt"]], color=c, lw=1)
            axs[2].plot(x, [np.mean(m["router_entropy"]) for m in r["mt"]], color=c, lw=1)
    axs[0].set_title("gradient norm (before clipping)"); axs[1].set_title("largest expert load (balanced = 1/8)")
    axs[2].set_title("router entropy (mean over layers)")
    axs[1].axhline(1 / 8, color="gray", ls=":")
    for a in axs:
        a.set_xlabel("tokens (M)"); a.grid(alpha=.3)
    axs[0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "health.png", dpi=150); plt.close(fig)


def fig_lr(runs_dir, out):
    rows = defaultdict(dict)
    for rd in Path(runs_dir).glob("LR-*/seed0"):
        if (rd / "summary.json").exists():
            fam, lr = rd.parent.name.split("-")[1], rd.parent.name.split("-", 2)[2]
            rows[fam][float(lr)] = json.loads((rd / "summary.json").read_text())["val_loss"]
    if rows:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for fam, d in rows.items():
            xs = sorted(d)
            ax.plot(xs, [d[x] for x in xs], "o-", label=fam)
        ax.set_xscale("log"); ax.set_xlabel("peak learning rate"); ax.set_ylabel("val loss at 3M tokens"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out / "lr_sweep.png", dpi=150); plt.close(fig)


def mean_loss(runs, i):
    return float(np.mean([r["loss"] for r in runs[i].values()]))


def scale_compare(sc, main, out):
    """30M-token runs (ids without the L- prefix) against the 10M-token runs of the same model.
    The two scales use different learning-rate schedules (cosine over 10M vs 30M tokens) and the same validation file,
    so only the FINAL losses (whole validation file) are compared, not the curves point by point."""
    ids = [i for i in ordered_ids(sc) if i in main]
    if not ids:
        print("\n== scale: no 30M run has a 10M counterpart in the main runs yet =="); return
    print("\n== scale: 10M tokens vs 30M tokens (same model, same validation file; final validation loss) ==")
    print(f"{'run':<8}{'10M seeds':<10}{'10M loss':<10}{'30M seeds':<10}{'30M loss':<10}{'gain':<9}{'experts/tok 10M':<17}{'experts/tok 30M':<17}status")
    fmt = lambda v: "-" if v != v else f"{v:.2f}"  # noqa: E731
    E = lambda rs: np.mean([r["E"] for r in rs.values() if r["E"] is not None]) if any(r["E"] is not None for r in rs.values()) else float("nan")  # noqa: E731
    for i in ids:
        l10, l30 = mean_loss(main, i), mean_loss(sc, i)
        status = "done" if all(r["done"] for r in sc[i].values()) else f"running (step {max(r['step'] for r in sc[i].values())})"
        print(f"{i:<8}{len(main[i]):<10}{l10:<10.4f}{len(sc[i]):<10}{l30:<10.4f}{l10 - l30:<+9.4f}{fmt(E(main[i])):<17}{fmt(E(sc[i])):<17}{status}")
    spread = None
    if "K2" in main and len(main["K2"]) > 1:
        v = [r["loss"] for r in main["K2"].values()]; spread = max(v) - min(v)
    pairs = [(a, b) for a, b in (("P60", "K2"), ("K2", "DENSE"), ("P60", "DENSE")) if a in ids and b in ids]
    if pairs:
        print("\ndifference A minus B (negative = A better):")
        print(f"{'A minus B':<16}{'10M':<10}{'30M':<10}{'change':<10}reading")
    for a, b in pairs:
        d10, d30 = mean_loss(main, a) - mean_loss(main, b), mean_loss(sc, a) - mean_loss(sc, b)
        ch = d30 - d10
        if spread is None:
            read = "no noise guide (the 10M K2 has one seed)"
        elif abs(ch) < spread:
            read = f"change smaller than the 10M K2 seed spread ({spread:.4f}): cannot tell from noise"
        else:
            read = ("gap flipped sign" if d10 * d30 < 0 else "A's advantage grew" if abs(d30) > abs(d10) and d30 < 0 else
                    "B's advantage grew" if abs(d30) > abs(d10) else "gap shrank")
        print(f"{a + ' - ' + b:<16}{d10:<+10.4f}{d30:<+10.4f}{ch:<+10.4f}{read}")
    if "P60" in ids and "K2" in ids:
        e = [(E(main["P60"]), E(main["K2"])), (E(sc["P60"]), E(sc["K2"]))]
        print(f"P60 uses {e[0][0]:.2f} experts per token at 10M and {e[1][0]:.2f} at 30M (K2 uses {e[0][1]:.2f}); "
              + ("compute is about the same at both scales" if abs(e[0][0] - e[1][0]) < 0.15 else "COMPUTE CHANGED between scales, so the P60 - K2 comparison is not like for like"))
    n30 = {len(sc[i]) for i in ids}
    print(f"(30M runs have {sorted(n30)} seed(s) each; with one seed there is no noise estimate at 30M, so use the 10M seed spread as a rough guide only)")

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for i in ids:
        c = COLORS.get(i, "k")
        for j, (src, ls, lab) in enumerate(((main, "--", "10M"), (sc, "-", "30M"))):
            r = src[i][min(src[i])]
            ev = [e for e in r["ev"] if e["step"] > 0 and not e.get("final")]
            axs[0].plot([e["tokens"] / 1e6 for e in ev], [e["val_loss"] for e in ev], ls, color=c, lw=1.2, label=f"{i} {lab}")
    allv = [e["val_loss"] for src in (main, sc) for i in ids for r in src[i].values() for e in r["ev"] if e["step"] > 0 and not e.get("final")]
    if allv:
        axs[0].set_ylim(min(allv) * 0.99, np.percentile(allv, 60))
    axs[0].set_xlabel("tokens (M)"); axs[0].set_ylabel("validation loss (262k-token subset)")
    axs[0].set_title("dashed: 10M-token run, solid: 30M-token run\n(different learning-rate schedules)", fontsize=9)
    axs[0].grid(alpha=.3); axs[0].legend(fontsize=7, ncol=2)
    moe = [i for i in ids if i != "DENSE"]
    if "DENSE" in ids and moe:
        x = np.arange(len(moe))
        d10 = [[r["loss"] - mean_loss(main, "DENSE") for r in main[i].values()] for i in moe]
        axs[1].bar(x - .2, [np.mean(d) for d in d10], .4, color="#9ecae1", label="10M tokens (bars: seed min/max)",
                   yerr=[[np.mean(d) - min(d) for d in d10], [max(d) - np.mean(d) for d in d10]], capsize=3)
        axs[1].bar(x + .2, [mean_loss(sc, i) - mean_loss(sc, "DENSE") for i in moe], .4, color="#e6550d", label="30M tokens")
        axs[1].set_xticks(x); axs[1].set_xticklabels(moe); axs[1].axhline(0, color="gray", lw=1)
        axs[1].set_ylabel("val loss minus dense (same scale)"); axs[1].grid(alpha=.3, axis="y"); axs[1].legend(fontsize=8)
        axs[1].set_title("MoE vs dense at each scale (below 0 = MoE better)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "scale.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="results/analyze")
    ap.add_argument("--only", choices=["main", "scale"], help="main: only the 10M runs; scale: ONLY the 30M-token runs "
                    "(L-*), with every table and figure, plus the comparison with the 10M runs. Default: main + scale comparison")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    allruns = load(a.runs)
    if not allruns:
        print("no runs with eval records found in", a.runs); return 1
    scale_raw = {i: s for i, s in allruns.items() if is_scale(i, s)}
    scale = {base_id(i): s for i, s in scale_raw.items()}  # K2, P60, DENSE ... trained on 30M tokens
    runs = {i: s for i, s in allruns.items() if i not in scale_raw}
    if a.only == "scale":  # the full analysis, on the 30M-token runs only
        if not scale:
            print("no 30M-token runs (L-*) with eval records found in", a.runs); return 1
        print(f"##### 30M-token runs only: {', '.join(sorted(scale))} #####")
        table(scale); gaps(scale); sweep_table(scale); health(scale)
        scale_compare(scale, runs, out)
        for f in (fig_frontier, fig_curves, fig_layers, fig_sweep, fig_health):
            f(scale, out)
        print(f"\nfigures written to {out}/: " + ", ".join(sorted(p.name for p in out.glob('*.png'))))
        return 0
    table(runs); gaps(runs); sweep_table(runs)
    if scale and a.only != "main":
        scale_compare(scale, runs, out)
    health({**runs, **scale_raw} if a.only != "main" else runs)
    for f in (fig_frontier, fig_curves, fig_layers, fig_sweep, fig_health):
        f(runs, out)
    fig_lr(a.runs, out)
    print(f"\nfigures written to {out}/: " + ", ".join(sorted(p.name for p in out.glob('*.png'))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())