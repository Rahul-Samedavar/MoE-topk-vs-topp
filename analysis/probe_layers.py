#!/usr/bin/env python3
"""probe_layers.py - per-layer routing probes on existing run files (no training).

    python analysis/probe_layers.py              # writes figures + probe_layers.json to results/probe_layers/
    python analysis/probe_layers.py --out somewhere --min-occ 30

Only finished runs (summary.json present) are used.

A. Token determinism (routing_dump.npz). How much of each layer's expert choice is fixed by the input token id alone,
   and how much more the previous token adds; for top-p and top-k runs.
B. Early training window (metrics.jsonl). For each layer: the dip (min avg experts in the first --dip-search tokens),
   the rebound peak after it, and router entropy / top prob / largest expert load / share at the cap at both points.
   Also the LR-P50-* and AL-P50-* sweeps (dip and rebound against learning rate and load-balance weight).
C. Floor/cap censoring and final per-layer table (eval.jsonl final record). Share of tokens at min_k and at max_k,
   mean experts among uncapped tokens, and raw vs renorm / ent vs noent comparisons.

Plug-in mutual information is biased upward for sparse groups, so every MI is also reported shuffle-corrected
(MI minus the mean MI after randomly permuting the expert labels), and only tokens / bigrams that occur at least
--min-occ times in the dump are used.
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_A = ["P50", "P60", "P70", "P50-renorm", "P50-noent", "K2", "K2-renorm", "L-P60", "L-P65", "L-P70", "L-K2"]
RUNS_B = ["P50", "P60", "P70", "L-P60", "L-P70", "P50-renorm", "K2"]
LR_SWEEP = ["LR-P50-1e-3", "LR-P50-2e-3", "LR-P50-4e-3", "LR-P50-8e-3"]
AL_SWEEP = ["AL-P50-1e-3", "AL-P50-1e-2", "AL-P50-1e-1", "AL-P50-3e-1"]
RUNS_C = ["P30", "P50", "P60", "P70", "P50-renorm", "P50-noent", "K2", "K2-renorm",
          "L-P60", "L-P65", "L-P70", "L-K2"]
LCOL = plt.get_cmap("tab10").colors


# ------------------------------------------------------------------ io / formatting
def jl(p):
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def seed_dirs(runs, rid):
    return sorted(d for d in (runs / rid).glob("seed*") if d.is_dir() and (d / "summary.json").exists())


def cfg(rd):
    return json.loads((rd / "config.json").read_text())


def fmt(v, d=2):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "-"
    return f"{v.mean():.{d}f}" + (f"±{v.std():.{d}f}" if len(v) > 1 else "")


def table(title, headers, rows):
    print(f"\n{title}")
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h)) for i, h in enumerate(headers)]
    print("  " + "  ".join(str(h).rjust(w[i]) for i, h in enumerate(headers)))
    for r in rows:
        print("  " + "  ".join(str(c).rjust(w[i]) for i, c in enumerate(r)))


def mean_dicts(ds):
    return {k: [d[k] for d in ds] for k in ds[0]}


# ------------------------------------------------------------------ A. token determinism
def ent(x):
    _, c = np.unique(x, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def cond_ent(y, g):
    return ent(g * (int(y.max()) + 1) + y) - ent(g)


def mi(y, g, rng, n_perm):
    """(plug-in MI, shuffle-corrected MI) in bits."""
    hy = ent(y)
    raw = hy - cond_ent(y, g)
    shuf = np.mean([hy - cond_ent(rng.permutation(y), g) for _ in range(n_perm)])
    return raw, raw - shuf


def frequent(g, min_occ):
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    return cnt[inv] >= min_occ


def purity(y, g):
    """Share of occurrences that use their token's most common expert set."""
    key = g * 256 + y
    uk, c = np.unique(key, return_counts=True)
    gk = uk // 256
    starts = np.r_[0, np.flatnonzero(np.diff(gk)) + 1]
    return float(np.maximum.reduceat(c, starts).sum() / len(y))


def eta2(x, g, rng, n_perm):
    """Share of variance of x explained by group g, minus the same after shuffling x."""
    _, inv = np.unique(g, return_inverse=True)
    n = np.bincount(inv)

    def f(xx):
        m = np.bincount(inv, xx) / n
        tot = ((xx - xx.mean()) ** 2).sum()
        return 1 - ((xx - m[inv]) ** 2).sum() / tot if tot > 0 else np.nan

    raw = f(x)
    return raw, raw - np.mean([f(rng.permutation(x)) for _ in range(n_perm)])


def probe_dump(path, min_occ, n_perm, rng):
    d = np.load(path)
    mask = d["expert_mask"].astype(bool)
    tok = d["tokens"].astype(np.int64)
    L, T, E = mask.shape
    seq = int(d["seq_len"])
    sets = (mask.astype(np.int64) << np.arange(E)).sum(-1)  # [L, T] expert set as a bitmask id
    counts = mask.sum(-1).astype(np.float64)
    pos = np.arange(T) % seq
    V = int(tok.max()) + 1
    bigram = np.where(pos > 0, np.r_[0, tok[:-1]] * V + tok, -(np.arange(T) + 1))  # window starts never repeat
    k_tok = frequent(tok, min_occ)
    k_bi = frequent(bigram, min_occ)
    out = []
    for l in range(L):
        y, g = sets[l][k_tok], tok[k_tok]
        H = ent(y)
        mi_raw, mi_c = mi(y, g, rng, n_perm)
        top_set_share = np.unique(y, return_counts=True)[1].max() / len(y)
        he, mie = 0.0, 0.0
        for e in range(E):
            ye = mask[l, k_tok, e].astype(np.int64)
            he += ent(ye)
            mie += mi(ye, g, rng, n_perm)[1]
        cnt_eta = eta2(counts[l][k_tok], g, rng, n_perm)[1] if counts[l][k_tok].std() > 0 else np.nan
        yb = sets[l][k_bi]
        hb = ent(yb)
        gain = (mi(yb, bigram[k_bi], rng, n_perm)[1] - mi(yb, tok[k_bi], rng, n_perm)[1]) / hb if hb > 0 else np.nan
        out.append(dict(H_set=H, MI_over_H=mi_c / H if H > 0 else np.nan, MI_raw_over_H=mi_raw / H if H > 0 else np.nan,
                        purity=purity(y, g), top_set_share=float(top_set_share),
                        expert_MI_over_H=mie / he if he > 0 else np.nan, count_eta2=cnt_eta, ctx_gain_over_H=gain,
                        avg_experts=float(counts[l].mean()), cover_tok=float(k_tok.mean()), cover_bi=float(k_bi.mean())))
    return out


def section_a(runs, a, rng, res):
    rows, res["A"] = [], {}
    for rid in RUNS_A:
        per_seed = [probe_dump(rd / "routing_dump.npz", a.min_occ, a.n_perm, rng)
                    for rd in seed_dirs(runs, rid) if (rd / "routing_dump.npz").exists()]
        if not per_seed:
            continue
        res["A"][rid] = []
        for l in range(len(per_seed[0])):
            m = mean_dicts([s[l] for s in per_seed])
            res["A"][rid].append(m)
            rows.append([rid, f"L{l}", fmt(m["avg_experts"]), fmt(m["H_set"]), fmt(m["MI_over_H"]),
                         fmt(m["MI_raw_over_H"]), f"{fmt(m['purity'])} ({fmt(m['top_set_share'])})",
                         fmt(m["expert_MI_over_H"]), fmt(m["count_eta2"]), fmt(m["ctx_gain_over_H"], 3)])
        c = res["A"][rid][0]
        print(f"  {rid}: {len(per_seed)} seeds; positions used: token {np.mean(c['cover_tok']):.0%}, "
              f"bigram {np.mean(c['cover_bi']):.0%}")
    table("A. TOKEN DETERMINISM (routing_dump.npz; mean±std over seeds; MI values shuffle-corrected unless 'raw')\n"
          "   MI/H      share of the expert-set entropy explained by the input token id (1 = pure lookup table)\n"
          "   purity    share of occurrences using the token's most common set (in brackets: most common set overall)\n"
          "   expMI/H   same as MI/H but per expert (does expert e fire?), summed over experts; not affected by set size\n"
          "   cnt eta2  share of the variance of the expert COUNT explained by the token id\n"
          "   ctx/H     extra share explained by (previous token, token) over token alone",
          ["run", "layer", "experts", "H(set)", "MI/H", "raw MI/H", "purity (base)", "expMI/H", "cnt eta2", "ctx/H"], rows)
    plot_a(res["A"], Path(a.out) / "A_token_determinism.png")


def plot_a(A, path):
    if not A:
        return
    keys = [("MI_over_H", "MI(set; token) / H(set)"), ("expert_MI_over_H", "per-expert MI / H"),
            ("count_eta2", "count variance explained by token"), ("ctx_gain_over_H", "extra from previous token / H")]
    fig, axs = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.6))
    ids = list(A)
    for ax, (k, t) in zip(axs, keys):
        L = len(A[ids[0]])
        w = 0.8 / L
        for l in range(L):
            m = [np.nanmean(A[i][l][k]) if l < len(A[i]) else np.nan for i in ids]
            s = [np.nanstd(A[i][l][k]) if l < len(A[i]) else np.nan for i in ids]
            ax.bar(np.arange(len(ids)) + (l - (L - 1) / 2) * w, m, w, yerr=s, color=LCOL[l], label=f"L{l}")
        ax.set_xticks(range(len(ids)), ids, rotation=60, ha="right", fontsize=8)
        ax.set_title(t, fontsize=10)
        ax.grid(axis="y", alpha=.3)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ B. early training window
def load_metrics(rd):
    c = cfg(rd)
    mc, tc = c["model"], c["train"]
    mt = [m for m in jl(rd / "metrics.jsonl") if "avg_experts" in m]
    if not mt:
        return None
    arr = lambda k: np.array([m[k] for m in mt], dtype=float)
    return dict(tokens=arr("tokens"), step=arr("step"), lr=arr("lr"), avg=arr("avg_experts"),
                ent=arr("router_entropy"), maxp=arr("max_prob"),
                load=np.array([np.max(m["expert_load"], axis=1) for m in mt]) * mc["n_experts"],
                cap=np.array([np.array(m["count_hist"])[:, mc["max_k"]] for m in mt]),
                floor=np.array([np.array(m["count_hist"])[:, mc["min_k"]] for m in mt]),
                rgn=np.array([m.get("router_grad_norm", [np.nan] * len(m["avg_experts"])) for m in mt], dtype=float),
                peak_lr=tc["lr"], route=mc["route_mode"],
                warmup_tok=tc["warmup_steps"] * tc["batch_size"] * tc["grad_accum"] * mc["seq_len"])


def run_metrics(runs, rid):
    return [m for m in (load_metrics(rd) for rd in seed_dirs(runs, rid)) if m is not None]


def seed_mean(ms):
    n = min(len(m["tokens"]) for m in ms)
    out = {k: np.mean([m[k][:n] for m in ms], axis=0) for k in ("tokens", "step", "lr", "avg", "ent", "maxp", "load",
                                                                "cap", "floor", "rgn")}
    out.update(warmup_tok=ms[0]["warmup_tok"], peak_lr=ms[0]["peak_lr"])
    return out


def dip_stats(m, l, search_tok, early_tok):
    t, a = m["tokens"], m["avg"][:, l]
    s = np.flatnonzero(t <= search_tok)
    if len(s) < 3:
        return None
    i = s[np.argmin(a[s])]
    after = np.flatnonzero((t > t[i]) & (t <= early_tok))
    if len(after) == 0:
        return None
    j = after[np.argmax(a[after])]
    pick = lambda k, idx: float(m[k][idx, l])
    return dict(start=float(a[0]), dip_tok=float(t[i]), dip_step=float(m["step"][i]), lr_frac=float(m["lr"][i] / m["peak_lr"]),
                dip=float(a[i]), peak_tok=float(t[j]), peak=float(a[j]),
                rebound=float((a[j] - a[i]) / max(a[0] - a[i], 1e-9)),
                ent_dip=pick("ent", i), ent_peak=pick("ent", j), maxp_dip=pick("maxp", i), maxp_peak=pick("maxp", j),
                load_dip=pick("load", i), load_peak=pick("load", j), cap_dip=pick("cap", i), cap_peak=pick("cap", j),
                rgn_dip=pick("rgn", i), rgn_peak=pick("rgn", j),
                all_layer_load_at_dip=m["load"][i].tolist(), all_layer_maxp_at_dip=m["maxp"][i].tolist())


def dip_rows(rid, ms, a, res):
    rows = []
    L = ms[0]["avg"].shape[1]
    res["B"][rid] = []
    for l in range(L):
        st = [d for d in (dip_stats(m, l, a.dip_search, a.early) for m in ms) if d is not None]
        if not st:
            continue
        s = mean_dicts(st)
        res["B"][rid].append(s)
        rows.append([rid, f"L{l}", fmt(s["start"]), f"{fmt(np.array(s['dip_tok']) / 1e6)}M", fmt(s["dip_step"], 0),
                     fmt(s["lr_frac"]), fmt(s["dip"]), f"{fmt(np.array(s['peak_tok']) / 1e6)}M", fmt(s["peak"]),
                     fmt(s["rebound"]), f"{fmt(s['ent_dip'])}/{fmt(s['ent_peak'])}",
                     f"{fmt(s['maxp_dip'])}/{fmt(s['maxp_peak'])}", f"{fmt(s['load_dip'])}/{fmt(s['load_peak'])}",
                     f"{fmt(s['cap_dip'])}/{fmt(s['cap_peak'])}", f"{fmt(s['rgn_dip'], 4)}/{fmt(s['rgn_peak'], 4)}"])
    return rows


DIP_HEAD = ["run", "layer", "start", "dip at", "step", "lr/peak", "dip", "peak at", "peak", "rebound",
            "entropy d/p", "top prob d/p", "max load d/p", "at cap d/p", "router gn d/p"]


def section_b(runs, a, res):
    res["B"] = {}
    rows, data = [], {}
    for rid in RUNS_B + LR_SWEEP + AL_SWEEP:
        ms = run_metrics(runs, rid)
        if not ms:
            continue
        data[rid] = ms
        if ms[0]["route"] == "topp":
            rows += dip_rows(rid, ms, a, res)
    table(f"B. EARLY WINDOW: per-layer dip (min avg experts in first {a.dip_search / 1e6:g}M tokens) and rebound peak "
          f"(max after the dip, up to {a.early / 1e6:g}M)\n"
          "   rebound = (peak - dip) / (start - dip); 'd/p' = value at dip / at peak; max load = largest expert share x "
          "n_experts (1 = balanced)\n   mean±std over seeds. Warmup ends at 100 steps (~0.82M tokens at seq 256, batch 32).",
          DIP_HEAD, rows)

    rows = []
    for rid, dips in res["B"].items():
        if len(dips) > 1:
            d1 = dips[1]
            ld = np.mean(d1["all_layer_load_at_dip"], axis=0)
            mp = np.mean(d1["all_layer_maxp_at_dip"], axis=0)
            rows.append([rid, " ".join(f"{x:.2f}" for x in ld), " ".join(f"{x:.2f}" for x in mp)])
    table("B2. ALL LAYERS AT THE MOMENT OF L1's DIP (is L1 uniquely collapsed / sharp there?)",
          ["run", "max load L0..L3", "top prob L0..L3"], rows)

    plot_b([r for r in RUNS_B if r in data], data, a, Path(a.out) / "B_early_window.png", a.early, log=False)
    plot_b([r for r in RUNS_B if r in data], data, a, Path(a.out) / "B_full_training.png", None, log=True)
    plot_sweeps(data, a, Path(a.out) / "B_lr_alpha_sweeps.png")


PANELS = [("avg", "avg experts"), ("ent", "router entropy (nats)"), ("maxp", "mean top prob"),
          ("load", "max expert load x E"), ("cap", "share at max_k"), ("floor", "share at min_k"), ("rgn", "router grad norm")]


def plot_b(ids, data, a, path, xmax, log):
    if not ids:
        return
    fig, axs = plt.subplots(len(PANELS), len(ids), figsize=(3.2 * len(ids), 2.1 * len(PANELS)), squeeze=False,
                            sharex="col")
    for c, rid in enumerate(ids):
        m = seed_mean(data[rid])
        x = m["tokens"] / 1e6
        sel = x <= xmax / 1e6 if xmax else np.ones_like(x, bool)
        for r, (k, t) in enumerate(PANELS):
            ax = axs[r, c]
            for l in range(m[k].shape[1]):
                ax.plot(x[sel], m[k][sel, l], color=LCOL[l], lw=1.2, label=f"L{l}")
            ax.axvline(m["warmup_tok"] / 1e6, color="k", ls=":", lw=.8)
            if log:
                ax.set_xscale("log")
            if r == 0:
                ax.set_title(rid, fontsize=10)
                tw = ax.twinx()
                tw.plot(x[sel], m["lr"][sel], color="0.6", lw=.8, ls="--")
                tw.tick_params(labelsize=6)
            if c == 0:
                ax.set_ylabel(t, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=.3)
        axs[-1, c].set_xlabel("tokens (M)", fontsize=8)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("per-layer routing over training (seed mean; dotted = end of warmup; grey dashed = lr)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sweeps(data, a, path):
    groups = [(g, [r for r in g if r in data]) for g in (LR_SWEEP, AL_SWEEP)]
    groups = [ids for _, ids in groups if ids]
    if not groups:
        return
    L = data[groups[0][0]][0]["avg"].shape[1]
    keys = [("avg", "avg experts"), ("load", "max load x E"), ("maxp", "top prob")]
    fig, axs = plt.subplots(len(keys) * len(groups), L, figsize=(3.3 * L, 2.2 * len(keys) * len(groups)), squeeze=False)
    cm = plt.get_cmap("viridis")
    for gi, ids in enumerate(groups):
        for ki, (k, t) in enumerate(keys):
            r = gi * len(keys) + ki
            for l in range(L):
                ax = axs[r, l]
                for j, rid in enumerate(ids):
                    m = seed_mean(data[rid])
                    x = m["tokens"] / 1e6
                    sel = x <= a.early / 1e6
                    ax.plot(x[sel], m[k][sel, l], color=cm(j / max(len(ids) - 1, 1)), lw=1.2, label=rid)
                ax.axvline(data[ids[0]][0]["warmup_tok"] / 1e6, color="k", ls=":", lw=.8)
                ax.grid(alpha=.3)
                ax.tick_params(labelsize=7)
                if r == 0:
                    ax.set_title(f"L{l}", fontsize=10)
                if l == 0:
                    ax.set_ylabel(t, fontsize=8)
                    if ki == 0:
                        ax.legend(fontsize=6)
    for l in range(L):
        axs[-1, l].set_xlabel("tokens (M)", fontsize=8)
    fig.suptitle("LR sweep (top block) and load-balance alpha sweep (bottom block), first "
                 f"{a.early / 1e6:g}M tokens", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ C. censoring + final per-layer table
def final_routing(rd):
    ev = [e for e in jl(rd / "eval.jsonl") if "routing" in e]
    fin = [e for e in ev if e.get("final")]
    return (fin or ev or [None])[-1]


def section_c(runs, a, res):
    res["C"] = {}
    rows, hist_rows, spread = [], [], {}
    for rid in RUNS_C:
        sds = seed_dirs(runs, rid)
        fins = [(cfg(rd)["model"], final_routing(rd)) for rd in sds]
        fins = [(mc, f) for mc, f in fins if f is not None]
        if not fins:
            continue
        mc = fins[0][0]
        lo, hi, E = mc["min_k"], mc["max_k"], mc["n_experts"]
        L = len(fins[0][1]["routing"]["avg_experts"])
        res["C"][rid] = []
        for l in range(L):
            h = np.array([f["routing"]["count_hist"][l] for _, f in fins])  # [seeds, E+1]
            c = np.arange(h.shape[1])
            unc = h.copy()
            unc[:, hi:] = 0
            s = dict(avg=[f["routing"]["avg_experts"][l] for _, f in fins],
                     maxp=[f["routing"]["max_prob"][l] for _, f in fins],
                     ent=[f["routing"]["entropy"][l] for _, f in fins],
                     floor=h[:, lo].tolist(), cap=h[:, hi].tolist(),
                     uncapped_mean=((unc * c).sum(1) / np.maximum(unc.sum(1), 1e-9)).tolist(),
                     load=[max(f["routing"]["load"][l]) * E for _, f in fins], hist=h.mean(0).tolist())
            res["C"][rid].append(s)
            rows.append([rid, f"L{l}", fmt(s["avg"]), fmt(s["maxp"]), fmt(s["ent"]), fmt(s["floor"]), fmt(s["cap"]),
                         fmt(s["uncapped_mean"]), fmt(s["load"])])
            hist_rows.append([rid, f"L{l}"] + [f"{x:.3f}" for x in h.mean(0)[lo:hi + 1]])
        spread[rid] = np.ptp([np.mean(s["avg"]) for s in res["C"][rid]])
    table("C. FINAL PER-LAYER ROUTING (eval.jsonl final record; mean±std over seeds)\n"
          "   at min_k / at max_k = share of tokens on the floor (1 expert) / at the cap (4); a layer mostly on the floor\n"
          "   is censored from below (it would use <1 if it could), one often at the cap is censored from above",
          ["run", "layer", "experts", "top prob", "entropy", "at min_k", "at max_k", "mean if uncapped", "max load xE"], rows)
    table("C2. EXPERT-COUNT HISTOGRAM (share of tokens using 1..max_k experts)",
          ["run", "layer"] + [f"{c}" for c in range(1, 5)], hist_rows)
    table("C3. SPREAD BETWEEN LAYERS (max - min avg experts): raw vs renorm gates, with vs without entropy loss",
          ["run", "spread"], [[k, f"{v:.2f}"] for k, v in spread.items()])


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/probe_layers")
    ap.add_argument("--min-occ", type=int, default=20, help="min occurrences of a token/bigram in the dump to be used")
    ap.add_argument("--n-perm", type=int, default=3, help="shuffles for the MI bias correction")
    ap.add_argument("--dip-search", type=float, default=1.5e6, help="look for the dip in the first N tokens")
    ap.add_argument("--early", type=float, default=3e6, help="early window length (tokens) for the rebound and plots")
    ap.add_argument("--skip", default="", help="sections to skip, e.g. 'A' (A is the slowest)")
    a = ap.parse_args()
    runs = Path(a.runs)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    res = {"args": vars(a)}
    np.set_printoptions(precision=3, suppress=True)
    if "C" not in a.skip:
        section_c(runs, a, res)
    if "B" not in a.skip:
        section_b(runs, a, res)
    if "A" not in a.skip:
        print("\nA. computing token determinism ...")
        section_a(runs, a, rng, res)
    (Path(a.out) / "probe_layers.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\nfigures and probe_layers.json written to {a.out}/")


if __name__ == "__main__":
    main()
