#!/usr/bin/env python3
"""analyze_tokens.py - token-level analysis of top-p routing (no training needed).

    python analysis/analyze_tokens.py --topp P30 P50 P60 P70 P50-noent P50-renorm --topk K2 --out results/tokens_10m
    python analysis/analyze_tokens.py --topp L-P60 L-P65 L-P70 --topk L-K2 --out results/tokens_30m   (30M runs)

The dataset comes from the --topk run's config and the tokenizer from that dataset's meta.json.

Uses runs/<ID>/seed<N>/routing_dump.npz (saved at the end of every MoE run): for the same fixed validation tokens it
holds each token's loss and which experts each layer used. Every model sees the same tokens in the same order, so
per-token losses of different models can be compared directly.

Questions answered:
  1. Do tokens that are harder (higher loss) get more experts?              (Spearman correlation, loss deciles)
  2. Is that just because hard tokens are rare tokens?                      (token frequency; partial correlation)
  3. Which tokens get the fewest and the most experts?                      (token table)
  4. Does the expert count depend on position, or on being the first token of a story?
  5. Where does top-p gain over top-k?  Per-token loss of the top-k model minus top-p model, grouped by an
     independent difficulty measure (the TOP-K model's own loss, so the grouping is not chosen by the top-p model),
     by token frequency, and by how many experts top-p used. Bootstrap 95% intervals over windows.
Only numpy and matplotlib are needed (the tokenizer package is optional, for readable token strings).
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rankdata(x):
    """Average ranks (ties share the mean rank), 1-based."""
    x = np.asarray(x)
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    hi = np.cumsum(cnt)
    return ((hi - cnt + 1) + hi)[inv] / 2.0


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def partial_spearman(x, y, controls):
    """Spearman correlation of x and y after removing what the control variables explain (rank-based, linear)."""
    Z = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in np.atleast_2d(controls)])
    def resid(v):
        r = rankdata(v)
        return r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]
    a, b = resid(x), resid(y)
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / d) if d > 1e-12 else float("nan")


def bootstrap_mean_ci(sum_per_window, n_per_window, n_boot=2000, seed=0):
    """95% interval for sum/count when whole windows are resampled (tokens in a window are not independent)."""
    rng = np.random.default_rng(seed)
    W = len(sum_per_window)
    idx = rng.integers(0, W, size=(n_boot, W))
    s, n = sum_per_window[idx].sum(1), n_per_window[idx].sum(1)
    ok = n > 0
    m = s[ok] / n[ok]
    return float(sum_per_window.sum() / max(n_per_window.sum(), 1)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def bootstrap_diff_ci(s1, n1, s2, n2, n_boot=2000, seed=0):
    """95% interval for (sum1/n1) - (sum2/n2) when whole windows are resampled together."""
    rng = np.random.default_rng(seed)
    W = len(s1)
    idx = rng.integers(0, W, size=(n_boot, W))
    a = s1[idx].sum(1) / np.maximum(n1[idx].sum(1), 1)
    b = s2[idx].sum(1) / np.maximum(n2[idx].sum(1), 1)
    d = a - b
    m = s1.sum() / max(n1.sum(), 1) - s2.sum() / max(n2.sum(), 1)
    return float(m), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def load_dump(run_dir):
    d = np.load(run_dir / "routing_dump.npz")
    mask = d["expert_mask"]  # [layers, tokens, experts] bool
    return {"tokens": d["tokens"].astype(np.int64), "loss": d["token_loss"].astype(np.float64),
            "counts": mask.sum(-1).astype(np.float64),  # [layers, tokens] experts used per token per layer
            "seq_len": int(d["seq_len"]), "n_experts": int(d["n_experts"])}


def load_runs(runs_dir, ids):
    out = {}
    for i in ids:
        for rd in sorted(Path(runs_dir).glob(f"{i}/seed*")):
            if (rd / "routing_dump.npz").exists():
                out.setdefault(i, {})[int(rd.name[4:])] = load_dump(rd)
    return out


def token_names(tokenizer_path, vocab):
    if not tokenizer_path or not Path(tokenizer_path).exists():
        return [f"<{i}>" for i in range(vocab)]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_path))
    return [(tok.id_to_token(i) or f"<{i}>").replace("Ġ", "_").replace("Ċ", "\\n") for i in range(vocab)]


def bin_by_quantile(values, n):
    """Group index 0..n-1 by quantile (ties broken by rank so groups are equal-sized)."""
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return np.minimum((order * n) // len(values), n - 1)


# ------------------------------------------------------------------ analyses
def analyse(runs, topp_ids, topk_id, freq, names, eos_id, min_count, out):
    ref = runs.get(topk_id, {})
    results, fig_data = {}, {}
    for i in topp_ids:
        if i not in runs:
            print(f"{i}: no routing dumps found, skipping"); continue
        seeds = sorted(runs[i])
        print(f"\n================ {i}  ({len(seeds)} seeds: {seeds}) ================")
        L = runs[i][seeds[0]]["counts"].shape[0]
        # ---- 1 and 2: difficulty and frequency
        # The router sees the CURRENT token; the loss is for predicting the NEXT token. So rarity is measured for both
        # (the last position of each window has no known next token and is left out of these correlations).
        print("\n[1-2] Spearman correlation (rho) between the experts used at a token and how hard the next token is to predict")
        print("      rho > 0: harder positions get more experts. 'rarity' = -log(how often the token occurs in the training data).")
        print(f"{'seed':<6}{'rho(experts,loss)':<20}{'rho(experts,own rarity)':<26}{'rho(loss,next rarity)':<24}{'partial rho(experts,loss | both rarities)'}")
        rows = []
        for s in seeds:
            r = runs[i][s]
            T_ = r["seq_len"]; v = (np.arange(len(r["tokens"])) % T_) != T_ - 1
            tgt = np.roll(r["tokens"], -1)
            tot = r["counts"].mean(0)[v]; ls = r["loss"][v]
            own = -np.log(freq[r["tokens"][v]] + 1); nxt = -np.log(freq[tgt[v]] + 1)
            rows.append((spearman(tot, ls), spearman(tot, own), spearman(ls, nxt), partial_spearman(tot, ls, [own, nxt])))
            print(f"{s:<6}{rows[-1][0]:<20.3f}{rows[-1][1]:<26.3f}{rows[-1][2]:<24.3f}{rows[-1][3]:.3f}")
        rows = np.array(rows)
        results[i] = {"rho_loss": rows[:, 0].tolist(), "rho_own_rarity": rows[:, 1].tolist(), "partial": rows[:, 3].tolist()}
        print("      per layer, rho(experts in that layer, loss), mean over seeds: " +
              "  ".join(f"L{l}=" + (lambda v_: "n/a (constant)" if np.isnan(v_) else f"{v_:+.3f}")(np.nanmean([spearman(runs[i][s]['counts'][l], runs[i][s]['loss']) for s in seeds])) for l in range(L)))
        # deciles of loss -> mean experts per layer (pooled over seeds)
        dec = np.zeros((10, L)); dfreq = np.zeros((10, L))
        for s in seeds:
            r = runs[i][s]
            g = bin_by_quantile(r["loss"], 10); gf = bin_by_quantile(-np.log(freq[r["tokens"]] + 1), 10)  # rarity of the token the router sees
            for k in range(10):
                dec[k] += r["counts"][:, g == k].mean(1) / len(seeds)
                dfreq[k] += r["counts"][:, gf == k].mean(1) / len(seeds)
        # ---- 3: token table
        allt = np.concatenate([runs[i][s]["tokens"] for s in seeds]); allc = np.concatenate([runs[i][s]["counts"].mean(0) for s in seeds])
        alll = np.concatenate([runs[i][s]["loss"] for s in seeds])
        V = len(freq); cnt = np.bincount(allt, minlength=V); sm = np.bincount(allt, weights=allc, minlength=V); sl = np.bincount(allt, weights=alll, minlength=V)
        ok = cnt >= min_count * len(seeds)
        mean_e = np.where(ok, sm / np.maximum(cnt, 1), np.nan)
        order = np.argsort(np.where(ok, mean_e, np.inf)); order_hi = np.argsort(np.where(ok, -mean_e, np.inf))
        print(f"\n[3] INPUT tokens (the token the router sees) with the FEWEST / MOST experts on average (seen at least {min_count} times per seed); loss = loss of predicting the token after it")
        print(f"{'fewest':<16}{'experts':<9}{'loss':<8}{'n':<8}| {'most':<16}{'experts':<9}{'loss':<8}n")
        for a, b in zip(order[:15], order_hi[:15]):
            if not ok[a] or not ok[b]:
                continue
            print(f"{names[a]:<16}{mean_e[a]:<9.2f}{sl[a] / cnt[a]:<8.2f}{cnt[a] // len(seeds):<8}| {names[b]:<16}{mean_e[b]:<9.2f}{sl[b] / cnt[b]:<8.2f}{cnt[b] // len(seeds)}")
        # ---- 4: position and story start
        T = runs[i][seeds[0]]["seq_len"]
        edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        edges = [e for e in edges if e < T] + [T]
        pos_means = []
        pos = np.arange(len(runs[i][seeds[0]]["tokens"])) % T
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (pos >= lo) & (pos < hi)
            pos_means.append(np.mean([runs[i][s]["counts"][:, sel].mean() for s in seeds]))
        print(f"\n[4] mean experts per token by position in the {T}-token window: " +
              "  ".join(f"[{lo},{hi}):{m:.2f}" for (lo, hi), m in zip(zip(edges[:-1], edges[1:]), pos_means)))
        if eos_id is not None:
            after = np.concatenate([[False], runs[i][seeds[0]]["tokens"][:-1] == eos_id])
            after &= pos != 0
            if after.sum() > 5:
                a_ = np.mean([runs[i][s]["counts"][:, after].mean() for s in seeds]); b_ = np.mean([runs[i][s]["counts"][:, ~after].mean() for s in seeds])
                print(f"    first token of a story/article (just after the end-of-document token, n={int(after.sum())}): {a_:.2f} experts vs {b_:.2f} for all other tokens")
        # ---- 5: where does top-p gain over top-k?
        gain = None
        if ref:
            refloss = np.mean([ref[s]["loss"] for s in sorted(ref)], axis=0)  # top-k loss averaged over its seeds
            tok0 = ref[sorted(ref)[0]]["tokens"]
            same = all((runs[i][s]["tokens"] == tok0).all() for s in seeds)
            if same:
                print(f"\n[5] where does {i} beat {topk_id}?  gain = {topk_id} loss minus {i} loss per token (positive = top-p better);"
                      f" {topk_id} loss averaged over its {len(ref)} seeds; groups are chosen WITHOUT using {i}'s own loss; 95% interval over windows")
                gain = {}
                groups = {
                    f"{topk_id} difficulty (loss tercile)": (bin_by_quantile(refloss, 3), ["easiest 1/3", "middle 1/3", "hardest 1/3"]),
                    "rarity of the token being predicted (tercile)": (bin_by_quantile(-np.log(freq[np.roll(tok0, -1)] + 1), 3), ["most common", "middle", "rarest"]),
                }
                W = len(tok0) // T
                valid = (np.arange(len(tok0)) % T) != T - 1  # last position of a window: next token not in the dump
                for gname, (g, labels) in groups.items():
                    print(f"  by {gname}")
                    for k, lab in enumerate(labels):
                        s_w = np.zeros(W); n_w = np.zeros(W)
                        for s in seeds:
                            d = refloss - runs[i][s]["loss"]
                            sel = (g == k) & valid
                            s_w += np.bincount(np.arange(len(tok0))[sel] // T, weights=d[sel], minlength=W)
                            n_w += np.bincount(np.arange(len(tok0))[sel] // T, minlength=W)
                        m, lo, hi = bootstrap_mean_ci(s_w, n_w)
                        gain[(gname, lab)] = (m, lo, hi)
                        print(f"    {lab:<16}{m:+.4f}   [{lo:+.4f}, {hi:+.4f}]")
                # Control for regression to the mean: tokens grouped as "hard" by one model's loss include tokens where that
                # model was unlucky, so ANY other model improves on them. Measure the same thing for the top-k model itself:
                # group by the mean loss of the OTHER top-k seeds and compare with the held-out seed. This control is a bit
                # noisier than the real comparison (2 seeds in the grouping instead of 3), so it is conservative for top-p.
                if len(ref) >= 3:
                    rs = sorted(ref)
                    g_all = {k: [np.zeros(W), np.zeros(W)] for k in range(3)}
                    for held in rs:
                        others = np.mean([ref[o]["loss"] for o in rs if o != held], axis=0)
                        g = bin_by_quantile(others, 3); d = others - ref[held]["loss"]
                        for k in range(3):
                            sel = (g == k) & valid
                            g_all[k][0] += np.bincount(np.arange(len(tok0))[sel] // T, weights=d[sel], minlength=W)
                            g_all[k][1] += np.bincount(np.arange(len(tok0))[sel] // T, minlength=W)
                    print(f"  CONTROL: {topk_id} against itself (group by the other {topk_id} seeds, compare with the held-out seed). "
                          f"This is the gain you would see by chance. 'corrected' = {i} gain minus this control.")
                    for k, lab in enumerate(["easiest 1/3", "middle 1/3", "hardest 1/3"]):
                        # the real gain for the same group, as sums per window, to resample together with the control
                        g_real = bin_by_quantile(refloss, 3)
                        s_w = np.zeros(W); n_w = np.zeros(W)
                        for s in seeds:
                            d = refloss - runs[i][s]["loss"]; sel = (g_real == k) & valid
                            s_w += np.bincount(np.arange(len(tok0))[sel] // T, weights=d[sel], minlength=W)
                            n_w += np.bincount(np.arange(len(tok0))[sel] // T, minlength=W)
                        mc, loc, hic = bootstrap_mean_ci(g_all[k][0], g_all[k][1])
                        mcorr, lo, hi = bootstrap_diff_ci(s_w, n_w, g_all[k][0], g_all[k][1])
                        gain[("control", lab)] = (mc, loc, hic)
                        gain[("corrected", lab)] = (mcorr, lo, hi)
                        print(f"    {lab:<14}control {mc:+.4f}   corrected {mcorr:+.4f}   [{lo:+.4f}, {hi:+.4f}]")
                # by the number of experts the top-p model used (its own routing; the loss difference is still per token)
                print(f"  by experts used by {i} (average over layers, tercile within each seed)")
                for k, lab in enumerate(["fewest experts", "middle", "most experts"]):
                    s_w = np.zeros(W); n_w = np.zeros(W)
                    for s in seeds:
                        g = bin_by_quantile(runs[i][s]["counts"].mean(0) + 1e-6 * np.arange(len(tok0)) / len(tok0), 3)
                        d = refloss - runs[i][s]["loss"]; sel = (g == k) & valid
                        s_w += np.bincount(np.arange(len(tok0))[sel] // T, weights=d[sel], minlength=W)
                        n_w += np.bincount(np.arange(len(tok0))[sel] // T, minlength=W)
                    m, lo, hi = bootstrap_mean_ci(s_w, n_w)
                    gain[("experts", lab)] = (m, lo, hi)
                    print(f"    {lab:<16}{m:+.4f}   [{lo:+.4f}, {hi:+.4f}]")
                overall = np.mean([np.mean(refloss - runs[i][s]["loss"]) for s in seeds])
                print(f"  overall mean gain: {overall:+.4f}")
            else:
                print(f"\n[5] skipped: {i} and {topk_id} were evaluated on different tokens")
        fig_data[i] = dict(dec=dec, dfreq=dfreq, pos=pos_means, edges=edges, gain=gain)
    return results, fig_data


def figure(fig_data, topk_id, out):
    ids = list(fig_data)
    if not ids:
        return
    fig, axs = plt.subplots(len(ids), 4, figsize=(17, 3.6 * len(ids)), squeeze=False)
    for row, i in zip(axs, ids):
        d = fig_data[i]
        for l in range(d["dec"].shape[1]):
            row[0].plot(range(1, 11), d["dec"][:, l], "o-", ms=3, label=f"layer {l}")
        row[0].plot(range(1, 11), d["dec"].mean(1), "k-", lw=2, label="mean")
        row[0].set_title(f"{i}: experts vs token loss"); row[0].set_xlabel("loss decile (1 = easiest, 10 = hardest)"); row[0].set_ylabel("experts used")
        row[0].legend(fontsize=7)
        for l in range(d["dfreq"].shape[1]):
            row[1].plot(range(1, 11), d["dfreq"][:, l], "o-", ms=3)
        row[1].plot(range(1, 11), d["dfreq"].mean(1), "k-", lw=2)
        row[1].set_title(f"{i}: experts vs token rarity"); row[1].set_xlabel("rarity decile (1 = most common, 10 = rarest)")
        row[2].bar(range(len(d["pos"])), d["pos"]); row[2].set_xticks(range(len(d["pos"])))
        row[2].set_xticklabels([f"{a}" for a in d["edges"][:-1]], fontsize=7); row[2].set_xlabel("position in window (start)"); row[2].set_title(f"{i}: experts vs position")
        if d["gain"]:
            keys = list(d["gain"]); m = np.array([d["gain"][k] for k in keys])
            y = np.arange(len(keys))[::-1]
            row[3].errorbar(m[:, 0], y, xerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt="o", capsize=3)
            row[3].set_yticks(y); row[3].set_yticklabels([("control: " if k[0] == "control" else "CORRECTED: " if k[0] == "corrected" else "raw (biased): " if k[0].endswith("difficulty (loss tercile)") else "") + f"{k[1]}" for k in keys], fontsize=7); row[3].axvline(0, color="gray", ls=":")
            row[3].set_title(f"{topk_id} loss minus {i} loss (positive = {i} better)", fontsize=9); row[3].set_xlabel("mean per-token gain, 95% interval")
    fig.tight_layout(); fig.savefig(out / "token_analysis.png", dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs"); ap.add_argument("--out", help="default: results/tokens_<dataset folder>")
    ap.add_argument("--data", help="folder with train.bin and meta.json (for token frequencies); default: the --topk run's data_dir")
    ap.add_argument("--tokenizer", help="default: the tokenizer recorded in <data>/meta.json")
    ap.add_argument("--topp", nargs="+", default=["P60", "P50", "P70"]); ap.add_argument("--topk", default="K2")
    ap.add_argument("--min-count", type=int, default=30, help="a token needs this many occurrences per seed to appear in the token table")
    a = ap.parse_args()
    if a.data is None:
        cfgs = sorted(Path(a.runs).glob(f"{a.topk}/seed*/config.json"))
        a.data = json.loads(cfgs[0].read_text())["train"]["data_dir"] if cfgs else "data/tinystories"
    meta = json.loads((Path(a.data) / "meta.json").read_text())
    a.tokenizer = a.tokenizer or meta.get("tokenizer")
    dataset = Path(a.data).name
    out = Path(a.out or f"results/tokens_{dataset}"); out.mkdir(parents=True, exist_ok=True)
    print(f"data {a.data}, tokenizer {a.tokenizer}, output {out}")
    V = meta["vocab_size"]
    freq = np.bincount(np.fromfile(Path(a.data) / "train.bin", dtype=np.uint16), minlength=V).astype(np.float64)
    runs = load_runs(a.runs, a.topp + [a.topk])
    if not any(i in runs for i in a.topp):
        print("no routing dumps found for", a.topp); return 1
    results, fig_data = analyse(runs, a.topp, a.topk, freq, token_names(a.tokenizer, V), meta.get("eos_id"), a.min_count, out)
    figure(fig_data, a.topk, out)
    (out / "token_analysis.json").write_text(json.dumps(results, indent=2))
    print(f"\nfigure: {out / 'token_analysis.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())