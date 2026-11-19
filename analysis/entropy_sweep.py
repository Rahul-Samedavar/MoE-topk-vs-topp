"""Entropy-weight (beta) sweep for top-p p=0.5 at 10M tokens: beta in {0, 1e-4, 1e-3, 1e-2}, seeds 0-1.

    python analysis/entropy_sweep.py [--runs runs]
Uses the same definitions as summary_stats.py (paired diffs, FLOPs frontier) and analyze.py (log-spaced E frontier).
"""
import argparse, json, math
from pathlib import Path
import numpy as np

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--runs", default="runs")
R = Path(_ap.parse_args().runs)
SWEEP = [("P50-noent", 0.0), ("P50", 1e-4), ("P50-ent1e-3", 1e-3), ("P50-ent1e-2", 1e-2)]
STEPS = [0, 100, 200, 300, 400, 600, 800, 1000, 1200, "final"]
DEAD = 0.00125  # share of assignments below which an expert counts as dead


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []
def seeds(rid):
    return sorted(int(d.name[4:]) for d in (R / rid).glob("seed*") if (d / "summary.json").exists())
def S(rid, s): return json.loads((R / rid / f"seed{s}" / "summary.json").read_text())
def ev(rid, s): return jl(R / rid / f"seed{s}" / "eval.jsonl")
def fin(rid, s): return [r for r in ev(rid, s) if r.get("final")][-1]
def cfg(rid, s): return json.loads((R / rid / f"seed{s}" / "config.json").read_text())
def sweep(rid, s):
    return {r["label"]: r for r in json.loads((R / rid / f"seed{s}" / "sweep_eval.json").read_text())["rows"]}


def find(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                return v
            r = find(v, key)
            if r is not None:
                return r


print("=== 0. CONFIG CHECK (everything except ent_coef must match P50) ===")
ref = cfg("P50", 0)
for rid, b in SWEEP:
    for s in seeds(rid):
        c = cfg(rid, s)
        diffs = []
        for sec in ("model", "train"):
            for k in set(ref.get(sec, {})) | set(c.get(sec, {})):
                if k in ("seed", "out_dir", "run_name"):
                    continue
                if ref.get(sec, {}).get(k) != c.get(sec, {}).get(k):
                    diffs.append(f"{sec}.{k}: {ref.get(sec, {}).get(k)} -> {c.get(sec, {}).get(k)}")
        print(f"{rid}/s{s}: ent_coef={find(c, 'ent_coef')} started={find(c, 'started')} resumes={S(rid, s)['n_resumes']} diffs vs P50/s0: {diffs or 'none'}")

print("\n=== 1. PER-RUN FINAL (full val) ===")
for rid, b in SWEEP:
    for s in seeds(rid):
        x = S(rid, s)
        print(f"{rid:<12} beta={b:<7g} s{s}: loss {x['val_loss']:.4f} ppl {x['val_ppl']:.3f} bpb {x['val_bpb']:.4f} E {x['avg_experts_mean']:.3f} "
              f"layers {[round(v, 3) for v in x['avg_experts_by_layer']]} effFLOPs {x['flops_per_token_effective_fwd']/1e6:.3f}M "
              f"train_lm {x['final_train_lm_loss']:.4f} val-train {x['val_loss']-x['final_train_lm_loss']:+.4f} wall {x['wall_time_s']:.0f}s tok/s {x['mean_tokens_per_s']:.0f}")

print("\n=== 2. SUMMARY PER BETA (mean, std ddof1, spread) ===")
for rid, b in SWEEP:
    L = [S(rid, s)["val_loss"] for s in seeds(rid)]
    E = [S(rid, s)["avg_experts_mean"] for s in seeds(rid)]
    F = [S(rid, s)["flops_per_token_effective_fwd"] for s in seeds(rid)]
    lay = np.mean([S(rid, s)["avg_experts_by_layer"] for s in seeds(rid)], 0)
    vt = np.mean([S(rid, s)["val_loss"] - S(rid, s)["final_train_lm_loss"] for s in seeds(rid)])
    print(f"beta={b:<7g} {rid:<12} n={len(L)} mean {np.mean(L):.4f} std {np.std(L, ddof=1):.4f} spread {max(L)-min(L):.4f} "
          f"ppl {math.exp(np.mean(L)):.2f} | E {np.mean(E):.3f} spread {max(E)-min(E):.3f} layers {[round(v, 2) for v in lay]} "
          f"| FLOPs {np.mean(F)/1e6:.2f}M | val-train {vt:+.4f}")

print("\n=== 3. PAIRED SAME-SEED DIFFERENCES (A - B; negative = A better) ===")
for a, b in [("P50-ent1e-3", "P50"), ("P50-ent1e-2", "P50"), ("P50-ent1e-3", "P50-noent"), ("P50-ent1e-2", "P50-noent"),
             ("P50-ent1e-2", "P50-ent1e-3"), ("P50-noent", "P50"), ("P50-ent1e-3", "K2"), ("P50-ent1e-2", "K2"), ("P50-ent1e-2", "K1")]:
    common = sorted(set(seeds(a)) & set(seeds(b)))
    d = [S(a, s)["val_loss"] - S(b, s)["val_loss"] for s in common]
    dE = [S(a, s)["avg_experts_mean"] - (S(b, s)["avg_experts_mean"] or 0) for s in common]
    print(f"{a:>12} - {b:<12} " + ", ".join(f"s{s}={x:+.4f}" for s, x in zip(common, d))
          + f" | mean {np.mean(d):+.4f} | dE " + ", ".join(f"{x:+.3f}" for x in dE))

print("\n=== 4. GAP TO THE TOP-K FRONTIER (positive = top-p better) ===")
kx = [np.mean([S(k, s)["flops_per_token_effective_fwd"] for s in seeds(k)]) for k in ("K1", "K2", "K3")]
ky = [np.mean([S(k, s)["val_loss"] for s in seeds(k)]) for k in ("K1", "K2", "K3")]
def topk_E(seed, x, log):
    pts = []
    for kid in ("K1", "K2", "K3"):
        if seed in seeds(kid):
            pts.append((S(kid, seed)["avg_experts_mean"] or {"K1": 1, "K2": 2, "K3": 3}[kid], S(kid, seed)["val_loss"]))
        else:
            pts.append(({"K1": 1, "K2": 2, "K3": 3}[kid], np.mean([S(kid, t)["val_loss"] for t in seeds(kid)])))
    pts.sort()
    f = np.log if log else (lambda v: v)
    return float(np.interp(f(x), [f(p[0]) for p in pts], [p[1] for p in pts]))
for rid, b in SWEEP:
    rows = []
    for s in seeds(rid):
        x = S(rid, s); e = x["avg_experts_mean"]; y = x["val_loss"]
        rows.append((s, e, np.interp(x["flops_per_token_effective_fwd"], kx, ky) - y, topk_E(s, e, True) - y, topk_E(s, e, False) - y))
    print(f"beta={b:<7g} {rid:<12} " + "; ".join(f"s{s} E {e:.3f}: FLOPs-seedmean {g1:+.4f}  logE-sameseed {g2:+.4f}  linE-sameseed {g3:+.4f}" for s, e, g1, g2, g3 in rows)
          + f" | mean logE {np.mean([r[3] for r in rows]):+.4f}")

print("\n=== 5. FINAL ROUTING PER LAYER (full val) ===")
print("max/E = largest share x 8; dead = experts with share < 0.125%; at1 = share of tokens at 1 expert; atcap = share at 4")
for rid, b in SWEEP:
    for s in seeds(rid):
        f = fin(rid, s); ro = f["routing"]; load = np.array(ro["load"]); ch = np.array(ro["count_hist"])
        print(f"beta={b:<7g} {rid}/s{s}: ent {[round(v, 3) for v in ro['entropy']]} top {[round(v, 3) for v in ro['max_prob']]} "
              f"max/E {[round(v, 2) for v in load.max(1)*8]} min/E {[round(v, 3) for v in load.min(1)*8]} dead {(load < DEAD).sum(1).tolist()} "
              f"L0 top-share {load[0].max():.3f} at1 {[round(v, 3) for v in ch[:, 1]]} atcap {[round(v, 3) for v in ch[:, 4]]} "
              f"| val_bal {f['val_bal_loss']:.4f} val_ent {f['val_ent_loss']:.4f} val_z {f['val_z_loss']:.3f}")
print("-- count histogram (share at 1/2/3/4 experts), seed mean --")
for rid, b in SWEEP:
    ch = np.mean([np.array(fin(rid, s)["routing"]["count_hist"]) for s in seeds(rid)], 0)
    print(f"beta={b:<7g} {rid:<12} " + " | ".join(f"L{l} {' '.join(f'{v:.3f}' for v in ch[l, 1:5])}" for l in range(4)))

print("\n=== 6. E AND VAL LOSS OVER TRAINING (seed mean; intermediate on 262k tokens, final on full val) ===")
for what in ("avg_experts_mean", "val_loss"):
    print(f"-- {what} --  " + " ".join(f"{str(x):>7}" for x in STEPS))
    for rid, b in SWEEP:
        by = {}
        for s in seeds(rid):
            for r in ev(rid, s):
                by.setdefault("final" if r.get("final") else r["step"], []).append(r[what])
        print(f"beta={b:<7g}" + " ".join(f"{np.mean(by[x]):7.4f}" if x in by else "      -" for x in STEPS))

print("\n=== 7. INFERENCE SWEEP (first 131,072 val tokens; seed mean loss [E]) ===")
labels = ["as_trained", "topp_p0.3", "topp_p0.5", "topp_p0.7", "topp_p0.9", "topk_k1", "topk_k2", "topk_k3", "topk_k4"]
for rid, b in SWEEP:
    sw = [sweep(rid, s) for s in seeds(rid)]
    print(f"beta={b:<7g} {rid:<12} " + "  ".join(f"{l.replace('topp_', '').replace('topk_', '')}={np.mean([w[l]['val_loss'] for w in sw]):.4f}[{np.mean([w[l]['avg_experts_mean'] for w in sw]):.2f}]" for l in labels))
    for s, w in zip(seeds(rid), sw):
        print(f"      s{s}: " + "  ".join(f"{l.replace('topp_', '').replace('topk_', '')}={w[l]['val_loss']:.4f}[{w[l]['avg_experts_mean']:.2f}]" for l in labels))
print("-- weights test: top-p model run as top-2 minus K2 as trained (per seed) --")
for rid, b in SWEEP:
    d = [sweep(rid, s)["topk_k2"]["val_loss"] - sweep("K2", s)["as_trained"]["val_loss"] for s in seeds(rid) if s in seeds("K2")]
    print(f"beta={b:<7g} {rid:<12} " + ", ".join(f"{x:+.4f}" for x in d) + f" | mean {np.mean(d):+.4f}")

print("\n=== 8. HEALTH ===")
for rid, b in SWEEP:
    for s in seeds(rid):
        m = jl(R / rid / f"seed{s}" / "metrics.jsonl")
        lm = np.array([r["lm_loss"] for r in m]); gn = np.array([r["grad_norm"] for r in m])
        rg = [r["router_grad_norm"] for r in m if r.get("router_grad_norm")]
        print(f"beta={b:<7g} {rid}/s{s}: nonfinite {int(np.sum(~np.isfinite(lm)))} grad_norm max {gn.max():.2f} last {gn[-1]:.3f} "
              f"final train ent {m[-1].get('ent_loss', float('nan')):.4f} router_gn last {[round(v, 4) for v in rg[-1]] if rg else '-'}")
