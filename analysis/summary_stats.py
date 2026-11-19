"""summary_stats.py - detailed statistics of the finished runs, from their log files.

    python analysis/summary_stats.py [--runs runs]

Paired same-seed differences, seed spreads, compute-matched gaps, learning curves, final routing and load, training
health, train-vs-val gaps, the inference-time sweep cross tables, throughput, and a step-0 sanity check.
"""
import argparse, json, glob, math, sys
from pathlib import Path
import numpy as np

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--runs", default="runs")
R = Path(_ap.parse_args().runs)
def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []
def seeds(rid):
    return sorted(int(d.name[4:]) for d in (R / rid).glob("seed*") if (d / "summary.json").exists())
def S(rid, s): return json.loads((R / rid / f"seed{s}" / "summary.json").read_text())
def ev(rid, s): return jl(R / rid / f"seed{s}" / "eval.jsonl")
def mt(rid, s): return jl(R / rid / f"seed{s}" / "metrics.jsonl")

MAIN = ["K1", "K2", "K3", "P30", "P50", "P60", "P70", "K2-ent", "P50-noent", "DENSE", "K2-renorm", "P50-renorm"]
L30 = ["L-K2", "L-P60", "L-P65", "L-P70", "L-DENSE"]

print("=== 1. PAIRED SAME-SEED DIFFERENCES (A - B final val loss; negative = A better; same seed = same init seed and data order) ===")
pairs = [("P60", "K2"), ("P70", "K2"), ("P50", "K2"), ("P30", "K2"), ("K3", "K2"), ("K1", "K2"), ("DENSE", "K2"),
         ("K2-ent", "K2"), ("P50-noent", "P50"), ("K2-renorm", "K2"), ("P50-renorm", "P50"), ("P60", "DENSE"), ("P70", "K3"),
         ("L-P60", "L-K2"), ("L-P65", "L-K2"), ("L-P70", "L-K2"), ("L-DENSE", "L-K2"), ("L-P65", "L-P60"), ("L-P60", "L-DENSE"), ("L-P65", "L-DENSE")]
for a, b in pairs:
    common = sorted(set(seeds(a)) & set(seeds(b)))
    d = [S(a, s)["val_loss"] - S(b, s)["val_loss"] for s in common]
    ea = [S(a, s).get("avg_experts_mean") for s in common]
    print(f"{a:>11} - {b:<9} seeds {common}: " + ", ".join(f"s{s}={x:+.4f}" for s, x in zip(common, d))
          + f" | mean {np.mean(d):+.4f}  min {min(d):+.4f} max {max(d):+.4f}  | A experts/tok " + ", ".join(f"{x:.3f}" if x else "-" for x in ea))

print("\n=== 2. SEED SPREAD (max - min final val loss) and mean/std ===")
for rid in MAIN + L30:
    v = [S(rid, s)["val_loss"] for s in seeds(rid)]
    e = [S(rid, s).get("avg_experts_mean") or float('nan') for s in seeds(rid)]
    print(f"{rid:<11} n={len(v)} mean={np.mean(v):.4f} std(ddof1)={np.std(v, ddof=1) if len(v)>1 else float('nan'):.4f} spread={max(v)-min(v):.4f} | experts mean={np.mean(e):.3f} spread={max(e)-min(e) if len(e)>1 else 0:.3f}")

print("\n=== 3. COMPUTE-MATCHED GAPS vs top-k frontier (10M) using effective forward FLOPs/token (linear interp between K1,K2,K3 seed-means) ===")
kx = [np.mean([S(k, s)["flops_per_token_effective_fwd"] for s in seeds(k)]) for k in ("K1", "K2", "K3")]
ky = [np.mean([S(k, s)["val_loss"] for s in seeds(k)]) for k in ("K1", "K2", "K3")]
print("top-k frontier points (FLOPs, loss):", [(f"{x/1e6:.3f}M", round(y, 4)) for x, y in zip(kx, ky)])
for rid in ["P30", "P50", "P60", "P70", "P50-noent", "P50-renorm", "K2-ent", "K2-renorm", "DENSE"]:
    for s in seeds(rid):
        x = S(rid, s)["flops_per_token_effective_fwd"]; y = S(rid, s)["val_loss"]
        f = np.interp(x, kx, ky)
        print(f"  {rid}/s{s}: FLOPs {x/1e6:.3f}M experts {S(rid, s).get('avg_experts_mean') or float('nan'):.3f} loss {y:.4f} topk-frontier {f:.4f} gap(frontier-run) {f-y:+.4f}" + ("  [outside frontier range]" if x < kx[0] or x > kx[-1] else ""))
print("top-k frontier slope K1->K2 per ln(experts):", (ky[1]-ky[0])/math.log(2), " K2->K3:", (ky[2]-ky[1])/math.log(1.5))
print("\n  30M: ROUGH estimate only (assumes the 10M top-k slope in ln(experts) holds at 30M; no L-K1/L-K3 runs exist)")
s12, s23 = (ky[1]-ky[0])/math.log(2), (ky[2]-ky[1])/math.log(1.5)
k2_30 = np.mean([S("L-K2", s)["val_loss"] for s in seeds("L-K2")])
for rid in ["L-P60", "L-P65", "L-P70"]:
    for s in seeds(rid):
        e = S(rid, s)["avg_experts_mean"]; y = S(rid, s)["val_loss"]
        f = k2_30 + (s12 if e < 2 else s23) * math.log(e / 2)
        print(f"  {rid}/s{s}: experts {e:.3f} loss {y:.4f} est. top-k at same experts {f:.4f} gap {f-y:+.4f}")

print("\n=== 4. LEARNING CURVES: val loss (seed mean) at eval steps; intermediate evals on first 262,144 val tokens, final on full 1,025,280 ===")
def curve(rid):
    by = {}
    for s in seeds(rid):
        for r in ev(rid, s):
            key = ("final" if r.get("final") else r["step"])
            by.setdefault(key, []).append((r["val_loss"], r.get("avg_experts_mean")))
    return by
for group, ids, steps in (("10M", MAIN, [100, 200, 300, 400, 600, 800, 1000, 1200, "final"]),
                          ("30M", L30, [100, 200, 400, 600, 800, 1000, 1200, 1600, 2000, 2400, 2800, 3200, 3600, "final"])):
    print(f"-- {group} --  step: " + " ".join(f"{str(x):>7}" for x in steps))
    for rid in ids:
        c = curve(rid)
        print(f"{rid:<11}" + " ".join(f"{np.mean([v[0] for v in c[x]]):7.4f}" if x in c else "      -" for x in steps))
    print("experts/token (seed mean) at the same steps:")
    for rid in ids:
        c = curve(rid)
        if "DENSE" in rid: continue
        print(f"{rid:<11}" + " ".join(f"{np.mean([v[1] for v in c[x]]):7.3f}" if x in c and c[x][0][1] is not None else "      -" for x in steps))
for a, b, ids in (("P60", "K2", None), ("P70", "K2", None), ("DENSE", "K2", None), ("L-P60", "L-K2", None), ("L-P65", "L-K2", None), ("L-DENSE", "L-K2", None)):
    ca, cb = curve(a), curve(b)
    ks = [k for k in ca if k in cb]
    ks = sorted([k for k in ks if k != "final"]) + (["final"] if "final" in ks else [])
    sel = [k for k in ks if k == "final" or k % (200 if a.startswith("L-") else 100) == 0]
    print(f"gap {a}-{b} over training: " + " ".join(f"{k}:{np.mean([v[0] for v in ca[k]]) - np.mean([v[0] for v in cb[k]]):+.4f}" for k in sel))

print("\n=== 5. FINAL ROUTING / LOAD (final eval record, full val set) per run per seed ===")
print("max/E = largest expert share x 8 (1 = balanced); min/E = smallest share x 8; ent = router entropy (nats, max ln8=2.079); top = mean max prob")
for rid in MAIN + L30:
    if "DENSE" in rid: continue
    for s in seeds(rid):
        f = [r for r in ev(rid, s) if r.get("final")][-1]
        ro = f["routing"]
        load = np.array(ro["load"])
        print(f"{rid}/s{s}: experts {[round(x,3) for x in ro['avg_experts']]}  max/E {[round(x,2) for x in load.max(1)*8]}  min/E {[round(x,3) for x in load.min(1)*8]}"
              f"  ent {ro['entropy']}  top {ro['max_prob']}  val_bal {f.get('val_bal_loss', float('nan')):.4f} val_ent {f.get('val_ent_loss', float('nan')):.4f} val_z {f.get('val_z_loss', float('nan')):.3f}")
    # overall share of all assignments
print("overall max share of ALL assignments (all layers pooled), K1/P30 health flags:")
for rid in ["K1", "P30", "P50", "P50-noent", "K2-renorm"]:
    for s in seeds(rid):
        f = [r for r in ev(rid, s) if r.get("final")][-1]
        load = np.array(f["routing"]["load"]); ae = np.array(f["routing"]["avg_experts"])
        print(f"  {rid}/s{s}: per-layer max share {[round(x,3) for x in load.max(1)]}, argmax expert {load.argmax(1).tolist()}")

print("\n=== 6. TRAINING HEALTH (metrics.jsonl) ===")
for rid in MAIN + L30:
    for s in seeds(rid):
        m = mt(rid, s)
        if not m: continue
        gn = np.array([r["grad_norm"] for r in m]); cf = np.array([r.get("clip_frac", 0) for r in m])
        lm = np.array([r["lm_loss"] for r in m])
        nan = int(np.sum(~np.isfinite(lm)))
        last = m[-1]
        rg = np.array([r["router_grad_norm"] for r in m if r.get("router_grad_norm")]) if any(r.get("router_grad_norm") for r in m) else None
        print(f"{rid}/s{s}: records {len(m)} nonfinite {nan}  grad_norm max {gn.max():.3f} (step {m[int(gn.argmax())]['step']}) last {gn[-1]:.3f}  clip_frac mean {cf.mean():.3f} after step 200 {np.mean([r.get('clip_frac',0) for r in m if r['step']>200]):.3f}"
              f"  final train lm {last['lm_loss']:.4f} bal {last.get('bal_loss', float('nan')):.4f} ent {last.get('ent_loss', float('nan')):.4f} z {last.get('z_loss', float('nan')):.3f}"
              + (f"  router_gn last {[round(x,4) for x in rg[-1]]}" if rg is not None else "") + f" mem {last.get('mem_gb')}")

print("\n=== 7. TRAIN vs VAL (final train lm loss = mean of last 10 steps; val = full val) ===")
for rid in MAIN + L30:
    g = [(S(rid, s)["final_train_lm_loss"], S(rid, s)["val_loss"]) for s in seeds(rid)]
    print(f"{rid:<11} train {np.mean([x[0] for x in g]):.4f} val {np.mean([x[1] for x in g]):.4f} val-train {np.mean([x[1]-x[0] for x in g]):+.4f}")

print("\n=== 8. H3: TRAINING-TIME p vs INFERENCE-TIME p (sweep_eval.json, first 131,072 val tokens; per seed) ===")
def sweep(rid, s):
    p = R / rid / f"seed{s}" / "sweep_eval.json"
    return {r["label"]: r for r in json.loads(p.read_text())["rows"]} if p.exists() else None
for rid in ["K1", "K2", "K3", "P30", "P50", "P60", "P70", "K2-ent", "P50-noent", "K2-renorm", "P50-renorm", "L-K2", "L-P60", "L-P65", "L-P70"]:
    for s in seeds(rid):
        sw = sweep(rid, s)
        if not sw: print(f"{rid}/s{s}: no sweep"); continue
        print(f"{rid}/s{s}: " + "  ".join(f"{k.replace('topp_','').replace('topk_','')}={v['val_loss']:.4f}[{v['avg_experts_mean']:.2f}]" for k, v in sw.items()))
print("\n-- cross table: model trained at p_train evaluated at p_eval (seed mean loss [experts]); diagonal = as trained --")
tp = {"P30": 0.3, "P50": 0.5, "P70": 0.7}
for pe in (0.3, 0.5, 0.7):
    row = []
    for rid in ["P30", "P50", "P60", "P70", "K2"]:
        vals = [sweep(rid, s)[f"topp_p{pe:g}"] for s in seeds(rid)]
        row.append(f"{rid}: {np.mean([v['val_loss'] for v in vals]):.4f} [{np.mean([v['avg_experts_mean'] for v in vals]):.2f}]")
    print(f"  eval p={pe}: " + " | ".join(row))
print("-- top-k models evaluated with top-k k' (seed mean) --")
for rid in ["K1", "K2", "K3", "K2-renorm", "L-K2"]:
    print(f"  {rid}: " + " ".join(f"k{k}={np.mean([sweep(rid, s)[f'topk_k{k}']['val_loss'] for s in seeds(rid)]):.4f}" for k in (1, 2, 3, 4)))

print("\n=== 9. THROUGHPUT / WALL TIME ===")
for rid in MAIN + L30 + ["LR-K2-4e-3", "LR-P50-4e-3", "LR-DENSE-4e-3"]:
    ss = seeds(rid)
    print(f"{rid:<14} tok/s " + ", ".join(f"{S(rid, s)['mean_tokens_per_s']:.0f}" for s in ss) + " | wall s " + ", ".join(f"{S(rid, s)['wall_time_s']:.0f}" for s in ss)
          + " | resumes " + ", ".join(str(S(rid, s)['n_resumes']) for s in ss) + f" | cum eff FLOPs {np.mean([S(rid, s)['cum_flops_effective'] for s in ss]):.4g} executed {np.mean([S(rid, s)['cum_flops_executed'] for s in ss]):.4g}")
print("\nparams (MoE):", json.dumps(S("K2", 0)["params"]), "active:", json.dumps(S("K2", 0)["active_params"]))
print("params (dense):", json.dumps(S("DENSE", 0)["params"]), "active:", json.dumps(S("DENSE", 0).get("active_params")))
f = [r for r in ev("K2", 0) if r.get("final")][-1]
print("executed fwd FLOPs/token MoE:", f["flops_per_token_executed"], " DENSE:", [r for r in ev("DENSE", 0) if r.get("final")][-1].get("flops_per_token_executed"))
for rid in ["K1", "K2", "K3", "DENSE"]:
    print(f"  effective fwd FLOPs/token {rid}: {S(rid, 0)['flops_per_token_effective_fwd']:.0f}")

print("\n=== 10. STEP-0 sanity (initial val loss; ln(4096)=8.318) ===")
for rid in ["K2", "P60", "DENSE", "L-K2"]:
    r0 = ev(rid, 0)[0]
    print(f"{rid}/s0 step0 val_loss {r0['val_loss']:.4f} experts {r0.get('avg_experts_mean')}")
