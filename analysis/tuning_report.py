#!/usr/bin/env python3
"""tuning_report.py - results of the two 3M-token tuning groups.

    python analysis/tuning_report.py lr    [--runs runs]   # lr-sweep: which peak learning rate to use
    python analysis/tuning_report.py alpha [--runs runs]   # alpha-sweep: sensitivity to the load-balance weight

lr applies this decision rule: share one learning rate across top-k, top-p and dense if their optima are at most
one grid step apart and the shared value costs every family less than 0.01 in loss; otherwise tune per family.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import experiments as X  # noqa: E402


def lr_report(out_dir: str = X.RUNS_DIR, tol: float = 0.01) -> dict:
    """Read the lr-sweep summaries and apply the decision rule. Returns {"table", "choice", "notes"}."""
    table: Dict[str, Dict[float, float]] = {"topk": {}, "topp": {}, "dense": {}}
    for e in X.EXPS:
        if e.group == "lr-sweep":
            rd = X.run_dir(e.id, e.seeds[0], out_dir)
            if X.is_done(rd):
                table[e.family][e.lr] = json.loads((rd / "summary.json").read_text())["val_loss"]
    notes: List[str] = []
    grid = sorted({e.lr for e in X.EXPS if e.group == "lr-sweep"})
    missing = [f for f, t in table.items() if len(t) < len(grid)]
    if missing:
        return {"table": table, "choice": None, "notes": [f"sweep incomplete for: {', '.join(missing)}"]}
    best = {f: min(t, key=t.get) for f, t in table.items()}
    for f, b in best.items():
        if b in (grid[0], grid[-1]):
            notes.append(f"{f}: best lr {b:g} is at the edge of the grid; extend the grid before freezing")
    idx = {f: grid.index(b) for f, b in best.items()}
    near = max(idx.values()) - min(idx.values()) <= 1
    shared = min(best.values(), key=lambda lr: max(table[f][lr] - table[f][best[f]] for f in table)) if near else None
    if shared is not None:
        cost = max(table[f][shared] - table[f][best[f]] for f in table)
        if cost < tol:
            return {"table": table, "choice": {f: shared for f in table},
                    "notes": notes + [f"shared lr {shared:g}: costs at most {cost:.4f} loss in any family (< {tol})"]}
        notes.append(f"best values are within one grid step but a shared lr would cost {cost:.4f} (>= {tol})")
    return {"table": table, "choice": dict(best), "notes": notes + ["tuned per family"]}


def cmd_lr(args) -> int:
    r = lr_report(args.runs)
    grid = sorted({lr for t in r["table"].values() for lr in t})
    print(f"{'lr':<8}" + "".join(f"{f:<10}" for f in r["table"]))
    for lr in grid:
        print(f"{lr:<8g}" + "".join(f"{r['table'][f].get(lr, float('nan')):<10.4f}" for f in r["table"]))
    for n in r["notes"]:
        print("-", n)
    if r["choice"]:
        print("\nset in experiments.py:\n  LR =", {k: v for k, v in r["choice"].items()})
    return 0 if r["choice"] else 1


def _base_alpha() -> float:
    from moe import MoEConfig
    return MoEConfig().bal_coef


def alpha_report(out_dir: str = X.RUNS_DIR) -> dict:
    """alpha-sweep results: validation loss, experts per token and load imbalance for each alpha, per family."""
    fams: Dict[str, Dict[float, dict]] = {"topk": {}, "topp": {}}
    for e in X.EXPS:
        if e.group != "alpha-sweep":
            continue
        a = e.model["bal_coef"]
        for s in e.seeds:
            rd = X.run_dir(e.id, s, out_dir)
            if not X.is_done(rd):
                continue
            summ = json.loads((rd / "summary.json").read_text())
            mt = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines() if x.strip()]
            last = mt[-1]
            d = fams[e.family].setdefault(a, {"loss": [], "E": [], "maxload": [], "minE": []})
            d["loss"].append(summ["val_loss"])
            d["E"].append(summ.get("avg_experts_mean") or 0.0)
            d["maxload"].append(float(np.max(last["expert_load"])))
            d["minE"].append(float(np.min(last["avg_experts"])))
    notes, rec = [], {}
    base = _base_alpha()
    for fam, by in fams.items():
        if base not in by or len(by) < 2:
            notes.append(f"{fam}: sweep incomplete")
            continue
        mean = {a: float(np.mean(d["loss"])) for a, d in by.items()}
        best = min(mean, key=mean.get)
        noise = max(max(d["loss"]) - min(d["loss"]) for d in by.values() if len(d["loss"]) > 1) if any(len(d["loss"]) > 1 for d in by.values()) else float("nan")
        gain = mean[base] - mean[best]
        rec[fam] = {"best": best, "gain_vs_default": gain, "noise": noise,
                    "action": "keep default" if (best == base or not gain > noise) else f"rerun {fam} at alpha={best:g}"}
        if fam == "topp":
            drift = max(np.mean(d["E"]) for d in by.values()) - min(np.mean(d["E"]) for d in by.values())
            if drift > 0.15:
                notes.append(f"topp: experts/token changes by {drift:.2f} across alpha, so loss differences are not at equal compute")
        if best in (min(by), max(by)):
            notes.append(f"{fam}: best alpha {best:g} is at the edge of the grid")
    return {"fams": fams, "rec": rec, "notes": notes}


def cmd_alpha(args) -> int:
    r = alpha_report(args.runs)
    for fam, by in r["fams"].items():
        print(f"\n{fam}: alpha   val_loss (mean [min,max])    experts/tok   max expert load   min layer experts   seeds")
        for a in sorted(by):
            d = by[a]
            print(f"       {a:<8g}{np.mean(d['loss']):.4f} [{min(d['loss']):.4f},{max(d['loss']):.4f}]      "
                  f"{np.mean(d['E']):<14.2f}{np.mean(d['maxload']):<18.2f}{np.mean(d['minE']):<20.2f}{len(d['loss'])}")
    for fam, x in r["rec"].items():
        print(f"\n{fam}: best alpha {x['best']:g}; gain over the default {x['gain_vs_default']:+.4f} (seed spread {x['noise']:.4f}) -> {x['action']}")
    for n in r["notes"]:
        print("-", n)
    print("\n(balanced load = 0.125 per expert; a top-p layer needing about 1 expert per token is not itself a problem)")
    return 0 if r["rec"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", choices=["lr", "alpha"])
    ap.add_argument("--runs", default=X.RUNS_DIR)
    args = ap.parse_args(argv)
    return {"lr": cmd_lr, "alpha": cmd_alpha}[args.report](args)


if __name__ == "__main__":
    sys.exit(main())
