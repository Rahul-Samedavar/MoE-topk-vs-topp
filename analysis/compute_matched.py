#!/usr/bin/env python3
"""compute_matched.py - top-p against the top-k curve at matched inference compute and at matched training compute.

    python analysis/compute_matched.py [--runs runs]

For each budget (10M and 30M tokens) prints every configuration's loss, final experts per token E (inference compute),
training-average E (training compute) and total training FLOPs, then the gap between the top-k curve and each run:

- inference compute: the run is placed at its final E, measured on the validation set at the end of training;
- training compute: the run is placed at its training-average E, recovered from cum_flops_effective, which train.py
  accumulates every step from that step's measured experts per token (forward FLOPs per token from
  moe.estimate_flops, times 3 for the backward pass). FLOPs are affine in E, so matching E matches FLOPs.

The top-k curve through top-1, top-2 and top-3 is interpolated linearly in log E (the conservative choice, since the
curve is convex), using the same-seed top-k run where it exists and that k's seed mean otherwise. Positive gap = top-p
has lower loss than top-k at the same compute. Only finished runs (summary.json present) are used.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moe import MoEConfig, estimate_flops  # noqa: E402

BUDGETS = [
    # name, top-k ids for k = 1, 2, 3, configurations compared with the curve, dense id
    ("10M", ("K1", "K2", "K3"), ["P30", "P50", "P60", "P70", "K2-ent", "P50-noent", "P50-ent1e-3", "P50-ent1e-2",
                                 "K2-renorm", "P50-renorm"], "DENSE"),
    ("30M", ("L-K1", "L-K2", "L-K3"), ["L-P60", "L-P65", "L-P70", "L-K2-renorm"], "L-DENSE"),
]


def load(runs_dir, exp):
    out = {}
    for sd in sorted((runs_dir / exp).glob("seed*")):
        if (sd / "summary.json").exists():
            s = json.loads((sd / "summary.json").read_text())
            s["model"] = json.loads((sd / "config.json").read_text())["model"]
            out[int(sd.name[4:])] = s
    return out


def train_E(s):
    """Training-average experts per token: inverts forward FLOPs/token = f(0) + E * (f(1) - f(0))."""
    cfg = MoEConfig(**s["model"])
    f0, f1 = (estimate_flops(cfg, e)["forward"] for e in (0.0, 1.0))
    return (s["cum_flops_effective"] / (3 * s["tokens"]) - f0) / (f1 - f0)


def interp_logE(E, ks, losses):
    """Loss at E, linear in log E between neighbouring k; extrapolated linearly outside [1, 3]."""
    i = int(np.clip(np.searchsorted(ks, E) - 1, 0, len(ks) - 2))
    x0, x1 = np.log(ks[i]), np.log(ks[i + 1])
    return losses[i] + (np.log(E) - x0) / (x1 - x0) * (losses[i + 1] - losses[i])


def topk_at(E, seed, topk):
    ys = [r[seed]["val_loss"] if seed in r else np.mean([s["val_loss"] for s in r.values()]) for r in topk]
    return interp_logE(E, [1, 2, 3], ys)


def fmt(g):
    return ", ".join(f"{x:+.3f}" for x in g) + f" (mean {np.mean(g):+.3f})"


def report(runs_dir, name, topk_ids, ids, dense):
    topk = [load(runs_dir, k) for k in topk_ids]
    if not all(topk):
        print(f"\n##### {name}: missing {[k for k, r in zip(topk_ids, topk) if not r]}, cannot draw the top-k curve")
        return
    k2 = topk[1]
    k2_flops = np.mean([s["cum_flops_effective"] for s in k2.values()])
    k2_spread = max(s["val_loss"] for s in k2.values()) - min(s["val_loss"] for s in k2.values())
    print(f"\n##### {name} tokens #####")
    print(f"{'config':<13}{'n':<3}{'mean loss':<11}{'spread':<9}{'final E':<9}{'train E':<9}"
          f"{'train FLOPs':<13}{'vs ' + topk_ids[1]:<10}per seed")
    for i in list(topk_ids) + ids + [dense]:
        r = load(runs_dir, i)
        if not r:
            continue
        L = [s["val_loss"] for s in r.values()]
        moe = r[min(r)]["avg_experts_mean"] is not None
        fE = f"{np.mean([s['avg_experts_mean'] for s in r.values()]):.3f}" if moe else "n/a"
        tE = f"{np.mean([train_E(s) for s in r.values()]):.3f}" if moe else "n/a"
        F = np.mean([s["cum_flops_effective"] for s in r.values()])
        print(f"{i:<13}{len(L):<3}{np.mean(L):<11.4f}{max(L) - min(L):<9.4f}{fE:<9}{tE:<9}{F:<13.3e}"
              f"{100 * (F / k2_flops - 1):<+10.1f}" + ", ".join(f"s{sd}={s['val_loss']:.4f}" for sd, s in r.items()))
    print(f"\ngap below the top-k curve (log-E interpolation; > 0 = better than top-k at the same compute); "
          f"{topk_ids[1]} seed spread {k2_spread:.4f}")
    for i in ids:
        r = load(runs_dir, i)
        if not r:
            continue
        inf = [topk_at(s["avg_experts_mean"], sd, topk) - s["val_loss"] for sd, s in r.items()]
        trn = [topk_at(train_E(s), sd, topk) - s["val_loss"] for sd, s in r.items()]
        print(f"  {i:<13} matched inference compute: {fmt(inf)}")
        print(f"  {'':<13} matched training compute:  {fmt(trn)}")
    d = load(runs_dir, dense)
    if d:
        dl = np.mean([s["val_loss"] for s in d.values()])
        ys = [np.mean([s["val_loss"] for s in r.values()]) for r in topk]
        print(f"  {dense:<13} same training FLOPs as {topk_ids[1]}; gap below the top-k curve at E = 2: "
              f"{ys[1] - dl:+.3f}" + (f"; top-k reaches the dense loss at E = "
                                     f"{2 ** ((ys[0] - dl) / (ys[0] - ys[1])):.2f}" if ys[1] < dl < ys[0] else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    for b in BUDGETS:
        report(Path(a.runs), *b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
