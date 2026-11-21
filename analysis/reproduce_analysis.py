#!/usr/bin/env python3
"""reproduce_analysis.py - run every analysis script on the finished runs.

    python analysis/reproduce_analysis.py                     # all steps; reads runs/, writes results/
    python analysis/reproduce_analysis.py --skip checkpoint_probe tokens_30m
    python analysis/reproduce_analysis.py --runs /path/to/downloaded/runs --out results

Each step's console output is saved as results/<step>.txt next to the figures it writes.
checkpoint_probe needs the final checkpoints (ckpt/final.pt); every other step needs only the small log files
(config.json, summary.json, eval.jsonl, metrics.jsonl, sweep_eval.json, routing_dump.npz).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def steps(runs: str, out: str):
    o = Path(out)
    return [
        # name, script, arguments, what it covers
        ("tuning_lr", "tuning_report.py", ["lr", "--runs", runs], "lr-sweep"),
        ("tuning_alpha", "tuning_report.py", ["alpha", "--runs", runs], "alpha-sweep"),
        ("analyze_main", "analyze.py", ["--runs", runs, "--only", "main", "--out", str(o / "analyze_main")],
         "10M runs"),
        ("analyze_scale", "analyze.py", ["--runs", runs, "--only", "scale", "--out", str(o / "analyze_scale")],
         "30M runs"),
        ("summary_stats", "summary_stats.py", ["--runs", runs], "all 10M and 30M runs"),
        ("compute_matched", "compute_matched.py", ["--runs", runs],
         "top-k curve gaps at matched inference and training compute"),
        ("entropy_sweep", "entropy_sweep.py", ["--runs", runs], "entropy-10m"),
        ("tokens_10m", "analyze_tokens.py", ["--runs", runs, "--topp", "P30", "P50", "P60", "P70", "P50-noent",
                                             "P50-renorm", "--topk", "K2", "--out", str(o / "tokens_10m")],
         "token level, 10M"),
        ("tokens_30m", "analyze_tokens.py", ["--runs", runs, "--topk", "L-K2", "--topp", "L-P60", "L-P65", "L-P70",
                                             "--out", str(o / "tokens_30m")], "token level, 30M"),
        ("probe_layers", "probe_layers.py", ["--runs", runs, "--out", str(o / "probe_layers")], "per-layer probes"),
        ("checkpoint_probe", "checkpoint_probe.py", ["--runs", runs, "--out", str(o / "checkpoint_probe.json")],
         "checkpoint probes"),
        ("figures", "figures.py", ["--runs", runs, "--out", str(o / "figures")], "summary figures"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", nargs="+", help="run only these steps")
    ap.add_argument("--skip", nargs="+", default=[], help="skip these steps")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in steps(a.runs, a.out) if (not a.only or s[0] in a.only) and s[0] not in a.skip]
    failed = []
    for name, script, args, what in todo:
        t0 = time.time()
        print(f"{name:<17} ({what}) ...", end=" ", flush=True)
        with open(out / f"{name}.txt", "w") as f:
            rc = subprocess.call([sys.executable, str(HERE / script), *args], stdout=f, stderr=subprocess.STDOUT)
        print(f"{'ok' if rc == 0 else f'FAILED (exit {rc})'}, {time.time() - t0:.0f}s -> {out / name}.txt")
        if rc != 0:
            failed.append(name)
    if failed:
        print("failed:", ", ".join(failed), "(see their .txt files)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
