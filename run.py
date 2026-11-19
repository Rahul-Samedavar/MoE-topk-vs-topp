#!/usr/bin/env python3
"""run.py - master experiment runner: train any subset of the registered runs (experiments.py).

    python run.py list                                   # every configuration, its group, seeds and status
    python run.py list --group main-30m
    python run.py train --all                            # every registered run
    python run.py train --group main-10m entropy-10m     # whole groups
    python run.py train --only K2 P60 --seeds 0          # some configurations, some seeds
    python run.py train --only K2 --seeds 3 4            # other seeds than the registered ones
    python run.py train --group main-10m --dry-run       # print the commands only
    python run.py train --only K2 -- --device cpu        # after --: extra arguments for train.py
    python run.py verify --runs runs                     # do finished runs match the registered configs?

Without --seeds, each configuration is trained with the seeds listed for it in experiments.py.
Each run is a fresh `python train.py --exp <ID> --seed <N>` process writing to runs/<ID>/seed<N>/. Finished runs
(summary.json present) are skipped, and interrupted ones resume from their last checkpoint, so after Ctrl-C or a
crash, run the same command again. Run from the repository root (data paths are relative to it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import experiments as X

ROOT = Path(__file__).resolve().parent
# Rough wall-clock estimate on an Apple-silicon laptop (MPS), one run at a time.
MIN_PER_10M = {"moe": 22.0, "dense": 11.0}


def _minutes(e: X.Exp) -> float:
    return MIN_PER_10M["dense" if e.family == "dense" else "moe"] * X.TOKENS[e.budget] / 10_000_000


def _status(rd: Path) -> str:
    if X.is_done(rd):
        return "done"
    return "partial" if (rd / "config.json").exists() else "-"


def cmd_list(a) -> int:
    todo = X.select(a.group, a.only)
    shown = {e.id for e, _ in todo}
    print(f"{'group':<14}{'ID':<14}{'tokens':<8}{'seeds':<8}{'status':<22}description")
    for e in X.EXPS:
        if e.id not in shown:
            continue
        st = [_status(X.run_dir(e.id, s, a.out_dir)) for s in e.seeds]
        n_done = st.count("done")
        status = f"{n_done}/{len(st)} done" + (f", {st.count('partial')} partial" if "partial" in st else "")
        print(f"{e.group:<14}{e.id:<14}{e.budget:<8}{','.join(map(str, e.seeds)):<8}{status:<22}{e.desc}")
    left = [(e, s) for e, s in todo if not X.is_done(X.run_dir(e.id, s, a.out_dir))]
    print(f"\n{len(todo)} runs selected, {len(todo) - len(left)} finished in {a.out_dir}/; the remaining {len(left)} take "
          f"roughly {sum(_minutes(e) for e, _ in left) / 60:.1f} h one at a time on an Apple-silicon laptop (MPS)")
    print("groups: " + "; ".join(f"{g.name} ({g.budget})" for g in X.GROUPS.values()))
    problems = X.check_controls()
    print("controls check:", "OK" if not problems else "PROBLEMS\n  " + "\n  ".join(problems))
    return 1 if problems else 0


def cmd_train(a) -> int:
    if not (a.all or a.group or a.only):
        print("select runs with --all, --group or --only (see `python run.py list`)")
        return 2
    problems = X.check_controls()
    if problems:
        print("registry problems, refusing to train:\n  " + "\n  ".join(problems))
        return 1
    todo = X.select(a.group, a.only, a.seeds)
    extra = [x for x in (a.extra or []) if x != "--"]
    res: Dict[str, List[str]] = {"done": [], "skipped": [], "failed": [], "no_data": []}
    for i, (e, seed) in enumerate(todo, 1):
        label = f"{e.id}/seed{seed}"
        rd = X.run_dir(e.id, seed, a.out_dir)
        if X.is_done(rd):
            res["skipped"].append(label)
            print(f"[{i}/{len(todo)}] {label}: already finished, skipping")
            continue
        if not (ROOT / e.data / "meta.json").exists():
            res["no_data"].append(label)
            print(f"[{i}/{len(todo)}] {label}: {e.data}/meta.json missing (run prepare_data.py), skipping")
            continue
        cmd = [sys.executable, "train.py", "--exp", e.id, "--seed", str(seed), "--out-dir", a.out_dir, *extra]
        print(f"[{i}/{len(todo)}] {label}: {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        try:
            rc = subprocess.call(cmd, cwd=ROOT)
        except KeyboardInterrupt:
            rc = 130
        if rc == 130:
            print("interrupted: stopping. Run the same command again to resume.")
            break
        res["done" if rc == 0 else "failed"].append(label)
        if rc != 0:
            print(f"  {label} exited with code {rc}; continuing with the next run")
    print("\n" + "  ".join(f"{k}: {len(v)}" for k, v in res.items()))
    for k in ("failed", "no_data"):
        if res[k]:
            print(f"{k}: {', '.join(res[k])}")
    return 1 if res["failed"] else 0


def expected_config(e: X.Exp, seed: int):
    """The model and train dicts (and config hash) that train.py records for this run in config.json."""
    from train import RESUME_MUTABLE, _json_default, build_configs
    mc, tc = build_configs(e.id, [], seed, None, "auto", False)
    model, train = mc.to_dict(), asdict(tc)
    h = hashlib.sha1(json.dumps({"model": model, "train": {k: v for k, v in train.items()
                                                            if k not in RESUME_MUTABLE and k != "seed"}},
                                sort_keys=True, default=_json_default).encode()).hexdigest()[:10]
    return model, json.loads(json.dumps(train, default=_json_default)), h


def cmd_verify(a) -> int:
    """Compare every registered run found in --runs with the registry. Settings that do not affect results
    (device, progress bar) are ignored; the vocabulary size is taken from the data, as train.py does."""
    ignore = {"device", "tqdm"}
    bad = missing = ok = 0
    for e, seed in X.select(a.group, a.only):
        rd = X.run_dir(e.id, seed, a.runs)
        if not (rd / "config.json").exists():
            missing += 1
            print(f"{e.id}/seed{seed}: not found")
            continue
        got = json.loads((rd / "config.json").read_text())
        model, train, h = expected_config(e, seed)
        model["vocab_size"] = got["data"]["vocab_size"]
        diffs = [f"model.{k}: {got['model'].get(k)!r} != {v!r}" for k, v in model.items() if got["model"].get(k) != v]
        diffs += [f"train.{k}: {got['train'].get(k)!r} != {v!r}" for k, v in train.items()
                  if k not in ignore and got["train"].get(k) != v]
        if got.get("config_hash") != h:
            diffs.append(f"config_hash {got.get('config_hash')} != {h}")
        state = "done" if X.is_done(rd) else "NOT FINISHED"
        if diffs:
            bad += 1
            print(f"{e.id}/seed{seed} ({state}): MISMATCH  " + "; ".join(diffs))
        else:
            ok += 1
            print(f"{e.id}/seed{seed} ({state}): matches (config_hash {h})")
    print(f"\n{ok} match, {bad} differ, {missing} not found")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def selection(p, seeds=True):
        p.add_argument("--group", nargs="+", choices=list(X.GROUPS), help="one or more groups")
        p.add_argument("--only", nargs="+", metavar="ID", help="one or more configuration IDs")
        if seeds:
            p.add_argument("--seeds", type=int, nargs="+", help="override the registered seeds")

    p = sub.add_parser("list", help="show the configurations, seeds and status")
    selection(p, seeds=False)
    p.add_argument("--out-dir", default=X.RUNS_DIR)
    p = sub.add_parser("train", help="train the selected runs one after another")
    selection(p)
    p.add_argument("--all", action="store_true", help="every registered run")
    p.add_argument("--out-dir", default=X.RUNS_DIR)
    p.add_argument("--dry-run", action="store_true", help="print the train.py commands without running them")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="after --: extra arguments for train.py")
    p = sub.add_parser("verify", help="check the config.json of existing runs against the registry")
    selection(p, seeds=False)
    p.add_argument("--runs", default=X.RUNS_DIR)
    a = ap.parse_args(argv)
    return {"list": cmd_list, "train": cmd_train, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
