"""train.py - trains ONE run (one configuration, one seed) and leaves a full record.

    python train.py --exp K2 --seed 0 [--device mps]                          # a registered run (experiments.py)
    python train.py --exp P50 --seed 1 --dry-run                              # print the resolved setup only
    python train.py --set route_mode=topp --set p=0.4 --run-name scratch      # ad hoc, no registry entry

To train many runs (whole groups, everything), use run.py, which calls this script once per run.
--set overrides any MoEConfig or TrainConfig field.

Everything for one run lives in  runs/<run-name>/seed<N>/ :
    config.json         resolved model + training config, data stats, environment, command line
    metrics.jsonl       training log, one record per log_interval steps (window means; see below)
    eval.jsonl          validation records at step 0, every eval_interval steps, and the final one
    events.jsonl        start / resume / interrupt / finish events with wall-clock times
    train.log           the human-readable console log
    ckpt/last.pt        full state for resuming (model, optimizer, step, counters); removed on success
    ckpt/final.pt       final weights + configs
    sweep_eval.json     inference-time sweep of p and k on a fixed validation subset
    routing_dump.npz    per-token routing on fixed validation windows: token ids, per-token loss,
                        and the expert mask [layers, tokens, experts]  (token-level analysis)
    summary.json        written last; its presence means the run is complete

Reproducibility:  weights init and data order depend only on --seed. The data order is a seeded
permutation of non-overlapping windows, so the batch at step s is a pure function of (seed, s):
there is no data-loader state to lose on resume. Resume is automatic: rerun the same command.
Ctrl-C or SIGTERM saves a checkpoint and exits with code 130.

Training metrics are means over the last log_interval steps (n_steps_in_window says how many). Grad norm is
measured before clipping. cum_flops_* accumulate analytic training FLOPs from the experts per token
measured at every step: "effective" is what a sparse implementation would need, "executed" is what this
dense-masked implementation actually runs (see moe.estimate_flops).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from moe import MoEConfig, MoETransformer, RoutingAccumulator, active_params, count_params, estimate_flops, summarize_routing


# Config

@dataclass
class TrainConfig:
    data_dir: str = "data/tinystories"
    # budget
    max_tokens: int = 10_000_000  # total training tokens (one pass over the data if the file is large enough)
    max_steps: int = 0  # if > 0, overrides max_tokens
    batch_size: int = 32  # sequences per micro-batch
    grad_accum: int = 1
    # optimisation (AdamW, linear warmup then cosine decay to min_lr_frac * lr)
    lr: float = 3e-3
    min_lr_frac: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1  # applied to matrices and expert tensors, not to norms or embeddings
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    # logging, evaluation, checkpoints (steps)
    log_interval: int = 10
    eval_interval: int = 100
    eval_tokens: int = 262_144  
    eval_batch_size: int = 64
    final_eval_tokens: int = 0  # 0 = the whole validation file
    ckpt_interval: int = 200  # must be a multiple of log_interval
    keep_every_ckpts: int = 0  # >0: also keep a weights-only copy every N checkpoints
    keep_last_ckpt: bool = False  # keep last.pt after a successful finish
    # final analysis artifacts
    final_sweep: bool = True
    sweep_tokens: int = 131_072
    sweep_p: tuple = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
    sweep_k: tuple = (1, 2, 3, 4)
    final_dump: bool = True
    dump_tokens: int = 131_072
    # misc
    seed: int = 0
    device: str = "auto"
    tqdm: bool = True


# Fields that may differ between the original run and a resumed one.
RESUME_MUTABLE = {"device", "tqdm", "log_interval", "eval_interval", "eval_tokens", "eval_batch_size",
                  "final_eval_tokens", "ckpt_interval", "keep_every_ckpts", "keep_last_ckpt", "final_sweep",
                  "sweep_tokens", "sweep_p", "sweep_k", "final_dump", "dump_tokens", "data_dir"}


# Small utilities

def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def jsonl_append(path: Path, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=_json_default) + "\n")
        f.flush()


def read_jsonl(path: Path) -> List[dict]:
    if not Path(path).exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break  
    return out


def truncate_jsonl(path: Path, max_step: int):
    """Keep only records with step <= max_step (drops records written after the last checkpoint)."""
    path = Path(path)
    if not path.exists():
        return
    keep = [r for r in read_jsonl(path) if r.get("step", -1) <= max_step]
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, default=_json_default) + "\n")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: dict):
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    os.replace(tmp, path)


def atomic_torch_save(obj, path: Path):
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rnd(a, nd=5):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def mem_gb(device: torch.device) -> Optional[float]:
    try:
        if device.type == "mps":
            return torch.mps.current_allocated_memory() / 1e9
        if device.type == "cuda":
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:  
        pass
    return None


def env_info(device: torch.device) -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "machine": platform.machine(), "device": str(device),
            "threads": torch.get_num_threads()}
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL,
            timeout=5).decode().strip()
    except Exception:  # noqa: BLE001
        info["git_commit"] = None
    return info


class RunLogger:
    """Prints through tqdm (so it does not break the bar) and appends to train.log."""

    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        tqdm.write(line)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# Data

def load_data(data_dir: Path):
    meta_path = Path(data_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found. Run the data preparation script first.")
    meta = json.loads(meta_path.read_text())
    assert meta.get("dtype", "uint16") == "uint16", "token files must be uint16"
    train = np.memmap(Path(data_dir) / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(Path(data_dir) / "val.bin", dtype=np.uint16, mode="r")
    return meta, train, val


class Sampler:
    """Deterministic one-pass sampler over non-overlapping windows of the training tokens.  """

    def __init__(self, tokens, seq_len: int, seed: int):
        self.tokens, self.L, self.seed = tokens, seq_len, seed
        self.n_seq = (len(tokens) - 1) // seq_len
        assert self.n_seq > 0, "training file shorter than one sequence"
        self._epoch, self._perm = None, None

    def _perm_for(self, epoch: int):
        if epoch != self._epoch:
            self._perm = np.random.default_rng([self.seed, epoch]).permutation(self.n_seq)
            self._epoch = epoch
        return self._perm

    def window_ids(self, first_seq: int, n: int) -> np.ndarray:
        ids = np.empty(n, dtype=np.int64)
        for j in range(n):
            epoch, pos = divmod(first_seq + j, self.n_seq)
            ids[j] = self._perm_for(epoch)[pos]
        return ids

    def batch(self, first_seq: int, n: int):
        offs = self.window_ids(first_seq, n)[:, None] * self.L + np.arange(self.L + 1)[None, :]
        w = np.asarray(self.tokens[offs]).astype(np.int64)  # [n, L+1]
        return torch.from_numpy(w[:, :-1].copy()), torch.from_numpy(w[:, 1:].copy())


# Optimisation helpers

def lr_at(step: int, tc: TrainConfig, total_steps: int) -> float:
    warm = min(tc.warmup_steps, max(total_steps - 1, 1))
    if step < warm:
        return tc.lr * (step + 1) / warm
    progress = min((step - warm) / max(1, total_steps - warm), 1.0)
    lo = tc.lr * tc.min_lr_frac
    return lo + 0.5 * (tc.lr - lo) * (1.0 + math.cos(math.pi * progress))


def make_optimizer(model: torch.nn.Module, tc: TrainConfig):
    decay, no_decay = [], []
    for name, prm in model.named_parameters():
        (decay if prm.ndim >= 2 and not name.startswith("tok_emb") else no_decay).append(prm)
    groups = [{"params": decay, "weight_decay": tc.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=tc.lr, betas=(tc.beta1, tc.beta2), eps=tc.adam_eps)


def param_norm(model: torch.nn.Module) -> float:
    return math.sqrt(sum(float(p.detach().float().pow(2).sum()) for p in model.parameters()))


@torch.no_grad()
def evaluate(model: MoETransformer, val, tc: TrainConfig, device: torch.device, n_tokens: int = 0,
             collect: bool = False, bytes_per_token: Optional[float] = None):
    """Validation loss and routing statistics on the first n_tokens (0 = all) of the validation file,
    as fixed non-overlapping windows. With collect=True also returns the per-token arrays."""
    cfg = model.cfg
    L = cfg.seq_len
    n_win = (len(val) - 1) // L
    if n_tokens > 0:
        n_win = min(n_win, max(1, n_tokens // L))
    assert n_win > 0, "validation file shorter than one sequence"
    was_training = model.training
    model.eval()
    acc = RoutingAccumulator()
    tot, cnt = 0.0, 0
    aux = {"bal": 0.0, "ent": 0.0, "z": 0.0}
    has_routing = False
    toks, losses = [], []
    masks: List[List[np.ndarray]] = []
    for w0 in range(0, n_win, tc.eval_batch_size):
        ws = np.arange(w0, min(w0 + tc.eval_batch_size, n_win))
        offs = ws[:, None] * L + np.arange(L + 1)[None, :]
        chunk = torch.from_numpy(np.asarray(val[offs]).astype(np.int64))
        x, y = chunk[:, :-1].to(device), chunk[:, 1:].to(device)
        out = model(x, return_routing=collect)
        tl = F.cross_entropy(out.logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1), reduction="none")
        n = tl.numel()
        tot += float(tl.sum())
        cnt += n
        if out.routing is not None:
            has_routing = True
            acc.update(out.routing, n)
            for key, v in (("bal", out.bal_loss), ("ent", out.ent_loss), ("z", out.z_loss)):
                aux[key] += float(v) * n
            if collect:
                if not masks:
                    masks = [[] for _ in out.routing.masks]
                for li, m in enumerate(out.routing.masks):
                    masks[li].append(m.cpu().numpy())
        if collect:
            toks.append(x.reshape(-1).cpu().numpy().astype(np.uint16))
            losses.append(tl.cpu().numpy().astype(np.float32))
    if was_training:
        model.train()

    loss = tot / cnt
    res = {"val_loss": loss, "val_ppl": math.exp(min(loss, 50.0)), "eval_tokens": cnt}
    if bytes_per_token:
        res["val_bpb"] = loss / math.log(2) / bytes_per_token
    if has_routing:
        r = acc.compute()
        avg = r["avg_experts"].tolist()
        res["routing"] = {"avg_experts": rnd(r["avg_experts"]), "load": rnd(r["load"]),
                          "count_hist": rnd(r["count_hist"]), "entropy": rnd(r["entropy"]),
                          "max_prob": rnd(r["max_prob"])}
        res["routing_summary"] = summarize_routing(r)
        res["avg_experts_mean"] = float(np.mean(avg))
        res.update({f"val_{k}_loss": v / cnt for k, v in aux.items()})
    else:
        avg = None
    res["flops_per_token_effective"] = estimate_flops(cfg, avg, attn_context="causal")["forward"]
    res["flops_per_token_executed"] = estimate_flops(cfg, avg, attn_context="causal", executed=True)["forward"]
    arrays = None
    if collect:
        arrays = {"tokens": np.concatenate(toks), "token_loss": np.concatenate(losses),
                  "expert_mask": np.stack([np.concatenate(m, axis=0) for m in masks]) if masks else None}
    return res, arrays


def routing_sweep(model: MoETransformer, val, tc: TrainConfig, device, bytes_per_token=None) -> List[dict]:
    """Evaluate the trained model under other routing settings (inference-time p and k)."""
    cfg = model.cfg
    orig = dict(route_mode=cfg.route_mode, k=cfg.k, p=cfg.p, min_k=cfg.min_k, max_k=cfg.max_k,
                gate_norm=cfg.gate_norm)
    settings = [("as_trained", {})]
    settings += [(f"topp_p{p:g}", dict(route_mode="topp", p=float(p))) for p in tc.sweep_p]
    settings += [(f"topk_k{k}", dict(route_mode="topk", k=int(k))) for k in tc.sweep_k if k <= cfg.n_experts]
    rows = []
    try:
        for label, change in settings:
            model.set_routing(**orig)
            if change:
                model.set_routing(**change)
            res, _ = evaluate(model, val, tc, device, n_tokens=tc.sweep_tokens, bytes_per_token=bytes_per_token)
            rows.append({"label": label, "route_mode": cfg.route_mode, "k": cfg.k, "p": cfg.p,
                         "max_k": cfg.max_k, "val_loss": res["val_loss"], "val_ppl": res["val_ppl"],
                         "val_bpb": res.get("val_bpb"), "eval_tokens": res["eval_tokens"],
                         "avg_experts_mean": res["avg_experts_mean"],
                         "avg_experts": res["routing"]["avg_experts"],
                         "flops_per_token_effective": res["flops_per_token_effective"]})
    finally:
        model.set_routing(**orig)
    return rows


# Checkpoints

def make_state(model, opt, mc, tc, step, tokens_per_step, total_steps, meta, counters) -> dict:
    return {
        "version": 1,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": opt.state_dict(),
        "step": step,
        "tokens": step * tokens_per_step,
        "total_steps": total_steps,
        "model_cfg": mc.to_dict(),
        "train_cfg": asdict(tc),
        "data": {"vocab_size": meta["vocab_size"], "train_tokens": meta.get("train_tokens")},
        "rng_cpu": torch.get_rng_state(),
        **counters,
    }


def check_resume_compat(state: dict, mc: MoEConfig, tc: TrainConfig, total_steps: int, meta: dict):
    problems = []
    if state["model_cfg"] != mc.to_dict():
        diff = {k for k in mc.to_dict() if mc.to_dict()[k] != state["model_cfg"].get(k)}
        problems.append(f"model config differs in {sorted(diff)}")
    old = state["train_cfg"]
    for f in fields(TrainConfig):
        if f.name in RESUME_MUTABLE:
            continue
        a, b = old.get(f.name), asdict(tc)[f.name]
        if isinstance(b, tuple):
            b = list(b)
        if a != b:
            problems.append(f"training config differs in {f.name}: checkpoint {a!r}, now {b!r}")
    if state["total_steps"] != total_steps:
        problems.append(f"total steps differ: checkpoint {state['total_steps']}, now {total_steps}")
    if state["data"]["vocab_size"] != meta["vocab_size"] or state["data"]["train_tokens"] != meta.get("train_tokens"):
        problems.append("the data files differ from the ones the checkpoint was trained on")
    if problems:
        raise RuntimeError("cannot resume: " + "; ".join(problems) +
                           ". Use --overwrite to start this run again from scratch.")


class _Stop:
    requested = False
    signum = None


def install_signal_handlers(stop: _Stop):
    old = {}

    def handler(signum, frame):
        if stop.requested:
            raise KeyboardInterrupt
        stop.requested, stop.signum = True, signum

    try:
        for s in (signal.SIGINT, signal.SIGTERM):
            old[s] = signal.signal(s, handler)
    except ValueError:  # not in the main thread
        pass
    return old


# Main training function

def train(tc: TrainConfig, mc: MoEConfig, run_dir, exp_id: Optional[str] = None, overwrite: bool = False,
          dry_run: bool = False, step_callback: Optional[Callable[[int], None]] = None,
          halt_after_steps: int = 0, halt_hard: bool = False) -> dict:
    """Train one run. Returns {"status": "done" | "skipped" | "interrupted" | "halted" | "dry_run", ...}.

    halt_after_steps / halt_hard are test hooks: stop after N steps as if interrupted (saving a checkpoint),
    or as if the process was killed (no checkpoint, logs left as they are)."""
    run_dir = Path(run_dir)
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    if (run_dir / "summary.json").exists():
        print(f"{run_dir}: already finished (summary.json exists), skipping")
        return {"status": "skipped", "summary": json.loads((run_dir / "summary.json").read_text())}

    meta, train_tok, val_tok = load_data(Path(tc.data_dir))
    if mc.vocab_size != meta["vocab_size"]:
        mc = replace(mc, vocab_size=meta["vocab_size"])
    assert tc.ckpt_interval % tc.log_interval == 0, "ckpt_interval must be a multiple of log_interval"
    device = pick_device(tc.device)
    tokens_per_step = tc.batch_size * tc.grad_accum * mc.seq_len
    total_steps = tc.max_steps if tc.max_steps > 0 else math.ceil(tc.max_tokens / tokens_per_step)
    run_name = f"{exp_id or run_dir.parent.name}/{run_dir.name}"
    bpt = (meta.get("val_stats") or {}).get("bytes_per_token")

    torch.manual_seed(tc.seed)
    model = MoETransformer(mc).to(device)
    is_moe = mc.arch == "moe"

    if dry_run:
        fl = estimate_flops(mc, mc.k if mc.route_mode == "topk" else mc.max_k)
        print(model.summary())
        print(f"run {run_name}: {total_steps} steps x {tokens_per_step} tokens = {total_steps * tokens_per_step:,} tokens "
              f"({total_steps * tokens_per_step / max(len(train_tok), 1):.2f} epochs of {len(train_tok):,}), device {device}")
        print(f"train FLOPs/token (top bound): {fl['train']:,.0f}; total {fl['train'] * total_steps * tokens_per_step:.3e}")
        print(json.dumps({"train": asdict(tc), "model": mc.to_dict()}, indent=1, default=_json_default))
        return {"status": "dry_run"}

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ckpt").mkdir(exist_ok=True)
    metrics_path, eval_path = run_dir / "metrics.jsonl", run_dir / "eval.jsonl"
    events_path, ckpt_path = run_dir / "events.jsonl", run_dir / "ckpt" / "last.pt"
    log = RunLogger(run_dir / "train.log")

    opt = make_optimizer(model, tc)
    sampler = Sampler(train_tok, mc.seq_len, tc.seed)
    counters = {"elapsed_s": 0.0, "step_time_total": 0.0, "n_resumes": 0,
                "cum_flops_effective": 0.0, "cum_flops_executed": 0.0}
    start_step = 0
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        check_resume_compat(state, mc, tc, total_steps, meta)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"])
        start_step = state["step"]
        counters.update({k: state[k] for k in counters})
        counters["n_resumes"] += 1
        truncate_jsonl(metrics_path, start_step)
        truncate_jsonl(eval_path, start_step)
        jsonl_append(events_path, {"event": "resume", "step": start_step, "time": time.strftime("%F %T")})
        log(f"RESUMED {run_name} at step {start_step}/{total_steps}")
    else:
        for pth in (metrics_path, eval_path):  # leftovers from a run that died before its first checkpoint
            if pth.exists():
                pth.unlink()
        cfg_record = {"exp": exp_id, "run_name": run_name, "seed": tc.seed, "model": mc.to_dict(),
                      "train": asdict(tc), "total_steps": total_steps, "tokens_per_step": tokens_per_step,
                      "data": meta, "env": env_info(device), "argv": sys.argv, "started": time.strftime("%F %T"),
                      "params": count_params(mc)}
        cfg_record["config_hash"] = hashlib.sha1(json.dumps(
            {"model": cfg_record["model"], "train": {k: v for k, v in cfg_record["train"].items()
                                                     if k not in RESUME_MUTABLE and k != "seed"}},
            sort_keys=True, default=_json_default).encode()).hexdigest()[:10]
        atomic_write_json(run_dir / "config.json", cfg_record)
        jsonl_append(events_path, {"event": "start", "time": time.strftime("%F %T")})
        log(model.summary())
        log(f"START {run_name}: {total_steps} steps x {tokens_per_step} tokens, device {device}, "
            f"{len(train_tok):,} training tokens available")

    t_session = time.time()
    elapsed = lambda: counters["elapsed_s"] + time.time() - t_session  # noqa: E731

    def run_eval(step: int, final: bool = False) -> dict:
        res, arrays = evaluate(model, val_tok, tc, device, n_tokens=tc.final_eval_tokens if final else tc.eval_tokens,
                               bytes_per_token=bpt)
        rec = {"step": step, "tokens": step * tokens_per_step, "final": final, "elapsed_s": elapsed(),
               "cum_flops_effective": counters["cum_flops_effective"],
               "cum_flops_executed": counters["cum_flops_executed"], "param_norm": param_norm(model), **res}
        jsonl_append(eval_path, rec)
        extra = f"  experts/token {res['avg_experts_mean']:.2f}" if "avg_experts_mean" in res else ""
        log(f"eval step {step}: val_loss {res['val_loss']:.4f}  ppl {res['val_ppl']:.2f}{extra}")
        return rec

    def save_ckpt(step: int):
        atomic_torch_save(make_state(model, opt, mc, tc, step, tokens_per_step, total_steps, meta,
                                     {**counters, "elapsed_s": elapsed()}), ckpt_path)
        n_ck = step // tc.ckpt_interval
        if tc.keep_every_ckpts and step % tc.ckpt_interval == 0 and n_ck % tc.keep_every_ckpts == 0:
            atomic_torch_save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                               "step": step, "model_cfg": mc.to_dict()}, run_dir / "ckpt" / f"step_{step:07d}.pt")

    if start_step == 0:
        run_eval(0)

    stop = _Stop()
    old_handlers = install_signal_handlers(stop)
    pbar = tqdm(total=total_steps, initial=start_step, desc=run_name, disable=not tc.tqdm, dynamic_ncols=True,
                mininterval=0.5)
    status, done = "done", start_step
    win: Dict[str, list] = defaultdict(list)
    win_time = 0.0
    model.train()
    try:
        for step in range(start_step, total_steps):
            t0 = time.perf_counter()
            lr = lr_at(step, tc, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            tot = {"loss": 0.0, "lm": 0.0, "bal": 0.0, "ent": 0.0, "z": 0.0}
            rt = None
            for micro in range(tc.grad_accum):
                x, y = sampler.batch((step * tc.grad_accum + micro) * tc.batch_size, tc.batch_size)
                out = model(x.to(device), y.to(device))
                (out.loss / tc.grad_accum).backward()
                tot["loss"] += float(out.loss.detach()) / tc.grad_accum
                tot["lm"] += float(out.lm_loss.detach()) / tc.grad_accum
                if out.routing is not None:
                    tot["bal"] += float(out.bal_loss.detach()) / tc.grad_accum
                    tot["ent"] += float(out.ent_loss.detach()) / tc.grad_accum
                    tot["z"] += float(out.z_loss.detach()) / tc.grad_accum
                    cur = {"avg_experts": out.routing.avg_experts, "load": out.routing.load,
                           "count_hist": out.routing.count_hist, "entropy": out.routing.entropy,
                           "max_prob": out.routing.max_prob}
                    cur = {k: v.detach().float().cpu().numpy() / tc.grad_accum for k, v in cur.items()}
                    rt = cur if rt is None else {k: rt[k] + cur[k] for k in cur}
            router_gn = [float(b.ffn.router.weight.grad.norm()) for b in model.blocks if b.is_moe] if is_moe else []
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip))  # pre-clip norm
            if not (math.isfinite(tot["loss"]) and math.isfinite(gn)):
                raise FloatingPointError(f"non-finite loss or gradient at step {step + 1}: loss {tot['loss']}, "
                                         f"grad norm {gn}. The last checkpoint is in {ckpt_path}.")
            opt.step()
            step_time = time.perf_counter() - t0
            done = step + 1

            avg = rt["avg_experts"].tolist() if rt is not None else None
            counters["cum_flops_effective"] += tokens_per_step * estimate_flops(mc, avg)["train"]
            counters["cum_flops_executed"] += tokens_per_step * estimate_flops(mc, avg, executed=True)["train"]
            counters["step_time_total"] += step_time
            win_time += step_time
            for k, v in tot.items():
                win[k].append(v)
            win["grad_norm"].append(gn)
            win["clipped"].append(float(gn > tc.grad_clip))
            if router_gn:
                win["router_grad_norm"].append(router_gn)
            if rt is not None:
                for k, v in rt.items():
                    win[k].append(v)

            if done % tc.log_interval == 0:
                n_win = len(win["loss"])
                rec = {"step": done, "tokens": done * tokens_per_step, "epoch": done * tokens_per_step / len(train_tok),
                       "lr": lr, "n_steps_in_window": n_win,
                       "loss": float(np.mean(win["loss"])), "lm_loss": float(np.mean(win["lm"])),
                       "bal_loss": float(np.mean(win["bal"])), "ent_loss": float(np.mean(win["ent"])),
                       "z_loss": float(np.mean(win["z"])), "grad_norm": float(np.mean(win["grad_norm"])),
                       "clip_frac": float(np.mean(win["clipped"])),
                       "tokens_per_s": n_win * tokens_per_step / max(win_time, 1e-9),
                       "step_time_s": win_time / n_win, "elapsed_s": elapsed(), "mem_gb": mem_gb(device),
                       "cum_flops_effective": counters["cum_flops_effective"],
                       "cum_flops_executed": counters["cum_flops_executed"]}
                if win["router_grad_norm"]:
                    rec["router_grad_norm"] = rnd(np.mean(win["router_grad_norm"], axis=0), 7)
                if win["avg_experts"]:
                    rec["avg_experts"] = rnd(np.mean(win["avg_experts"], axis=0))
                    rec["avg_experts_mean"] = float(np.mean(win["avg_experts"]))
                    rec["expert_load"] = rnd(np.mean(win["load"], axis=0))
                    rec["count_hist"] = rnd(np.mean(win["count_hist"], axis=0))
                    rec["router_entropy"] = rnd(np.mean(win["entropy"], axis=0))
                    rec["max_prob"] = rnd(np.mean(win["max_prob"], axis=0))
                jsonl_append(metrics_path, rec)
                win.clear()
                win_time = 0.0

            pbar.update(1)
            pbar.set_postfix(loss=f"{tot['lm']:.3f}", E=f"{np.mean(avg):.2f}" if avg else "-", lr=f"{lr:.1e}",
                             tok_s=f"{tokens_per_step / step_time:,.0f}", refresh=False)

            if done % tc.eval_interval == 0 and done < total_steps:
                run_eval(done)
            if done % tc.ckpt_interval == 0 and done < total_steps:
                save_ckpt(done)
            if step_callback is not None:
                step_callback(done)
            if stop.requested:
                save_ckpt(done)
                jsonl_append(events_path, {"event": "interrupt", "step": done, "signal": stop.signum,
                                           "time": time.strftime("%F %T")})
                log(f"INTERRUPTED at step {done}; checkpoint saved. Run the same command to resume.")
                status = "interrupted"
                break
            if halt_after_steps and done >= halt_after_steps and done < total_steps:
                if not halt_hard:
                    save_ckpt(done)
                jsonl_append(events_path, {"event": "halt_hard" if halt_hard else "halt", "step": done,
                                           "time": time.strftime("%F %T")})
                status = "halted_hard" if halt_hard else "halted"
                break
    finally:
        pbar.close()
        for s, h in old_handlers.items():
            signal.signal(s, h)
    if status != "done":
        return {"status": status, "step": done}

    # ---- finished training: final stages. Each can be redone after a crash because last.pt is kept until the end.
    save_ckpt(total_steps)
    truncate_jsonl(eval_path, total_steps - 1)  # replace any final record from an earlier attempt
    final = run_eval(total_steps, final=True)
    if is_moe and tc.final_sweep:
        rows = routing_sweep(model, val_tok, tc, device, bytes_per_token=bpt)
        atomic_write_json(run_dir / "sweep_eval.json", {"trained_with": {
            "route_mode": mc.route_mode, "k": mc.k, "p": mc.p, "max_k": mc.max_k}, "rows": rows})
        log("routing sweep: " + "  ".join(f"{r['label']}={r['val_loss']:.3f}@{r['avg_experts_mean']:.2f}"
                                          for r in rows))
    if is_moe and tc.final_dump:
        _, arr = evaluate(model, val_tok, tc, device, n_tokens=tc.dump_tokens, collect=True, bytes_per_token=bpt)
        np.savez_compressed(run_dir / "routing_dump.npz", tokens=arr["tokens"], token_loss=arr["token_loss"],
                            expert_mask=arr["expert_mask"], seq_len=np.int64(mc.seq_len),
                            n_experts=np.int64(mc.n_experts))
    atomic_torch_save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                       "model_cfg": mc.to_dict(), "train_cfg": asdict(tc), "step": total_steps}, run_dir / "ckpt" / "final.pt")
    last_train = (read_jsonl(metrics_path) or [{}])[-1]
    avg_final = final["routing"]["avg_experts"] if "routing" in final else None
    summary = {
        "exp": exp_id, "run_name": run_name, "seed": tc.seed, "status": "done", "steps": total_steps,
        "tokens": total_steps * tokens_per_step, "val_loss": final["val_loss"], "val_ppl": final["val_ppl"],
        "val_bpb": final.get("val_bpb"), "avg_experts_mean": final.get("avg_experts_mean"),
        "avg_experts_by_layer": avg_final, "final_train_lm_loss": last_train.get("lm_loss"),
        "cum_flops_effective": counters["cum_flops_effective"], "cum_flops_executed": counters["cum_flops_executed"],
        "flops_per_token_effective_fwd": final["flops_per_token_effective"],
        "params": count_params(mc), "active_params": active_params(mc, avg_final) if is_moe else None,
        "wall_time_s": elapsed(), "mean_tokens_per_s": total_steps * tokens_per_step / max(counters["step_time_total"], 1e-9),
        "n_resumes": counters["n_resumes"], "device": str(device)}
    atomic_write_json(run_dir / "summary.json", summary)
    jsonl_append(events_path, {"event": "finish", "time": time.strftime("%F %T")})
    if not tc.keep_last_ckpt and ckpt_path.exists():
        ckpt_path.unlink()
    log(f"DONE {run_name}: val_loss {summary['val_loss']:.4f}, {summary['wall_time_s'] / 60:.1f} min, "
        f"{summary['mean_tokens_per_s']:,.0f} tokens/s")
    return {"status": "done", "summary": summary}


# Command line

def _parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def build_configs(exp: Optional[str], sets: List[str], seed: int, data_dir: Optional[str], device: str,
                  no_tqdm: bool):
    model_over: Dict = {}
    train_over: Dict = {}
    if exp:
        from experiments import resolve
        model_over, train_over, exp_data = resolve(exp)
        train_over["data_dir"] = exp_data
    m_names = {f.name for f in fields(MoEConfig)}
    t_names = {f.name for f in fields(TrainConfig)}
    for s in sets:
        if "=" not in s:
            raise SystemExit(f"--set expects KEY=VALUE, got {s!r}")
        key, raw = s.split("=", 1)
        if key in m_names:
            model_over[key] = _parse_value(raw)
        elif key in t_names:
            train_over[key] = _parse_value(raw)
        else:
            raise SystemExit(f"--set: unknown key {key!r}")
    if data_dir:
        train_over["data_dir"] = data_dir
    train_over.update(seed=seed, device=device, tqdm=not no_tqdm)
    for key in ("sweep_p", "sweep_k"):
        if key in train_over:
            train_over[key] = tuple(train_over[key])
    return MoEConfig(**model_over), TrainConfig(**train_over)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", help="experiment ID from experiments.py (e.g. K2, P60, L-P65)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", help="directory name under --out-dir (default: the experiment ID)")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--data-dir")
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any MoEConfig or TrainConfig field, e.g. --set lr=0.002 --set p=0.4")
    ap.add_argument("--overwrite", action="store_true", help="delete the run directory and start again")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved setup and exit")
    ap.add_argument("--no-tqdm", action="store_true")
    ap.add_argument("--halt-after-steps", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--halt-hard", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    mc, tc = build_configs(args.exp, args.set, args.seed, args.data_dir, args.device, args.no_tqdm)
    name = args.run_name or args.exp or "adhoc"
    rd = Path(args.out_dir) / name / f"seed{args.seed}"
    res = train(tc, mc, rd, exp_id=name, overwrite=args.overwrite, dry_run=args.dry_run,
                halt_after_steps=args.halt_after_steps, halt_hard=args.halt_hard)
    return 130 if res["status"] == "interrupted" else 0


if __name__ == "__main__":
    sys.exit(main())