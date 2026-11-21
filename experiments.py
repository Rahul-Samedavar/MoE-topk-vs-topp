"""experiments.py - the registry of every training run (single source of truth).

Each entry is one configuration (a run ID such as K2 or L-P65), trained once per listed seed. Each (ID, seed) pair is
one run, stored in runs/<ID>/seed<N>/. Configurations are organised in groups; all runs in a group share one
training budget:

    group           budget  contents
    lr-sweep        3M      peak learning rate, for top-2, top-p 0.5 and dense
    alpha-sweep     3M      load-balance weight alpha, for top-2 and top-p 0.5
    main-10m        10M     top-1/2/3, top-p 0.3/0.5/0.6/0.7, dense
    entropy-10m     10M     router entropy loss: added to top-2; removed from and varied for top-p 0.5
    gatenorm-10m    10M     renormalized gates, for top-2 and top-p 0.5
    main-30m        30M     top-1/2/3, top-p 0.6/0.65/0.7, dense
    gatenorm-30m    30M     renormalized gates, for top-2

Every setting not listed here is the default in moe.MoEConfig and train.TrainConfig.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PEAK_LR = 4e-3  # every run except the lr-sweep
ENT = 1e-4  # router entropy weight for top-p runs (Huang et al. 2024)
TOPP_CAP = 4  # max experts per token for top-p
LR_GRID = ("1e-3", "2e-3", "4e-3", "8e-3")
ALPHA_GRID = ("1e-3", "1e-2", "1e-1", "3e-1")  # 1e-2 is the default used by every other run

TOKENS = {"3M": 3_000_000, "10M": 10_000_000, "30M": 30_000_000}
# Pre-tokenized data written by prepare_data.py. The first 10M tokens of the 30M file are the 10M file, and both
# share one validation file. The 3M runs use the first 3M tokens of each seed's order over the 10M file.
DATA_DIRS = {"3M": "data/tinystories", "10M": "data/tinystories", "30M": "data/tinystories30m"}
RUNS_DIR = "runs"


@dataclass(frozen=True)
class Group:
    name: str
    budget: str
    desc: str


GROUPS: Dict[str, Group] = {g.name: g for g in [
    Group("lr-sweep", "3M", "peak learning rate sweep, one per model family"),
    Group("alpha-sweep", "3M", "load-balance weight sweep"),
    Group("main-10m", "10M", "top-k, top-p thresholds and dense baseline"),
    Group("entropy-10m", "10M", "entropy-loss ablations and entropy-weight sweep"),
    Group("gatenorm-10m", "10M", "renormalized-gate ablation"),
    Group("main-30m", "30M", "top-k, top-p and dense at 3x the training budget"),
    Group("gatenorm-30m", "30M", "renormalized-gate ablation at 3x the training budget"),
]}


@dataclass(frozen=True)
class Exp:
    id: str
    group: str
    desc: str
    model: Dict = field(default_factory=dict)  # MoEConfig overrides
    seeds: Tuple[int, ...] = (0,)
    lr: float = PEAK_LR

    @property
    def budget(self) -> str:
        return GROUPS[self.group].budget

    @property
    def data(self) -> str:
        return DATA_DIRS[self.budget]

    @property
    def family(self) -> str:
        """Model family: dense, topp or topk."""
        if self.model.get("arch") == "dense":
            return "dense"
        return "topp" if self.model.get("route_mode") == "topp" else "topk"

    @property
    def train(self) -> Dict:
        """TrainConfig overrides. Everything not listed here keeps its train.TrainConfig default."""
        return {"max_tokens": TOKENS[self.budget], "lr": self.lr}


def _topk(k: int, **kw) -> Dict:
    return dict(route_mode="topk", k=k, **kw)


def _topp(p: float, **kw) -> Dict:
    kw.setdefault("ent_coef", ENT)
    return dict(route_mode="topp", p=p, max_k=TOPP_CAP, **kw)


DENSE = dict(arch="dense")

EXPS: List[Exp] = [
    # ---- lr-sweep: 3M tokens, seed 0 ----
    *[Exp(f"LR-K2-{s}", "lr-sweep", f"top-2, peak lr {s}", _topk(2), lr=float(s)) for s in LR_GRID],
    *[Exp(f"LR-P50-{s}", "lr-sweep", f"top-p 0.5, peak lr {s}", _topp(0.5), lr=float(s)) for s in LR_GRID],
    *[Exp(f"LR-DENSE-{s}", "lr-sweep", f"dense, peak lr {s}", DENSE, lr=float(s)) for s in LR_GRID],
    # ---- alpha-sweep: 3M tokens, seeds 0-1 ----
    *[Exp(f"AL-K2-{a}", "alpha-sweep", f"top-2, balance weight {a}", _topk(2, bal_coef=float(a)), (0, 1))
      for a in ALPHA_GRID],
    *[Exp(f"AL-P50-{a}", "alpha-sweep", f"top-p 0.5, balance weight {a}", _topp(0.5, bal_coef=float(a)), (0, 1))
      for a in ALPHA_GRID],
    # ---- main-10m ----
    Exp("K1", "main-10m", "top-1", _topk(1), (0, 1)),
    Exp("K2", "main-10m", "top-2 (main baseline)", _topk(2), (0, 1, 2)),
    Exp("K3", "main-10m", "top-3", _topk(3), (0, 1)),
    Exp("P30", "main-10m", "top-p 0.3", _topp(0.3), (0, 1)),
    Exp("P50", "main-10m", "top-p 0.5", _topp(0.5), (0, 1)),
    Exp("P60", "main-10m", "top-p 0.6", _topp(0.6), (0, 1, 2)),
    Exp("P70", "main-10m", "top-p 0.7", _topp(0.7), (0, 1)),
    Exp("DENSE", "main-10m", "dense FFN, same active parameters as top-2", DENSE, (0, 1, 2)),
    # ---- entropy-10m ----
    Exp("K2-ent", "entropy-10m", "top-2 with the entropy loss (beta 1e-4)", _topk(2, ent_coef=ENT), (0, 1)),
    Exp("P50-noent", "entropy-10m", "top-p 0.5 without the entropy loss (beta 0)", _topp(0.5, ent_coef=0.0), (0, 1)),
    Exp("P50-ent1e-3", "entropy-10m", "top-p 0.5, entropy weight 1e-3", _topp(0.5, ent_coef=1e-3), (0, 1)),
    Exp("P50-ent1e-2", "entropy-10m", "top-p 0.5, entropy weight 1e-2", _topp(0.5, ent_coef=1e-2), (0, 1)),
    # ---- gatenorm-10m ----
    Exp("K2-renorm", "gatenorm-10m", "top-2, renormalized gates", _topk(2, gate_norm="renorm"), (0, 1, 2)),
    Exp("P50-renorm", "gatenorm-10m", "top-p 0.5, renormalized gates", _topp(0.5, gate_norm="renorm"), (0, 1)),
    # ---- main-30m ----
    Exp("L-K2", "main-30m", "top-2", _topk(2), (0, 1, 2)),
    Exp("L-P60", "main-30m", "top-p 0.6", _topp(0.6), (0, 1, 2)),
    Exp("L-P65", "main-30m", "top-p 0.65", _topp(0.65), (0, 1, 2)),
    Exp("L-P70", "main-30m", "top-p 0.7", _topp(0.7), (0,)),
    Exp("L-DENSE", "main-30m", "dense FFN, same active parameters as top-2", DENSE, (0, 1, 2)),
    Exp("L-K1", "main-30m", "top-1", _topk(1), (0, 1)),
    Exp("L-K3", "main-30m", "top-3", _topk(3), (0, 1)),
    # ---- gatenorm-30m ----
    Exp("L-K2-renorm", "gatenorm-30m", "top-2, renormalized gates", _topk(2, gate_norm="renorm"), (0, 1)),
]
BY_ID: Dict[str, Exp] = {e.id: e for e in EXPS}


def resolve(exp_id: str):
    """Returns (model overrides, train overrides, data dir) for an experiment ID."""
    if exp_id not in BY_ID:
        raise KeyError(f"unknown experiment {exp_id!r}; known: {', '.join(BY_ID)}")
    e = BY_ID[exp_id]
    return dict(e.model), dict(e.train), e.data


def select(groups: Optional[List[str]] = None, only: Optional[List[str]] = None,
           seeds: Optional[List[int]] = None) -> List[Tuple[Exp, int]]:
    """(experiment, seed) pairs in registry order. No filter selects everything."""
    for g in groups or []:
        if g not in GROUPS:
            raise KeyError(f"unknown group {g!r}; known: {', '.join(GROUPS)}")
    for i in only or []:
        if i not in BY_ID:
            raise KeyError(f"unknown experiment {i!r}; known: {', '.join(BY_ID)}")
    out = []
    for e in EXPS:
        if groups and e.group not in groups:
            continue
        if only and e.id not in only:
            continue
        for s in (seeds if seeds else e.seeds):
            out.append((e, s))
    return out


def run_dir(exp_id: str, seed: int, out_dir: str = RUNS_DIR) -> Path:
    return Path(out_dir) / exp_id / f"seed{seed}"


def is_done(rd: Path) -> bool:
    return (rd / "summary.json").exists()


# Each configuration may differ from its reference only in the listed MoEConfig fields.
REFERENCE = {"topk": "K2", "topp": "P50", "dense": "DENSE"}
ALLOWED_DIFF = {
    "lr-sweep": set(), "alpha-sweep": {"bal_coef"},
    "main-10m": {"k", "p"}, "entropy-10m": {"ent_coef"}, "gatenorm-10m": {"gate_norm"},
    "main-30m": {"k", "p"}, "gatenorm-30m": {"gate_norm"},
}


def check_controls() -> List[str]:
    """Problems that would make a comparison unfair. Empty list = fine."""
    from moe import MoEConfig 

    problems: List[str] = []
    ids = [e.id for e in EXPS]
    if len(ids) != len(set(ids)):
        problems.append("duplicate experiment IDs")
    for e in EXPS:
        if e.group not in GROUPS:
            problems.append(f"{e.id}: unknown group {e.group!r}")
            continue
        try:
            got = MoEConfig(**e.model).to_dict()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{e.id}: invalid model overrides ({exc})")
            continue
        if not e.seeds:
            problems.append(f"{e.id}: no seeds")
        base = MoEConfig(**BY_ID[REFERENCE[e.family]].model).to_dict()
        changed = {k for k in got if got[k] != base[k]}
        if not changed <= ALLOWED_DIFF[e.group]:
            problems.append(f"{e.id}: differs from {REFERENCE[e.family]} in {sorted(changed)}; "
                            f"group {e.group} may only change {sorted(ALLOWED_DIFF[e.group])}")
        if e.group != "lr-sweep" and e.lr != PEAK_LR:
            problems.append(f"{e.id}: only the lr-sweep may change the learning rate")
    for fam in ("topk", "topp", "dense"):  # each LR sweep tunes exactly its reference model over one grid
        grid = sorted(e.lr for e in EXPS if e.group == "lr-sweep" and e.family == fam)
        if grid != sorted(float(s) for s in LR_GRID):
            problems.append(f"lr-sweep grid for {fam} is {grid}")
    a_grids = {fam: sorted(e.model["bal_coef"] for e in EXPS if e.group == "alpha-sweep" and e.family == fam)
               for fam in ("topk", "topp")}
    if a_grids["topk"] != a_grids["topp"] or MoEConfig().bal_coef not in a_grids["topk"]:
        problems.append(f"alpha sweeps differ between families or miss the default: {a_grids}")
    return problems
