"""checkpoint_probe.py - forward-only probes of the final checkpoints.

    python analysis/checkpoint_probe.py [--runs runs] [--out results/checkpoint_probe.json]

Per layer: router probability mass of the selected experts and the FFN output norm relative to the residual stream;
the loss change when one layer at a time is switched to the other gate normalization; and, for top-p models, the
loss and expert count with the cap of 4 experts removed (max_k = 8). All on the first 131,072 validation tokens.
Needs runs/<ID>/seed<N>/ckpt/final.pt and the data folder recorded in each run's config.json.
"""
import argparse, json, sys, math, dataclasses
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moe import MoEConfig, MoETransformer

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--runs", default="runs")
_ap.add_argument("--out", default="results/checkpoint_probe.json")
_args = _ap.parse_args()
RUNS_DIR = Path(_args.runs)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NTOK = 131072
L = 256

def load(rid, s):
    st = torch.load(RUNS_DIR / rid / f"seed{s}" / "ckpt" / "final.pt", map_location="cpu", weights_only=False)
    cfg = MoEConfig.from_dict(st["model_cfg"])
    m = MoETransformer(cfg); m.load_state_dict(st["model"]); m.to(dev).eval()
    return m, cfg

def val_batches(data_dir):
    val = np.memmap(Path(data_dir) / "val.bin", dtype=np.uint16, mode="r")
    n = NTOK // L
    for w0 in range(0, n, 64):
        ws = np.arange(w0, min(w0 + 64, n))
        offs = ws[:, None] * L + np.arange(L + 1)[None, :]
        c = torch.from_numpy(np.asarray(val[offs]).astype(np.int64))
        yield c[:, :-1].to(dev), c[:, 1:].to(dev)

@torch.no_grad()
def probe(m, cfg, data_dir):
    stats = {i: dict(mass=0., rel=0., xn=0., yn=0., n=0) for i in range(cfg.n_layer)}
    hooks = []
    for i, blk in enumerate(m.blocks):
        def hk(mod, inp, out, i=i):
            x = inp[0]; y = out[0]
            xf = x.reshape(-1, x.shape[-1]).float()
            logits = mod.router(xf).float(); probs = logits.softmax(-1)
            from moe import select_experts
            mask = select_experts(probs, mod.cfg.route_mode, mod.cfg.k, mod.cfg.p, mod.cfg.min_k, mod.cfg.max_k).float()
            st = stats[i]
            st["mass"] += float((probs * mask).sum(-1).sum()); st["n"] += xf.shape[0]
            st["yn"] += float(y.reshape(-1, y.shape[-1]).float().norm(dim=-1).sum())
        hooks.append(blk.ffn.register_forward_hook(hk))
        def hres(mod, inp, i=i):
            stats[i]["xn"] += float(inp[0].reshape(-1, inp[0].shape[-1]).float().norm(dim=-1).sum())
        hooks.append(blk.register_forward_pre_hook(hres))
    tot, cnt = 0., 0
    for x, y in val_batches(data_dir):
        out = m(x)
        tot += float(F.cross_entropy(out.logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1), reduction="sum")); cnt += y.numel()
    for h in hooks: h.remove()
    return tot / cnt, {i: dict(sel_mass=s["mass"] / s["n"], ffn_over_resid=s["yn"] / s["xn"]) for i, s in stats.items()}

@torch.no_grad()
def loss_only(m, cfg, data_dir):
    tot, cnt, E = 0., 0, []
    for x, y in val_batches(data_dir):
        out = m(x, return_routing=True)
        tot += float(F.cross_entropy(out.logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1), reduction="sum")); cnt += y.numel()
        E.append([float(mk.float().sum(-1).mean()) for mk in out.routing.masks])
    return tot / cnt, np.mean(E, 0)

RUNS = [("K2", [0, 1, 2]), ("K2-renorm", [0, 1, 2]), ("P50", [0, 1]), ("P50-renorm", [0, 1]), ("P60", [0, 1, 2]), ("P70", [0, 1]),
        ("L-K2", [0, 1, 2]), ("L-P60", [0, 1, 2]), ("L-P65", [0, 1, 2])]
res = {}
for rid, seeds in RUNS:
    for s in seeds:
        m, cfg = load(rid, s)
        dd = json.load(open(RUNS_DIR / rid / f"seed{s}" / "config.json"))["train"]["data_dir"]
        base, per = probe(m, cfg, dd)
        r = dict(loss=base, layers=per)
        # one layer at a time switched to the other gate normalization at inference
        other = "renorm" if cfg.gate_norm == "raw" else "raw"
        sw = []
        for li, blk in enumerate(m.blocks):
            orig = blk.ffn.cfg
            blk.ffn.cfg = dataclasses.replace(orig, gate_norm=other)
            l, _ = loss_only(m, cfg, dd); sw.append(l - base)
            blk.ffn.cfg = orig
        r[f"switch_one_layer_to_{other}"] = sw
        if cfg.route_mode == "topp":
            orig = cfg.max_k
            m.set_routing(max_k=8); l, E = loss_only(m, cfg, dd); r["uncapped_maxk8"] = dict(loss=l, experts=E.tolist())
            m.set_routing(max_k=orig)
        res[f"{rid}/s{s}"] = r
        print(f"{rid}/s{s} loss {base:.4f} sel_mass {[round(v['sel_mass'],3) for v in per.values()]} ffn/resid {[round(v['ffn_over_resid'],3) for v in per.values()]} "
              f"switch->{other} dloss {[round(v,4) for v in sw]}" + (f" | maxk8 loss {r['uncapped_maxk8']['loss']:.4f} experts {[round(e,3) for e in r['uncapped_maxk8']['experts']]}" if "uncapped_maxk8" in r else ""), flush=True)
Path(_args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(_args.out, "w"), indent=1)
