#!/usr/bin/env python3
"""Smoke tests for moe.py.  Run:  python tests/test_moe.py [--device cpu|mps]   (also works with pytest)

These check the things that would silently invalidate a routing comparison: the selection rules,
dense and sparse implementations agreeing, causality, whether the router gets gradients, loss
values, FLOP accounting against PyTorch's own counter, and that the model can actually learn.
"""
import argparse
import math
import os
import sys
import time
import traceback
from dataclasses import replace

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from moe import (
    MoEConfig, MoETransformer, RoutingAccumulator, active_params, count_params, estimate_flops,
    load_balance_loss, router_entropy, router_z_loss, select_experts, summarize_routing,
)

DEVICE = os.environ.get("MOE_TEST_DEVICE", "cpu")

# Tiny config: fast on any machine, but exercises every code path.
BASE = MoEConfig(vocab_size=97, n_layer=2, d_model=64, n_head=4, seq_len=32, n_experts=4,
                 expert_hidden=96, dense_hidden=192, max_k=3, k=2, p=0.5)


def make(cfg, seed=0):
    torch.manual_seed(seed)
    return MoETransformer(cfg).to(DEVICE)


def batch(cfg, b=2, t=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, cfg.vocab_size, (b, t + 1), generator=g)
    return x[:, :-1].to(DEVICE), x[:, 1:].to(DEVICE)


def rand_probs(n=512, e=8, seed=0, sharp=2.0):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(n, e, generator=g) * sharp, -1).to(DEVICE)


# ----------------------------------------------------------------------------- selection rules
def test_topk_selects_exactly_k():
    probs = rand_probs()
    for k in (1, 2, 3, 8):
        mask = select_experts(probs, "topk", k=k)
        assert mask.dtype == torch.bool and (mask.sum(-1) == k).all()
        # the selected ones really are the k largest
        kth = probs.topk(k, -1).values[:, -1:]
        assert (probs[mask] >= kth.expand_as(probs)[mask] - 1e-9).all()


def test_topp_properties():
    probs = rand_probs()
    e = probs.shape[-1]
    # tiny p -> exactly one expert (min_k = 1); p = 1 with no cap -> every expert
    assert (select_experts(probs, "topp", p=1e-6).sum(-1) == 1).all()
    assert (select_experts(probs, "topp", p=1.0, max_k=e).sum(-1) == e).all()
    # cap and floor are respected
    assert select_experts(probs, "topp", p=1.0, max_k=3).sum(-1).max() == 3
    assert select_experts(probs, "topp", p=1e-6, min_k=2).sum(-1).min() == 2
    # monotone in p
    means = [select_experts(probs, "topp", p=p, max_k=e).sum(-1).float().mean().item()
             for p in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)]
    assert all(a <= b + 1e-9 for a, b in zip(means, means[1:])), means
    # minimality: selected mass >= p, and dropping the last selected expert falls below p
    for p in (0.3, 0.5, 0.8):
        mask = select_experts(probs, "topp", p=p, max_k=e)
        sp = probs.sort(-1, descending=True).values
        cnt = mask.sum(-1)
        mass = sp.cumsum(-1).gather(-1, (cnt - 1).unsqueeze(-1)).squeeze(-1)
        assert (mass >= p - 1e-6).all()
        before = (sp.cumsum(-1) - sp).gather(-1, (cnt - 1).unsqueeze(-1)).squeeze(-1)
        assert (before < p + 1e-6).all()
        # selected experts are the top ones
        topc = torch.zeros_like(probs).scatter(-1, probs.argsort(-1, descending=True), (
            torch.arange(e, device=probs.device).expand_as(probs) < cnt.unsqueeze(-1)).float()) > 0.5
        assert (topc == mask).all()


def test_topp_exact_thresholds():
    # Dyadic probabilities are exact in floating point, so the boundary "mass == p" is hit exactly.
    # Rule under test: smallest set whose mass reaches p (>=), so mass exactly p stops there.
    probs = torch.tensor([[0.125, 0.5, 0.125, 0.25]], device=DEVICE)  # sorted: .5 .25 .125 .125
    expect = {0.5: 1, 0.51: 2, 0.75: 2, 0.76: 3, 0.875: 3, 0.88: 4, 1.0: 4}
    for p, n in expect.items():
        mask = select_experts(probs, "topp", p=p)
        assert int(mask.sum()) == n, (p, int(mask.sum()), n)
    # and it is the top experts that get picked: p=0.75 picks the 0.5 and 0.25 entries
    assert select_experts(probs, "topp", p=0.75).tolist() == [[False, True, False, True]]


# ----------------------------------------------------------------------------- layer correctness
def test_topk_all_experts_equals_full_mixture():
    cfg = replace(BASE, route_mode="topk", k=BASE.n_experts, gate_norm="renorm")
    m = make(cfg)
    layer = m.blocks[0].ffn
    x = torch.randn(2, 8, cfg.d_model, device=DEVICE)
    y, _ = layer(x)
    xf = x.reshape(-1, cfg.d_model)
    probs = torch.softmax(layer.router(xf), -1)
    ref = sum(probs[:, e:e + 1] * (torch.nn.functional.silu(xf @ layer.experts.w1[e]) *
                                   (xf @ layer.experts.w3[e])) @ layer.experts.w2[e]
              for e in range(cfg.n_experts))
    assert torch.allclose(y.reshape(-1, cfg.d_model), ref, atol=1e-5, rtol=1e-4)


def _equiv_case(**over):
    cfg = replace(BASE, **over)
    a = make(cfg)
    b = make(replace(cfg, impl="sparse"))
    b.load_state_dict(a.state_dict())
    x, y = batch(cfg)
    oa, ob = a(x, y), b(x, y)
    assert torch.allclose(oa.logits, ob.logits, atol=1e-5, rtol=1e-4)
    assert torch.allclose(oa.loss, ob.loss, atol=1e-6)
    oa.loss.backward(), ob.loss.backward()
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-5, rtol=1e-3), n


def test_dense_and_sparse_impl_match():
    _equiv_case(route_mode="topk", k=2)
    _equiv_case(route_mode="topk", k=1, gate_norm="renorm")
    _equiv_case(route_mode="topp", p=0.6, max_k=3)
    _equiv_case(route_mode="topp", p=0.4, gate_norm="renorm", ent_coef=1e-3, z_coef=1e-3)
    _equiv_case(route_mode="topk", k=2, n_shared=1, shared_hidden=64)


def test_causality():
    for cfg in (replace(BASE, route_mode="topk"), replace(BASE, route_mode="topp", p=0.5)):
        m = make(cfg).eval()
        x, _ = batch(cfg, b=1)
        x2 = x.clone()
        x2[:, 16:] = (x2[:, 16:] + 7) % cfg.vocab_size  # change only the future
        with torch.no_grad():
            a, b = m(x).logits, m(x2).logits
        assert torch.allclose(a[:, :16], b[:, :16], atol=1e-5), "future tokens leaked into the past"
        assert not torch.allclose(a[:, 16:], b[:, 16:], atol=1e-5)


def test_sdpa_matches_manual_attention():
    cfg = replace(BASE, route_mode="topk")
    a = make(cfg)
    b = make(replace(cfg, use_sdpa=False))
    b.load_state_dict(a.state_dict())
    x, _ = batch(cfg)
    assert torch.allclose(a(x).logits, b(x).logits, atol=1e-5, rtol=1e-4)


# ----------------------------------------------------------------------------- gradients to the router
def _router_grad(**over):
    cfg = replace(BASE, bal_coef=0.0, ent_coef=0.0, z_coef=0.0, **over)  # language-model loss only
    m = make(cfg)
    x, y = batch(cfg)
    m(x, y).loss.backward()
    return max(float(blk.ffn.router.weight.grad.abs().max()) for blk in m.blocks)


def test_router_gets_gradient_only_when_it_should():
    assert _router_grad(route_mode="topk", k=1, gate_norm="raw") > 1e-6
    assert _router_grad(route_mode="topk", k=2, gate_norm="renorm") > 1e-6
    assert _router_grad(route_mode="topp", p=0.5, gate_norm="raw") > 1e-6
    # The trap this test documents: renormalised single-expert gates are constant 1.0, so the
    # language-model loss cannot train the router (top-1 with renorm, or p small enough that every
    # token picks one expert). What is left is floating-point residue, about 1e-11 against about
    # 1e-3 for a router that is really being trained.
    healthy = _router_grad(route_mode="topk", k=1, gate_norm="raw")
    assert _router_grad(route_mode="topk", k=1, gate_norm="renorm") < 1e-6 * healthy
    assert _router_grad(route_mode="topp", p=1e-6, gate_norm="renorm") < 1e-6 * healthy


# ----------------------------------------------------------------------------- loss terms
def test_loss_term_values():
    t, e = 64, 8
    uniform = torch.full((t, e), 1.0 / e, device=DEVICE)
    # perfectly balanced top-2 assignment: token i -> experts (i, i+1) mod e
    mask = torch.zeros(t, e, device=DEVICE)
    for i in range(t):
        mask[i, i % e] = 1
        mask[i, (i + 1) % e] = 1
    assert abs(float(load_balance_loss(uniform, mask)) - 1.0) < 1e-5  # 1 at balance, whatever k
    assert abs(float(load_balance_loss(uniform, mask, "tokens")) - 2.0) < 1e-5  # scales with k
    collapsed = torch.zeros(t, e, device=DEVICE)
    collapsed[:, 0] = 1
    assert abs(float(load_balance_loss(collapsed, collapsed)) - e) < 1e-5  # N when fully collapsed
    logp = uniform.log()
    assert abs(float(router_entropy(uniform, logp)) - math.log(e)) < 1e-5
    assert abs(float(router_z_loss(torch.zeros(t, e, device=DEVICE))) - math.log(e) ** 2) < 1e-5


def test_total_loss_composition_and_init_loss():
    cfg = replace(BASE, bal_coef=0.02, ent_coef=0.003, z_coef=0.001, route_mode="topp")
    m = make(cfg)
    x, y = batch(cfg)
    o = m(x, y)
    want = o.lm_loss + 0.02 * o.bal_loss + 0.003 * o.ent_loss + 0.001 * o.z_loss
    assert torch.allclose(o.loss, want, atol=1e-6)
    assert abs(float(o.lm_loss) - math.log(cfg.vocab_size)) < 0.15, float(o.lm_loss)  # untrained = uniform
    assert float(o.bal_loss) < 1.2  # near-uniform routing at init


# ----------------------------------------------------------------------------- stats and control
def test_routing_stats_consistency():
    cfg = replace(BASE, route_mode="topp", p=0.6, max_k=3)
    m = make(cfg)
    x, _ = batch(cfg)
    r = m(x, return_routing=True).routing
    n_layers, e = cfg.n_layer, cfg.n_experts
    assert r.avg_experts.shape == (n_layers,) and r.load.shape == (n_layers, e)
    assert r.count_hist.shape == (n_layers, e + 1)
    assert torch.allclose(r.load.sum(-1), torch.ones(n_layers, device=DEVICE), atol=1e-5)
    assert torch.allclose(r.count_hist.sum(-1), torch.ones(n_layers, device=DEVICE), atol=1e-5)
    counts = torch.arange(e + 1, device=DEVICE).float()
    assert torch.allclose((r.count_hist * counts).sum(-1), r.avg_experts, atol=1e-5)
    for layer, mask in enumerate(r.masks):
        assert mask.shape == (x.numel(), e)
        assert abs(float(mask.sum(-1).float().mean()) - float(r.avg_experts[layer])) < 1e-5
        assert (mask.sum(-1) >= 1).all() and (mask.sum(-1) <= 3).all()
    assert m(x).routing.masks is None  # masks only on request


def test_accumulator_matches_single_pass():
    cfg = replace(BASE, route_mode="topp", p=0.6)
    m = make(cfg).eval()
    x, _ = batch(cfg, b=4)
    with torch.no_grad():
        whole = m(x).routing
        acc = RoutingAccumulator()
        for part in x.split(2):
            acc.update(m(part).routing, part.numel())
    res = acc.compute()
    assert torch.allclose(res["avg_experts"], whole.avg_experts.cpu(), atol=1e-5)
    assert torch.allclose(res["load"], whole.load.cpu(), atol=1e-5)
    assert torch.allclose(res["count_hist"], whole.count_hist.cpu(), atol=1e-5)
    flat = summarize_routing(res)
    assert "avg_experts/mean" in flat and all(isinstance(v, float) for v in flat.values())


def test_set_routing_changes_expert_count():
    m = make(replace(BASE, route_mode="topp", p=0.2, max_k=4)).eval()
    x, _ = batch(m.cfg)
    avg = []
    with torch.no_grad():
        for p in (0.2, 0.5, 0.9):
            m.set_routing(p=p)
            avg.append(float(m(x).routing.avg_experts.mean()))
    assert avg[0] < avg[1] < avg[2], avg
    m.set_routing(route_mode="topk", k=3)
    with torch.no_grad():
        assert abs(float(m(x).routing.avg_experts.mean()) - 3.0) < 1e-6
    try:
        m.set_routing(k=99)
        raise RuntimeError("invalid k was accepted")
    except AssertionError:
        pass
    assert m.cfg.k == 3  # a rejected change must not leave the config half-updated


def test_determinism_and_checkpoint_roundtrip():
    cfg = replace(BASE, route_mode="topp")
    x, y = batch(cfg)
    a, b = make(cfg, seed=3), make(cfg, seed=3)
    assert torch.equal(a(x, y).logits, b(x, y).logits)
    c = make(cfg, seed=4)
    c.load_state_dict(a.state_dict())
    assert torch.equal(c(x).logits, a(x).logits)
    assert not torch.equal(make(cfg, seed=5)(x).logits, a(x).logits)


# ----------------------------------------------------------------------------- accounting
def test_param_counts_match_real_model():
    for over in (dict(), dict(n_shared=1, shared_hidden=64), dict(qk_norm=False, tie_embeddings=False),
                 dict(arch="dense")):
        cfg = replace(BASE, **over)
        m = make(cfg)
        pc = count_params(cfg)
        assert pc["total"] == m.num_params(), (over, pc["total"], m.num_params())
        if cfg.arch == "moe":  # using every expert means every parameter is active
            assert active_params(cfg, cfg.n_experts)["total"] == pc["total"]
        assert active_params(cfg, 2)["total"] <= pc["total"]


def _measured_flops(model, x):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        out = model(x)
    return fc.get_total_flops() / x.numel(), out


def test_flop_estimate_matches_pytorch_counter():
    x_cfg = replace(BASE, use_sdpa=False)  # explicit attention, so the counter sees the matmuls
    x, _ = batch(x_cfg, b=3)
    cases = {
        "moe dense-impl top-2": replace(x_cfg, route_mode="topk", k=2, impl="dense"),
        "moe sparse-impl top-p": replace(x_cfg, route_mode="topp", p=0.6, max_k=3, impl="sparse"),
        "moe sparse-impl + shared": replace(x_cfg, route_mode="topk", k=3, n_shared=1, shared_hidden=64,
                                            impl="sparse"),
        "dense baseline": replace(x_cfg, arch="dense"),
    }
    for name, cfg in cases.items():
        m = make(cfg).eval()
        measured, out = _measured_flops(m, x)
        avg = out.routing.avg_experts.tolist() if out.routing is not None else None
        est = estimate_flops(cfg, avg, seq_len=x.shape[1], attn_context="full", executed=True)["forward"]
        assert abs(measured - est) / est < 1e-6, (name, measured, est)
    # effective FLOPs of a sparse top-k model equal the executed FLOPs of its sparse implementation,
    # and are strictly lower than what the dense-masked implementation runs
    cfg = replace(x_cfg, route_mode="topk", k=2)
    eff = estimate_flops(cfg, seq_len=32, attn_context="full")["forward"]
    ex_dense = estimate_flops(cfg, seq_len=32, attn_context="full", executed=True)["forward"]
    ex_sparse = estimate_flops(replace(cfg, impl="sparse"), seq_len=32, attn_context="full", executed=True)["forward"]
    assert eff == ex_sparse < ex_dense
    # top-p needs a measured count; silently assuming one would hide the compute differences
    try:
        estimate_flops(replace(cfg, route_mode="topp"))
        raise RuntimeError("top-p FLOPs without avg_experts should be refused")
    except ValueError:
        pass


def test_matched_compute_dense_baseline():
    # the dense baseline is sized to match top-2 in active parameters and effective FLOPs
    cfg = MoEConfig()
    moe_fl = estimate_flops(replace(cfg, route_mode="topk", k=2))["forward"]
    dense_fl = estimate_flops(replace(cfg, arch="dense"))["forward"]
    assert abs(moe_fl - dense_fl) / dense_fl < 0.01, (moe_fl, dense_fl)
    pa = active_params(replace(cfg, route_mode="topk", k=2), 2)["total"]
    assert abs(pa - count_params(replace(cfg, arch="dense"))["total"]) / pa < 0.01


# ----------------------------------------------------------------------------- can it learn
def _overfit(cfg, steps=250):
    m = make(cfg)
    x, y = batch(cfg, b=2)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    first = None
    for _ in range(steps):
        out = m(x, y)
        first = first if first is not None else float(out.lm_loss)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    last = m(x, y)
    assert all(torch.isfinite(p).all() for p in m.parameters())
    return first, float(last.lm_loss), last.routing


def test_overfits_tiny_batch():
    for name, cfg in {
        "top-2": replace(BASE, route_mode="topk", k=2),
        "top-p": replace(BASE, route_mode="topp", p=0.5, ent_coef=1e-4),
        "dense": replace(BASE, arch="dense"),
    }.items():
        first, last, routing = _overfit(cfg)
        assert first > 4.3 and last < 0.5, (name, first, last)
        if routing is not None:
            assert (routing.load.max(-1).values < 0.95).all(), f"{name}: routing collapsed to one expert"


# ----------------------------------------------------------------------------- runner
def main():
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=DEVICE)
    args = ap.parse_args()
    DEVICE = args.device
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    print(f"device={DEVICE}  torch={torch.__version__}  {len(tests)} tests")
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"  PASS  {name}  ({time.time() - t0:.1f}s)")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()