"""moe.py - a small, modern MoE decoder-only transformer.

Architecture:
    pre-norm RMSNorm, rotary positions, optional QK-norm, tied embeddings, no biases,
    SwiGLU experts, a softmax router computed in fp32, optional always-on shared experts.

Routing modes (cfg.route_mode):
    "topk"  fixed k experts per token.
    "topp"  smallest set of experts, in descending probability order, whose cumulative
            router probability reaches p; at least min_k and at most max_k experts.

Gate weights (cfg.gate_norm):
    "raw"     gate_i = P_i for the selected experts (Switch / OLMoE style).
    "renorm"  gate_i = P_i / sum of selected P (Mixtral / Qwen3 style).
    NOTE: with "renorm", a token that selects exactly one expert has gate 1.0 whatever the
    router says, so the language-model loss sends NO gradient to the router for that token.
    That breaks top-1 and low-p top-p. Default is "raw" for this reason (see test_moe.py).

Two implementations of the expert computation, numerically equivalent:
    impl="dense"   every expert runs on every token, outputs are multiplied by the gates
                   (zero for unselected experts).
    impl="sparse"  each expert runs only on its routed tokens. Real FLOP savings, dynamic
                   shapes. 

Compute accounting: estimate_flops() gives analytic FLOPs per token, both "effective"
(what a sparse implementation needs) and "executed" (what this implementation runs).
Checked against torch.utils.flop_counter in test_moe.py.

loss terms, each averaged over MoE layers:
    bal  load balance, N * sum_i f_i * Q_i     (Switch Transformer)
    ent  mean router entropy                   (Huang et al. 2024, pushes routing to be sharp)
    z    mean squared router logsumexp         (ST-MoE, optional)
    total = lm + bal_coef * bal + ent_coef * ent + z_coef * z

Usage:
    from moe import MoEConfig, MoETransformer
    model = MoETransformer(MoEConfig(route_mode="topp", p=0.5))
    out = model(idx, targets)          # out.loss, out.lm_loss, out.routing, ...
    print(model.summary())
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class MoEConfig:
    # --- backbone ---
    vocab_size: int = 4096
    n_layer: int = 4
    d_model: int = 256
    n_head: int = 4
    seq_len: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    qk_norm: bool = True
    tie_embeddings: bool = True
    use_sdpa: bool = True  # False: explicit softmax attention (used to cross-check FLOP counts)
    # --- feed-forward ---
    arch: str = "moe"  # "moe" or "dense"
    dense_hidden: int = 1024  # dense baseline FFN width; 2 * expert_hidden matches top-2
    n_experts: int = 8
    expert_hidden: int = 512
    n_shared: int = 0  # always-on shared experts (DeepSeek / Qwen style); 0 = none
    shared_hidden: int = 512  # width of EACH shared expert
    # --- routing ---
    route_mode: str = "topk"  # "topk" or "topp"
    k: int = 2  # topk: experts per token
    p: float = 0.5  # topp: cumulative probability threshold
    min_k: int = 1  # topp: lower bound on experts per token
    max_k: int = 4  # topp: upper bound on experts per token
    gate_norm: str = "raw"  # "raw" or "renorm"
    impl: str = "dense"  # "dense" or "sparse"
    # --- losses ---
    bal_coef: float = 1e-2
    ent_coef: float = 0.0
    z_coef: float = 0.0
    bal_f_norm: str = "assignments"  # "assignments": f sums to 1, so bal is 1 at perfect balance for any k
    #                                  "tokens": f_i = fraction of tokens that selected i (Huang et al.; sums to avg k)
    # --- init ---
    init_std: float = 0.02
    router_init_std: float = 0.02

    def __post_init__(self):
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        assert self.head_dim % 2 == 0, "head_dim must be even for rotary embeddings"
        assert self.arch in ("moe", "dense"), self.arch
        assert self.route_mode in ("topk", "topp"), self.route_mode
        assert self.gate_norm in ("raw", "renorm"), self.gate_norm
        assert self.impl in ("dense", "sparse"), self.impl
        assert self.bal_f_norm in ("assignments", "tokens"), self.bal_f_norm
        assert 1 <= self.k <= self.n_experts, "need 1 <= k <= n_experts"
        assert 0.0 < self.p <= 1.0, "need 0 < p <= 1"
        assert 1 <= self.min_k <= self.max_k <= self.n_experts, "need 1 <= min_k <= max_k <= n_experts"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MoEConfig":
        return cls(**d)


# Routing
def select_experts(
    probs: torch.Tensor,
    mode: str,
    k: int = 2,
    p: float = 0.5,
    min_k: int = 1,
    max_k: Optional[int] = None,
) -> torch.Tensor:
    n_exp = probs.shape[-1]
    if mode == "topk":
        idx = probs.topk(k, dim=-1).indices
        sel = torch.zeros_like(probs).scatter(-1, idx, 1.0)
        return sel > 0.5
    if mode != "topp":
        raise ValueError(f"unknown route mode {mode!r}")
    max_k = n_exp if max_k is None else max_k
    sp, si = probs.sort(dim=-1, descending=True, stable=True)
    keep = (sp.cumsum(-1) - sp) < p 
    keep[..., :min_k] = True
    keep[..., max_k:] = False
    sel = torch.zeros_like(probs).scatter(-1, si, keep.to(probs.dtype))
    return sel > 0.5

#  loss functions
def load_balance_loss(probs: torch.Tensor, mask: torch.Tensor, f_norm: str = "assignments") -> torch.Tensor:
    "Switch-style balance loss N * sum_i f_i * Q_i. probs: [T, E]; mask: [T, E] with 0/1 entries."
    n_tokens, n_exp = probs.shape
    counts = mask.to(probs.dtype).sum(0)
    if f_norm == "assignments":
        f = counts / counts.sum().clamp_min(1.0)
    else:
        f = counts / n_tokens
    q = probs.mean(0)
    return n_exp * (f.detach() * q).sum()

def router_entropy(probs: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
    return -(probs * logp).sum(-1).mean()

def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(logits, dim=-1).pow(2).mean()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight


def build_rope(seq_len: int, head_dim: int, theta: float):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv)  # [T, hd/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, hd]; cos, sin: [T, hd/2]. Half-split (NeoX) rotation."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None].to(x.dtype)
    sin = sin[None, None].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        d, hd = cfg.d_model, cfg.head_dim
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        if cfg.qk_norm:
            self.q_norm = RMSNorm(hd, cfg.norm_eps)
            self.k_norm = RMSNorm(hd, cfg.norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, cfg.n_head, cfg.head_dim).permute(2, 0, 3, 1, 4)
        if cfg.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cfg.use_sdpa:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (cfg.head_dim ** -0.5)
            causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~causal, float("-inf")).softmax(-1)
            y = att @ v
        return self.o(y.transpose(1, 2).reshape(b, t, d))


class SwiGLU(nn.Module):
    """Dense gated FFN: w2( silu(w1 x) * w3 x )."""

    def __init__(self, d: int, h: int):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False)  # gate projection
        self.w3 = nn.Linear(d, h, bias=False)  # up projection
        self.w2 = nn.Linear(h, d, bias=False)  # down projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ExpertBank(nn.Module):
    """SwiGLU experts stored as stacked tensors w1, w3: [E, d, h]; w2: [E, h, d]."""

    def __init__(self, n_exp: int, d: int, h: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(n_exp, d, h))
        self.w3 = nn.Parameter(torch.empty(n_exp, d, h))
        self.w2 = nn.Parameter(torch.empty(n_exp, h, d))

    def forward_dense(self, x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        """All experts on all tokens; gates [T, E] zero out the unselected ones.

        Three large GEMMs: gate and up projections as [d, E*h], and the gated hidden states
        through the stacked down projection [E*h, d], so no [E, T, d] intermediate exists.
        """
        n_exp, d, h = self.w1.shape
        w1 = self.w1.permute(1, 0, 2).reshape(d, n_exp * h)
        w3 = self.w3.permute(1, 0, 2).reshape(d, n_exp * h)
        a = F.silu(x @ w1) * (x @ w3)  # [T, E*h]
        a = (a.view(-1, n_exp, h) * gates.unsqueeze(-1).to(a.dtype)).reshape(-1, n_exp * h)
        return a @ self.w2.reshape(n_exp * h, d)

    def forward_sparse(self, x: torch.Tensor, gates: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Each expert on its routed tokens only (real FLOP savings, dynamic shapes)."""
        out = torch.zeros_like(x)
        for e in range(self.w1.shape[0]):
            idx = mask[:, e].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            xe = x.index_select(0, idx)
            ye = (F.silu(xe @ self.w1[e]) * (xe @ self.w3[e])) @ self.w2[e]
            ye = ye * gates.index_select(0, idx)[:, e].unsqueeze(-1).to(ye.dtype)
            out.index_add_(0, idx, ye)
        return out


@dataclass
class LayerRouting:
    """Per-layer routing output. bal/ent/z carry gradients; everything else is detached."""

    bal: torch.Tensor
    ent: torch.Tensor
    z: torch.Tensor
    avg_experts: torch.Tensor  # scalar: mean routed experts per token
    load: torch.Tensor  # [E] share of expert assignments (sums to 1)
    count_hist: torch.Tensor  # [E+1] fraction of tokens that selected exactly c experts
    max_prob: torch.Tensor  # scalar: mean top router probability
    mask: Optional[torch.Tensor] = None  # [T, E] bool, only if requested
    probs: Optional[torch.Tensor] = None  # [T, E], only if requested


class MoELayer(nn.Module):
    """Router + expert bank (+ optional shared experts). Replaces the FFN of a block."""

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg  # shared with the model, so set_routing() takes effect everywhere
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = ExpertBank(cfg.n_experts, cfg.d_model, cfg.expert_hidden)
        self.shared = SwiGLU(cfg.d_model, cfg.n_shared * cfg.shared_hidden) if cfg.n_shared > 0 else None

    def forward(self, x: torch.Tensor, keep: bool = False):
        cfg = self.cfg
        b, t, d = x.shape
        xf = x.reshape(-1, d)

        logits = self.router(xf).float()  # routing math always in fp32
        logp = F.log_softmax(logits, dim=-1)
        probs = logp.exp()

        mask = select_experts(probs.detach(), cfg.route_mode, cfg.k, cfg.p, cfg.min_k, cfg.max_k)
        maskf = mask.to(probs.dtype)
        gates = probs * maskf
        if cfg.gate_norm == "renorm":
            gates = gates / gates.sum(-1, keepdim=True)

        if cfg.impl == "dense":
            y = self.experts.forward_dense(xf, gates)
        else:
            y = self.experts.forward_sparse(xf, gates, mask)
        if self.shared is not None:
            y = y + self.shared(xf)

        n_exp = cfg.n_experts
        count = maskf.sum(-1)  # [T]
        with torch.no_grad():
            avg_experts = count.mean()
            load = maskf.sum(0) / maskf.sum().clamp_min(1.0)
            count_hist = F.one_hot(count.long(), n_exp + 1).float().mean(0)
            max_prob = probs.max(-1).values.mean()
        info = LayerRouting(
            bal=load_balance_loss(probs, maskf, cfg.bal_f_norm),
            ent=router_entropy(probs, logp),
            z=router_z_loss(logits),
            avg_experts=avg_experts,
            load=load,
            count_hist=count_hist,
            max_prob=max_prob,
            mask=mask.detach() if keep else None,
            probs=probs.detach() if keep else None,
        )
        return y.view(b, t, d), info


class Block(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.is_moe = cfg.arch == "moe"
        self.ffn = MoELayer(cfg) if self.is_moe else SwiGLU(cfg.d_model, cfg.dense_hidden)

    def forward(self, x, cos, sin, keep: bool = False):
        x = x + self.attn(self.norm1(x), cos, sin)
        h = self.norm2(x)
        if self.is_moe:
            y, info = self.ffn(h, keep)
        else:
            y, info = self.ffn(h), None
        return x + y, info


@dataclass
class RoutingStats:
    """Routing statistics stacked over MoE layers (all detached)."""

    avg_experts: torch.Tensor  # [L]
    load: torch.Tensor  # [L, E]
    count_hist: torch.Tensor  # [L, E+1]
    entropy: torch.Tensor  # [L]
    max_prob: torch.Tensor  # [L]
    masks: Optional[List[torch.Tensor]] = None  # L x [T, E] bool
    probs: Optional[List[torch.Tensor]] = None  # L x [T, E]


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None  # total training loss (with grad)
    lm_loss: Optional[torch.Tensor] = None  # detached, for logging
    bal_loss: Optional[torch.Tensor] = None  # detached, mean over MoE layers
    ent_loss: Optional[torch.Tensor] = None
    z_loss: Optional[torch.Tensor] = None
    routing: Optional[RoutingStats] = None


class MoETransformer(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = build_rope(cfg.seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._init_weights()

    def _init_weights(self):
        cfg = self.cfg
        resid_scale = 1.0 / math.sqrt(2 * cfg.n_layer)  # GPT-2 style scaling of residual-stream writers
        for name, prm in self.named_parameters():
            if prm.ndim < 2:
                continue  # norm weights stay at 1
            std = cfg.router_init_std if name.endswith("router.weight") else cfg.init_std
            if name.endswith(("attn.o.weight", "experts.w2", "w2.weight")):
                std *= resid_scale
            nn.init.normal_(prm, mean=0.0, std=std)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, return_routing: bool = False):
        cfg = self.cfg
        b, t = idx.shape
        assert t <= cfg.seq_len, f"sequence length {t} exceeds cfg.seq_len {cfg.seq_len}"
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]
        infos: List[LayerRouting] = []
        for blk in self.blocks:
            x, info = blk(x, cos, sin, keep=return_routing)
            if info is not None:
                infos.append(info)
        logits = self.lm_head(self.norm_f(x))

        out = ModelOutput(logits=logits)
        bal = ent = z = None
        if infos:
            bal = torch.stack([i.bal for i in infos]).mean()
            ent = torch.stack([i.ent for i in infos]).mean()
            z = torch.stack([i.z for i in infos]).mean()
            out.bal_loss, out.ent_loss, out.z_loss = bal.detach(), ent.detach(), z.detach()
            out.routing = RoutingStats(
                avg_experts=torch.stack([i.avg_experts for i in infos]),
                load=torch.stack([i.load for i in infos]),
                count_hist=torch.stack([i.count_hist for i in infos]),
                entropy=torch.stack([i.ent.detach() for i in infos]),
                max_prob=torch.stack([i.max_prob for i in infos]),
                masks=[i.mask for i in infos] if return_routing else None,
                probs=[i.probs for i in infos] if return_routing else None,
            )
        if targets is not None:
            lm = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), targets.reshape(-1))
            total = lm
            if infos:
                if cfg.bal_coef != 0:
                    total = total + cfg.bal_coef * bal
                if cfg.ent_coef != 0:
                    total = total + cfg.ent_coef * ent
                if cfg.z_coef != 0:
                    total = total + cfg.z_coef * z
            out.loss, out.lm_loss = total, lm.detach()
        return out

    # ---- routing control (used for the inference-time p sweep) -------------
    def set_routing(self, route_mode=None, k=None, p=None, min_k=None, max_k=None, gate_norm=None):
        """Change routing in place (all layers share cfg). Validates by rebuilding the config."""
        new = {n: v for n, v in dict(route_mode=route_mode, k=k, p=p, min_k=min_k, max_k=max_k,
                                     gate_norm=gate_norm).items() if v is not None}
        checked = replace(self.cfg, **new)  # re-runs validation
        for key in new:
            setattr(self.cfg, key, getattr(checked, key))

    def num_params(self) -> int:
        return sum(prm.numel() for prm in self.parameters())

    def param_counts(self) -> Dict[str, int]:
        return count_params(self.cfg)

    def flops_per_token(self, avg_experts=None, seq_len=None, attn_context="causal", executed=False):
        return estimate_flops(self.cfg, avg_experts, seq_len, attn_context, executed)

    def summary(self) -> str:
        cfg = self.cfg
        pc = count_params(cfg)
        lines = [
            f"MoETransformer  arch={cfg.arch}  layers={cfg.n_layer}  d={cfg.d_model}  heads={cfg.n_head}  "
            f"seq={cfg.seq_len}  vocab={cfg.vocab_size}",
            f"params: total {pc['total']:,}   non-embedding {pc['non_embedding']:,}",
        ]
        if cfg.arch == "moe":
            lines.append(
                f"experts: {cfg.n_experts} x SwiGLU(h={cfg.expert_hidden})  shared={cfg.n_shared}  "
                f"route={cfg.route_mode}  gate_norm={cfg.gate_norm}  impl={cfg.impl}"
            )
            ks = [cfg.k] if cfg.route_mode == "topk" else sorted({cfg.min_k, cfg.max_k})
            for kk in ks:
                ap = active_params(cfg, kk)
                fl = estimate_flops(cfg, kk)
                lines.append(
                    f"  at {kk} experts/token: active params {ap['total']:,} "
                    f"(non-emb {ap['non_embedding']:,}), forward FLOPs/token {fl['forward']:,.0f}, "
                    f"train FLOPs/token {fl['train']:,.0f}"
                )
            ex = estimate_flops(cfg, cfg.k if cfg.route_mode == "topk" else cfg.max_k, executed=True)
            lines.append(f"  executed forward FLOPs/token by impl={cfg.impl}: {ex['forward']:,.0f}")
        else:
            fl = estimate_flops(cfg)
            lines.append(f"dense FFN h={cfg.dense_hidden}: forward FLOPs/token {fl['forward']:,.0f}, "
                         f"train FLOPs/token {fl['train']:,.0f}")
        return "\n".join(lines)


# Parameter and FLOP accounting
def _per_layer_avg(cfg: MoEConfig, avg_experts: Union[None, float, Sequence[float]]) -> List[float]:
    if avg_experts is None:
        if cfg.route_mode == "topp":
            raise ValueError("top-p routing has a data-dependent expert count: pass the measured avg_experts")
        return [float(cfg.k)] * cfg.n_layer
    if isinstance(avg_experts, (int, float)):
        return [float(avg_experts)] * cfg.n_layer
    vals = [float(v) for v in avg_experts]
    assert len(vals) == cfg.n_layer, f"need one value per MoE layer ({cfg.n_layer})"
    return vals


def count_params(cfg: MoEConfig) -> Dict[str, int]:
    d, hd, V, L = cfg.d_model, cfg.head_dim, cfg.vocab_size, cfg.n_layer
    embed = V * d
    attn = L * (3 * d * d + d * d + (2 * hd if cfg.qk_norm else 0))
    norms = L * 2 * d + d
    out = {"embedding": embed, "attention": attn, "norms": norms}
    if cfg.arch == "moe":
        out["router"] = L * d * cfg.n_experts
        out["routed_experts"] = L * cfg.n_experts * 3 * d * cfg.expert_hidden
        out["shared_experts"] = L * 3 * d * cfg.n_shared * cfg.shared_hidden
    else:
        out["dense_ffn"] = L * 3 * d * cfg.dense_hidden
    out["lm_head"] = 0 if cfg.tie_embeddings else V * d
    out["total"] = sum(out.values())
    out["non_embedding"] = out["total"] - embed - out["lm_head"]
    return out


def active_params(cfg: MoEConfig, avg_experts=None) -> Dict[str, int]:
    """Parameters touched per token when `avg_experts` routed experts are used per layer."""
    pc = count_params(cfg)
    total = pc["total"]
    if cfg.arch == "moe":
        per_expert = 3 * cfg.d_model * cfg.expert_hidden
        active_routed = sum(_per_layer_avg(cfg, avg_experts)) * per_expert
        total = total - pc["routed_experts"] + round(active_routed)
    return {"total": int(total), "non_embedding": int(total - pc["embedding"] - pc["lm_head"])}


def estimate_flops(
    cfg: MoEConfig,
    avg_experts: Union[None, float, Sequence[float]] = None,
    seq_len: Optional[int] = None,
    attn_context: str = "causal",
    executed: bool = False,
) -> Dict[str, float]:
    """Analytic FLOPs per token for one forward pass, and ~3x that for training.

    Convention: one multiply-add = 2 FLOPs. Only matrix multiplies are counted (projections,
    attention score and value products, router, experts, LM head); softmax, norms, RoPE, SiLU and
    gate multiplications are ignored, as in the usual 6N-style accounting.

    avg_experts: routed experts per token, a scalar or one value per layer (for top-p;
        defaults to k for top-k). executed=False gives "effective" FLOPs, what a sparse
        implementation needs. executed=True gives what this implementation runs: impl="dense"
        computes all experts on every token.
    attn_context: "causal" counts the (T+1)/2 positions each token actually attends to on average;
        "full" counts T (the PaLM / Megatron convention, and what an explicit masked matmul runs).
    """
    assert attn_context in ("causal", "full")
    d, V, L = cfg.d_model, cfg.vocab_size, cfg.n_layer
    t = seq_len or cfg.seq_len
    ctx = (t + 1) / 2 if attn_context == "causal" else t
    out = {
        "attn_proj": L * 8 * d * d,  # qkv (6 d^2) + output (2 d^2)
        "attn_scores": L * 4 * d * ctx,  # q.k^T and attn.v, 2*ctx*d each
        "lm_head": 2 * d * V,
    }
    if cfg.arch == "moe":
        ae = _per_layer_avg(cfg, avg_experts)
        n_run = [float(cfg.n_experts)] * L if (executed and cfg.impl == "dense") else ae
        out["router"] = L * 2 * d * cfg.n_experts
        out["routed_experts"] = sum(n_run) * 6 * d * cfg.expert_hidden
        out["shared_experts"] = L * 6 * d * cfg.n_shared * cfg.shared_hidden
    else:
        out["dense_ffn"] = L * 6 * d * cfg.dense_hidden
    out["forward"] = float(sum(out.values()))
    out["train"] = 3.0 * out["forward"]  # backward is about 2x forward
    return out


# Routing statistics aggregation (for eval loops and logging)

class RoutingAccumulator:
    """Token-weighted aggregation of RoutingStats over many batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n_tokens = 0
        self._sum: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, r: RoutingStats, n_tokens: int):
        avg = r.avg_experts.detach().float().cpu()
        parts = {
            "avg_experts": avg * n_tokens,
            "assign": r.load.detach().float().cpu() * (avg * n_tokens)[:, None],  # assignment counts
            "count_hist": r.count_hist.detach().float().cpu() * n_tokens,
            "entropy": r.entropy.detach().float().cpu() * n_tokens,
            "max_prob": r.max_prob.detach().float().cpu() * n_tokens,
        }
        for key, val in parts.items():
            self._sum[key] = self._sum[key] + val if key in self._sum else val
        self.n_tokens += n_tokens

    def compute(self) -> Dict[str, torch.Tensor]:
        n = max(self.n_tokens, 1)
        assign = self._sum["assign"]
        return {
            "avg_experts": self._sum["avg_experts"] / n,  # [L]
            "load": assign / assign.sum(-1, keepdim=True).clamp_min(1.0),  # [L, E]
            "count_hist": self._sum["count_hist"] / n,  # [L, E+1]
            "entropy": self._sum["entropy"] / n,  # [L]
            "max_prob": self._sum["max_prob"] / n,  # [L]
            "n_tokens": torch.tensor(float(self.n_tokens)),
        }


def summarize_routing(res: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Flatten an accumulator result into scalars for CSV or JSON logging. Keep the raw
    result as well if you want the full load vectors and count histograms."""
    out: Dict[str, float] = {}
    n_layers, n_exp = res["load"].shape
    for i in range(n_layers):
        load = res["load"][i]
        out[f"avg_experts/l{i}"] = float(res["avg_experts"][i])
        out[f"router_entropy/l{i}"] = float(res["entropy"][i])
        out[f"max_prob/l{i}"] = float(res["max_prob"][i])
        out[f"load_max_over_mean/l{i}"] = float(load.max() * n_exp)
        out[f"dead_experts/l{i}"] = float((load < 0.01 / n_exp).sum())
    out["avg_experts/mean"] = float(res["avg_experts"].mean())
    out["router_entropy/mean"] = float(res["entropy"].mean())
    out["load_max_over_mean/max"] = max(out[f"load_max_over_mean/l{i}"] for i in range(n_layers))
    return out


if __name__ == "__main__":
    for name, c in [
        ("top-2", MoEConfig(route_mode="topk", k=2)),
        ("top-p 0.5", MoEConfig(route_mode="topp", p=0.5)),
        ("dense baseline", MoEConfig(arch="dense")),
    ]:
        print(f"--- {name}")
        print(MoETransformer(c).summary())