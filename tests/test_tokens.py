#!/usr/bin/env python3
"""Tests for analysis/analyze_tokens.py using planted structure (the true answer is known). python tests/test_tokens.py"""
import json, shutil, sys, tempfile, traceback
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import analyze_tokens as A

TMP = Path(tempfile.mkdtemp(prefix="test_tokens_"))
V, T, W, L, E = 60, 32, 300, 3, 8
rng = np.random.default_rng(0)
zipf = 1.0 / np.arange(1, V + 1); zipf /= zipf.sum()
FREQ = np.bincount(rng.choice(V, size=200_000, p=zipf), minlength=V).astype(float)


def write_dump(run, seed, tokens, loss, counts):
    d = TMP / "runs" / run / f"seed{seed}"; d.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((L, len(tokens), E), dtype=bool)
    for l in range(L):
        for k in range(1, E + 1):
            mask[l, counts[l] >= k, k - 1] = True
    np.savez_compressed(d / "routing_dump.npz", tokens=tokens.astype(np.uint16), token_loss=loss.astype(np.float32),
                        expert_mask=mask, seq_len=np.int64(T), n_experts=np.int64(E))


def load(ids):
    return A.load_runs(TMP / "runs", ids)


def tokens_and_nextrarity():
    tokens = rng.choice(V, size=W * T, p=zipf)
    nxt = -np.log(FREQ[np.roll(tokens, -1)] + 1)
    return tokens, nxt


def planted(kind, name="P", seeds=(0, 1)):
    """kind: 'loss' (experts follow the model's own loss), 'null' (independent), 'rarity' (experts follow next-token rarity only)."""
    tokens, nxt = tokens_and_nextrarity()
    z = (nxt - nxt.mean()) / nxt.std()
    base = 2.0 + 0.4 * z + rng.normal(0, 0.5, len(tokens))   # K2 loss: partly rarity, partly noise
    for s in seeds:
        gain = np.where(base > np.quantile(base, 2 / 3), 0.3, 0.0)  # top-p is better only on K2's hardest third
        loss_p = base - gain + rng.normal(0, 0.02, len(tokens))
        if kind == "loss":
            score = loss_p
        elif kind == "rarity":
            score = nxt + rng.normal(0, 0.05, len(tokens))
        else:
            score = rng.normal(size=len(tokens))
        r = A.rankdata(score) / len(score)
        counts = np.stack([1 + np.minimum(np.floor(r * 4), 3) for _ in range(L)])  # 1..4 experts, ties common
        write_dump(name, s, tokens, loss_p, counts)
    write_dump("K2", 0, tokens, base, np.full((L, len(tokens)), 2.0))
    return tokens


def run_analysis(name="P"):
    runs = load([name, "K2"])
    return A.analyse(runs, [name], "K2", FREQ, [f"t{i}" for i in range(V)], None, 5, TMP)


def fresh():
    shutil.rmtree(TMP / "runs", ignore_errors=True)


# ------------------------------------------------------------------ tests
def test_ranks_and_spearman():
    assert list(A.rankdata([10, 20, 20, 30])) == [1.0, 2.5, 2.5, 4.0]  # ties share the mean rank
    assert abs(A.spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) - 1) < 1e-12
    assert abs(A.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) + 1) < 1e-12
    x = np.array([1, 1, 2, 2, 3, 3]); y = np.array([1, 2, 1, 2, 3, 3])
    ref = np.corrcoef(A.rankdata(x), A.rankdata(y))[0, 1]
    assert abs(A.spearman(x, y) - ref) < 1e-12


def test_partial_correlation_removes_confounder():
    z = rng.normal(size=4000)
    x, y = z + 0.3 * rng.normal(size=4000), z + 0.3 * rng.normal(size=4000)
    assert A.spearman(x, y) > 0.8 and abs(A.partial_spearman(x, y, [z])) < 0.08
    y2 = x + 0.5 * rng.normal(size=4000)  # a real relation survives controlling for an unrelated variable
    assert A.partial_spearman(x, y2, [rng.normal(size=4000)]) > 0.7


def test_bootstrap_interval():
    s, n = np.full(200, 0.5), np.ones(200)
    m, lo, hi = A.bootstrap_mean_ci(s, n)
    assert abs(m - 0.5) < 1e-12 and lo <= m <= hi
    noisy = rng.normal(0.1, 1.0, 200)
    m, lo, hi = A.bootstrap_mean_ci(noisy, np.ones(200))
    assert lo < m < hi and hi - lo > 0.1  # a real interval, not a point


def test_planted_loss_dependence_is_found():
    fresh(); planted("loss")
    res, fig = run_analysis()
    assert min(res["P"]["rho_loss"]) > 0.6, res
    assert min(res["P"]["partial"]) > 0.5, res  # still strong after controlling for rarity


def test_null_gives_no_correlation():
    fresh(); planted("null")
    res, _ = run_analysis()
    assert max(abs(v) for v in res["P"]["rho_loss"]) < 0.06 and max(abs(v) for v in res["P"]["partial"]) < 0.06, res


def test_rarity_confound_is_not_mistaken_for_difficulty():
    fresh(); planted("rarity")  # experts follow next-token rarity; loss also follows rarity
    res, _ = run_analysis()
    assert min(res["P"]["rho_loss"]) > 0.15, res         # a raw correlation appears...
    assert max(abs(v) for v in res["P"]["partial"]) < 0.08, res  # ...and disappears after controlling for rarity


def test_gain_by_independent_difficulty_group():
    fresh(); planted("loss")
    _, fig = run_analysis()
    g = fig["P"]["gain"]
    hardest = g[("K2 difficulty (loss tercile)", "hardest 1/3")]
    easiest = g[("K2 difficulty (loss tercile)", "easiest 1/3")]
    assert abs(hardest[0] - 0.3) < 0.03 and hardest[1] > 0.25, hardest      # planted gain found, interval excludes 0
    assert easiest[1] <= 0.0 <= easiest[2] and abs(easiest[0]) < 0.01, easiest  # no gain where none was planted


def test_last_window_position_is_excluded():
    """The next token of the last position in a window is not in the dump, so those positions must not enter the grouped gains."""
    fresh()
    tokens, nxt = tokens_and_nextrarity()
    base = 2.0 + rng.normal(0, 0.3, len(tokens))
    last = (np.arange(len(tokens)) % T) == T - 1
    loss_p = base - np.where(last, 5.0, 0.0)  # a huge gain exists ONLY at last positions
    write_dump("P", 0, tokens, loss_p, np.full((L, len(tokens)), 2.0))
    write_dump("K2", 0, tokens, base, np.full((L, len(tokens)), 2.0))
    _, fig = run_analysis()
    for key, (m, lo, hi) in fig["P"]["gain"].items():
        assert abs(m) < 0.01, (key, m)


def test_dump_roundtrip_and_counts():
    fresh(); tokens = planted("null")
    r = load(["P"])["P"][0]
    assert (r["tokens"] == tokens).all() and r["counts"].shape == (L, len(tokens)) and r["counts"].min() >= 1 and r["counts"].max() <= 4


def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in tests:
        try:
            f(); print("PASS", n)
        except Exception:  # noqa: BLE001
            bad += 1; print("FAIL", n); traceback.print_exc()
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(tests) - bad}/{len(tests)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())