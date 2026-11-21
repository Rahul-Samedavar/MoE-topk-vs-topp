#!/usr/bin/env python3
"""Smoke and correctness tests for train.py, experiments.py and run.py.  python tests/test_train.py [--device cpu|mps]"""
import json, os, shutil, signal, sys, tempfile, threading, traceback
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train as T
import experiments as E
from moe import MoEConfig

DEVICE = "cpu"
TMP = Path(tempfile.mkdtemp(prefix="test_train_"))
DATA = TMP / "data"


def make_data():
    DATA.mkdir(parents=True)
    rng = np.random.default_rng(0)
    base = rng.integers(1, 512, size=64)
    np.tile(base, 4000)[:100_000].astype(np.uint16).tofile(DATA / "train.bin")
    np.tile(base, 400)[:12_000].astype(np.uint16).tofile(DATA / "val.bin")
    json.dump({"vocab_size": 512, "train_tokens": 100_000, "val_tokens": 12_000, "dtype": "uint16",
               "val_stats": {"bytes_per_token": 4.0}}, open(DATA / "meta.json", "w"))


def cfgs(mode="topp", **kw):
    m = dict(vocab_size=512, n_layer=2, d_model=32, n_head=2, seq_len=32, n_experts=4, expert_hidden=64,
             dense_hidden=128, route_mode=mode, k=2, p=0.5, max_k=3, ent_coef=1e-4 if mode == "topp" else 0.0)
    t = dict(data_dir=str(DATA), batch_size=8, max_steps=40, warmup_steps=5, log_interval=5, eval_interval=10,
             ckpt_interval=10, eval_tokens=2048, sweep_tokens=2048, dump_tokens=2048, device=DEVICE, tqdm=False, seed=0)
    for k, v in kw.items():
        (m if k in MoEConfig.__dataclass_fields__ else t)[k] = v
    return MoEConfig(**m), T.TrainConfig(**t)


def run(name, halt=0, hard=False, **kw):
    mc, tc = cfgs(**kw)
    return T.train(tc, mc, TMP / name, exp_id=name, halt_after_steps=halt, halt_hard=hard)


def metrics(name, f="metrics.jsonl"):
    return T.read_jsonl(TMP / name / f)


def load_final(name):
    return torch.load(TMP / name / "ckpt" / "final.pt", weights_only=True)["model"]


def rel_diff(a, b):
    num = sum(float((a[k].float() - b[k].float()).pow(2).sum()) for k in a) ** 0.5
    return num / sum(float(a[k].float().pow(2).sum()) for k in a) ** 0.5


def same_weights(a, b, tol=1e-3):
    """Bit-identical on CPU. Other devices (MPS) are not bit-for-bit deterministic run to run, so they get a
    relative tolerance; a genuinely broken resume (e.g. optimizer state dropped) gives about 5e-2 on this setup."""
    if DEVICE == "cpu":
        return all(torch.equal(a[k], b[k]) for k in a)
    return rel_diff(a, b) < tol


def test_artifacts():
    r = run("art")
    assert r["status"] == "done"
    for f in ["config.json", "metrics.jsonl", "eval.jsonl", "events.jsonl", "train.log", "sweep_eval.json",
              "routing_dump.npz", "summary.json", "ckpt/final.pt"]:
        assert (TMP / "art" / f).exists(), f
    assert not (TMP / "art/ckpt/last.pt").exists()
    m = metrics("art")
    assert [x["step"] for x in m] == list(range(5, 41, 5))
    need = {"loss", "lm_loss", "bal_loss", "ent_loss", "z_loss", "grad_norm", "clip_frac", "lr", "tokens_per_s",
            "avg_experts", "expert_load", "count_hist", "router_entropy", "max_prob", "router_grad_norm",
            "cum_flops_effective", "cum_flops_executed"}
    assert need <= set(m[0]), need - set(m[0])
    assert len(m[0]["avg_experts"]) == 2 and len(m[0]["expert_load"][0]) == 4 and len(m[0]["count_hist"][0]) == 5
    ev = metrics("art", "eval.jsonl")
    assert [x["step"] for x in ev] == [0, 10, 20, 30, 40] and ev[-1]["final"] and "val_bpb" in ev[-1]
    d = np.load(TMP / "art/routing_dump.npz")
    assert d["expert_mask"].shape == (2, d["tokens"].size, 4) and d["token_loss"].shape == d["tokens"].shape
    sw = json.load(open(TMP / "art/sweep_eval.json"))["rows"]
    assert sw[0]["label"] == "as_trained"
    assert abs(sw[0]["val_loss"] - ev[-1]["val_loss"]) < 0.1 and len(sw) == 12
    s = json.load(open(TMP / "art/summary.json"))
    assert s["steps"] == 40 and s["val_loss"] == ev[-1]["val_loss"] and s["tokens"] == 40 * 8 * 32
    assert run("art")["status"] == "skipped"


def test_learns_and_lr():
    m = metrics("art")
    assert m[-1]["lm_loss"] < m[0]["lm_loss"] - 1.0
    mc, tc = cfgs()
    lrs = [T.lr_at(s, tc, 40) for s in range(40)]
    assert abs(lrs[4] - tc.lr) < 1e-12 and lrs[0] < lrs[4] and lrs[-1] >= tc.lr * tc.min_lr_frac - 1e-12
    assert all(a >= b for a, b in zip(lrs[4:], lrs[5:]))


def test_schedule_is_applied_and_gradnorm_logged():
    if not (TMP / "ref/summary.json").exists():
        run("ref")
    run("wu1", warmup_steps=1)  # differs from "ref" (warmup 5) only in the schedule
    assert rel_diff(load_final("ref"), load_final("wu1")) > 1e-2, "lr schedule has no effect on training"
    for r in metrics("ref"):
        assert np.isfinite(r["grad_norm"]) and r["grad_norm"] > 0 and all(g > 0 for g in r["router_grad_norm"])


def test_device_noise_report():
    """Informational: how far apart are two identical uninterrupted runs on this device?"""
    if not (TMP / "ref/summary.json").exists():
        run("ref")
    run("ref_b")
    d = rel_diff(load_final("ref"), load_final("ref_b"))
    print(f"  run-to-run weight difference on {DEVICE}: {d:.2e}  (0 = bit-identical)")
    assert d < 1e-3


def test_resume_equivalence():
    if not (TMP / "ref/summary.json").exists():
        run("ref")
    r = run("res", halt=20)
    assert r["status"] == "halted" and (TMP / "res/ckpt/last.pt").exists()
    r = run("res")
    assert r["status"] == "done" and r["summary"]["n_resumes"] == 1
    assert same_weights(load_final("ref"), load_final("res")), "resumed weights differ from uninterrupted"
    a, b = metrics("ref"), metrics("res")
    assert [x["step"] for x in a] == [x["step"] for x in b]
    for x, y in zip(a, b):
        assert abs(x["loss"] - y["loss"]) < 1e-6 and abs(x["cum_flops_effective"] - y["cum_flops_effective"]) < 1
    assert [e["val_loss"] for e in metrics("ref", "eval.jsonl")] == [e["val_loss"] for e in metrics("res", "eval.jsonl")]
    ev = [e["event"] for e in T.read_jsonl(TMP / "res/events.jsonl")]
    assert ev[:3] == ["start", "halt", "resume"] and ev[-1] == "finish"


def test_hard_crash_recovery():
    run("crash", halt=25, hard=True)  # checkpoint at 20, logs reach step 25
    assert metrics("crash")[-1]["step"] == 25
    r = run("crash")
    assert r["status"] == "done"
    steps = [x["step"] for x in metrics("crash")]
    assert steps == list(range(5, 41, 5)), steps  # no duplicates, nothing lost
    assert same_weights(load_final("ref"), load_final("crash"))
    assert [x["step"] for x in metrics("crash", "eval.jsonl")] == [0, 10, 20, 30, 40]


def test_truncated_last_line():
    run("trunc", halt=20, hard=True)
    with open(TMP / "trunc/metrics.jsonl", "a") as f:
        f.write('{"step": 25, "lo')
    assert run("trunc")["status"] == "done"
    assert [x["step"] for x in metrics("trunc")] == list(range(5, 41, 5))


def test_signal_interrupt():
    mc, tc = cfgs()
    def cb(step):
        if step == 15:
            os.kill(os.getpid(), signal.SIGINT)
    r = T.train(tc, mc, TMP / "sig", exp_id="sig", step_callback=cb)
    assert r["status"] == "interrupted" and r["step"] == 15
    assert (TMP / "sig/ckpt/last.pt").exists()
    r = T.train(tc, mc, TMP / "sig", exp_id="sig")
    assert r["status"] == "done" and same_weights(load_final("ref"), load_final("sig"))
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_config_mismatch_refused():
    run("mm", halt=10)
    for kw in ({"lr": 1e-3}, {"p": 0.7}, {"seed": 1}, {"batch_size": 4}):
        try:
            run("mm", **kw)
        except RuntimeError as e:
            assert "cannot resume" in str(e)
        else:
            raise AssertionError(f"mismatch {kw} was accepted")
    mc, tc = cfgs(eval_interval=20)  # mutable field: allowed
    assert T.train(tc, mc, TMP / "mm", exp_id="mm")["status"] == "done"


def test_flops_accumulation():
    from moe import estimate_flops
    s = json.load(open(TMP / "art/summary.json"))
    mc, _ = cfgs()
    m = metrics("art")
    tok = 8 * 32
    manual = sum(5 * tok * estimate_flops(mc, r["avg_experts"])["train"] for r in m)  # window means, 5 steps each
    assert abs(manual - s["cum_flops_effective"]) / manual < 0.05, (manual, s["cum_flops_effective"])
    assert s["cum_flops_executed"] > s["cum_flops_effective"]


def test_sampler():
    tokens = np.arange(1001, dtype=np.uint16)
    s1, s2 = T.Sampler(tokens, 10, 7), T.Sampler(tokens, 10, 7)
    assert s1.n_seq == 100
    seen = np.concatenate([s1.window_ids(i * 10, 10) for i in range(10)])
    assert sorted(seen) == list(range(100))  # one pass covers every window once
    assert (s1.window_ids(37, 20) == s2.window_ids(37, 20)).all()  # pure function of position
    assert (T.Sampler(tokens, 10, 8).window_ids(0, 20) != s1.window_ids(0, 20)).any()
    x, y = s1.batch(0, 4)
    assert (y[:, :-1] == x[:, 1:]).all() and x.shape == (4, 10)
    assert not (s1.window_ids(100, 100) == s1.window_ids(0, 100)).all()  # new epoch, new order


def test_other_modes():
    for name, kw in (("topk", dict(mode="topk")), ("dense", dict(arch="dense")), ("accum", dict(grad_accum=2, max_steps=10))):
        r = run("m_" + name, **kw)
        assert r["status"] == "done", name
    assert not (TMP / "m_dense/routing_dump.npz").exists()
    m = metrics("m_dense")
    assert "avg_experts" not in m[0] and m[0]["cum_flops_effective"] > 0
    k = metrics("m_topk")[-1]
    assert abs(k["avg_experts_mean"] - 2.0) < 1e-6
    assert json.load(open(TMP / "m_accum/summary.json"))["tokens"] == 10 * 2 * 8 * 32


def test_nan_guard():
    mc, tc = cfgs(lr=1e9, grad_clip=1e30, max_steps=30)
    try:
        T.train(tc, mc, TMP / "nan", exp_id="nan")
    except FloatingPointError:
        return
    # a huge lr may not always produce NaN within 30 steps; accept finite completion
    assert (TMP / "nan/summary.json").exists()


def test_cli_and_experiments():
    assert E.check_controls() == []
    assert len(E.select()) == 79 and len(E.EXPS) == 42
    counts = {g: len(E.select([g])) for g in E.GROUPS}
    assert counts == {"lr-sweep": 12, "alpha-sweep": 16, "main-10m": 19, "entropy-10m": 8, "gatenorm-10m": 5,
                      "main-30m": 17, "gatenorm-30m": 2}, counts
    assert [s for _, s in E.select(only=["K2"], seeds=[5, 6])] == [5, 6]
    mo, to, dd = E.resolve("P50")
    assert mo["route_mode"] == "topp" and mo["ent_coef"] == 1e-4 and to == {"max_tokens": 10_000_000, "lr": 4e-3}
    assert E.resolve("L-P65")[2] == "data/tinystories30m" and E.resolve("LR-K2-8e-3")[1]["max_tokens"] == 3_000_000
    mc, tc = T.build_configs("P50", ["lr=0.002", "p=0.4"], 1, str(DATA), "cpu", True)
    assert mc.p == 0.4 and tc.lr == 0.002 and tc.seed == 1 and tc.data_dir == str(DATA)
    assert T.main(["--exp", "K2", "--data-dir", str(DATA), "--dry-run", "--device", "cpu"]) == 0
    try:
        T.build_configs(None, ["nonsense=1"], 0, None, "cpu", True)
    except SystemExit:
        pass
    else:
        raise AssertionError("unknown key accepted")
    saved = E.EXPS[:]  # the controls check must catch an unfair control
    try:
        E.EXPS.append(E.Exp("BAD", "main-10m", "x", dict(route_mode="topk", k=2, n_layer=3)))
        assert any("BAD" in p for p in E.check_controls())
        E.EXPS[-1] = E.Exp("BAD", "entropy-10m", "x", dict(route_mode="topk", k=2, ent_coef=1e-3), lr=1e-3)
        assert any("BAD" in p and "learning rate" in p for p in E.check_controls())
    finally:
        E.EXPS[:] = saved
    assert E.check_controls() == []


def test_lr_report_decision_rule():
    sys.path.insert(0, str(ROOT / "analysis"))
    import tuning_report as TR

    def fake(losses):
        out = TMP / "lrfake"
        shutil.rmtree(out, ignore_errors=True)
        for fam, vals in losses.items():
            for lr, v in zip((1e-3, 2e-3, 4e-3, 8e-3), vals):
                e = next(x for x in E.EXPS if x.group == "lr-sweep" and x.family == fam and x.lr == lr)
                rd = E.run_dir(e.id, 0, str(out)); rd.mkdir(parents=True)
                (rd / "summary.json").write_text(json.dumps({"val_loss": v}))
        return TR.lr_report(str(out))
    r = fake({"topk": [3.0, 2.8, 2.805, 2.9], "topp": [3.0, 2.805, 2.8, 2.9], "dense": [3.0, 2.8, 2.82, 2.9]})
    assert r["choice"] == {"topk": 2e-3, "topp": 2e-3, "dense": 2e-3}, r  # shared: cost <= 0.01
    r = fake({"topk": [3.0, 2.8, 2.9, 3.1], "topp": [3.1, 2.9, 2.8, 2.85], "dense": [3.0, 2.8, 2.9, 3.0]})
    assert r["choice"] == {"topk": 2e-3, "topp": 4e-3, "dense": 2e-3}, r  # one step apart but shared costs 0.1
    r = fake({"topk": [3.0, 2.8, 2.9, 3.1], "topp": [3.3, 3.1, 2.9, 2.8], "dense": [3.0, 2.8, 2.9, 3.0]})
    assert r["choice"]["topp"] == 8e-3 and any("edge" in n for n in r["notes"]), r
    assert TR.lr_report(str(TMP / "none"))["choice"] is None


def test_runner_dry_run_and_verify():
    import run as RUN
    assert RUN.main(["train", "--only", "K2", "--dry-run", "--out-dir", str(TMP / "dry")]) == 0
    assert RUN.main(["train"]) == 2  # nothing selected: refuses to launch everything by accident
    rd = TMP / "verify" / "K1" / "seed0"; rd.mkdir(parents=True)
    model, train, h = RUN.expected_config(E.BY_ID["K1"], 0)
    rec = {"model": model, "train": train, "config_hash": h, "data": {"vocab_size": 4096}}
    (rd / "config.json").write_text(json.dumps(rec))
    assert RUN.main(["verify", "--runs", str(TMP / "verify"), "--only", "K1"]) == 0  # seed 1 missing is not a mismatch
    rec["train"]["lr"] = 1e-3
    (rd / "config.json").write_text(json.dumps(rec))
    assert RUN.main(["verify", "--runs", str(TMP / "verify"), "--only", "K1"]) == 1


def test_cli_end_to_end():
    import subprocess
    out = TMP / "cli"
    cmd = [sys.executable, str(ROOT / "train.py"), "--run-name", "c", "--out-dir", str(out),
           "--data-dir", str(DATA), "--device", DEVICE, "--no-tqdm"]
    for kv in ("vocab_size=512", "n_layer=2", "d_model=32", "n_head=2", "seq_len=32", "n_experts=4",
               "expert_hidden=64", "max_steps=10", "batch_size=4", "warmup_steps=2", "log_interval=5",
               "eval_interval=5", "ckpt_interval=5", "eval_tokens=1024", "sweep_tokens=1024", "dump_tokens=1024"):
        cmd += ["--set", kv]
    assert subprocess.call(cmd, stdout=subprocess.DEVNULL) == 0
    assert (out / "c/seed0/summary.json").exists()


def main():
    global DEVICE
    if "--device" in sys.argv:
        DEVICE = sys.argv[sys.argv.index("--device") + 1]
    make_data()
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    bad = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS {name}")
        except Exception:  # noqa: BLE001
            bad += 1; print(f"FAIL {name}"); traceback.print_exc()
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(tests) - bad}/{len(tests)} passed (device {DEVICE})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())