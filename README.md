# Top-k vs top-p expert routing in a small MoE language model

> **Research paper:** [Read the paper (PDF)](paper/top-k-vs-top-p-moe.pdf)
>
> The manuscript is included in this repository at `paper/top-k-vs-top-p-moe.pdf`. Start here for the research paper; the rest of this README documents the code and steps needed to reproduce the experiments.

Code to train and analyse the models in the paper: a 4-layer mixture-of-experts transformer (8 SwiGLU experts per
layer) trained on TinyStories with top-k or top-p expert routing, plus a dense baseline.

## Setup

Python 3.11. The results were produced with the exact versions in `requirements.txt` on an Apple-silicon Mac
(PyTorch MPS backend, fp32). CUDA and CPU also work.

```bash
git clone https://github.com/Rahul-Samedavar/MoE-topk-vs-topp.git && cd MoE-topk-vs-topp
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run every command from the repository root.

## Data

```bash
python prepare_data.py
```

This downloads TinyStories (`roneneldan/TinyStories` at a pinned revision, about 2 GB), then writes:

| Folder                   | Training tokens | Used by                    |
| ------------------------ | --------------- | -------------------------- |
| `data/tinystories/`    | 10M             | the 3M- and 10M-token runs |
| `data/tinystories30m/` | 30M             | the 30M-token runs         |

Both use the shipped tokenizer (`tokenizer/tinystories_bpe_4096/tokenizer.json`, a 4,096-token byte-level BPE) and
share one validation file. The script ends by checking SHA-256 checksums against the files our runs were trained on,
and prints `all files match the reference checksums` if they are identical. `--retrain-tokenizer` retrains the
tokenizer from the downloaded data instead of using the shipped file (with the pinned versions this gives the
identical file).

## Running experiments

Every run is registered in `experiments.py` with its exact configuration and seeds. A *configuration* has an ID
(e.g. `K2`, `P60`, `L-P65`); each configuration is trained with one or more seeds, and each (ID, seed) pair is one
*run*. Configurations are organised in groups:

| Group            | Budget     | Configurations (seeds)                                                                 |
| ---------------- | ---------- | -------------------------------------------------------------------------------------- |
| `lr-sweep`     | 3M tokens  | `LR-K2-*`, `LR-P50-*`, `LR-DENSE-*` for peak LR 1e-3, 2e-3, 4e-3, 8e-3 (seed 0)  |
| `alpha-sweep`  | 3M tokens  | `AL-K2-*`, `AL-P50-*` for load-balance weight 1e-3, 1e-2, 1e-1, 3e-1 (seeds 0-1)   |
| `main-10m`     | 10M tokens | K1 (0-1), K2 (0-2), K3 (0-1), P30 (0-1), P50 (0-1), P60 (0-2), P70 (0-1), DENSE (0-2)  |
| `entropy-10m`  | 10M tokens | K2-ent, P50-noent, P50-ent1e-3, P50-ent1e-2 (0-1 each)                                 |
| `gatenorm-10m` | 10M tokens | K2-renorm (0-2), P50-renorm (0-1)                                                      |
| `main-30m`     | 30M tokens | L-K1 (0-1), L-K2 (0-2), L-K3 (0-1), L-P60 (0-2), L-P65 (0-2), L-P70 (0), L-DENSE (0-2) |
| `gatenorm-30m` | 30M tokens | L-K2-renorm (0-1)                                                                      |

Naming: `K<k>` is top-k routing; `P<nn>` is top-p routing with p = 0.nn (at most 4 experts per token, router entropy
loss weight 1e-4); `DENSE` replaces each MoE layer with one FFN with the same active parameters as top-2; `L-` marks
the 30M-token runs. `python run.py list` prints every configuration with a description and its status.

### The master runner

```bash
python run.py list                                   # all configurations, seeds, status, time estimate
python run.py train --all                            # every registered run (79 runs)
python run.py train --group main-10m                 # one or more groups
python run.py train --only K2 P60                    # specific configurations, with their registered seeds
python run.py train --only K2 --seeds 0              # specific seeds
python run.py train --group main-10m --seeds 3 4     # new seeds
python run.py train --group main-30m --dry-run       # print the commands without running them
python run.py train --only K2 -- --device cpu        # arguments after -- are passed to train.py
```

Without `--seeds`, each configuration is trained with exactly the seeds we used. A seed fixes both the weight
initialization and the order of training data, so two configurations with the same seed see the same batches.

Each run is written to `runs/<ID>/seed<N>/`. Runs that already have a `summary.json` are skipped, and interrupted
runs resume from their last checkpoint (saved every 200 steps), so after Ctrl-C, a crash or a reboot, run the same
command again. Runs execute one at a time; to use several devices, start several `run.py` commands with disjoint
selections.

Rough wall-clock times on an Apple-silicon laptop, one run at a time: 3M-token run 5-7 min, 10M-token MoE run about
20 min (dense about 10), 30M-token MoE run about 1 h. All 79 runs take roughly 33 h.

### Single runs and your own configurations

`run.py` calls `train.py` once per run. You can call it directly:

```bash
python train.py --exp K2 --seed 0                                    # one registered run
python train.py --exp K2 --seed 0 --dry-run                          # print the resolved configuration
python train.py --exp P50 --set p=0.4 --run-name P40 --seed 0        # a variant of a registered configuration
python train.py --set route_mode=topp --set p=0.4 --set ent_coef=1e-4 --set lr=4e-3 --run-name mine --seed 0
```

`--set KEY=VALUE` overrides any field of `MoEConfig` (`moe.py`: model, routing, loss weights) or `TrainConfig`
(`train.py`: data, budget, optimizer, evaluation). `--run-name` sets the output folder name, `--out-dir` the runs
folder (default `runs`), `--device` one of `auto`, `mps`, `cuda`, `cpu`. To add a configuration to the runner, add an
entry to `EXPS` in `experiments.py`; `python run.py list` checks that each configuration differs from its reference
model only in the settings its group is meant to vary.

### What a run writes

| File                            | Contents                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------- |
| `config.json`                 | resolved model and training config, data info, environment, command line, `config_hash`         |
| `metrics.jsonl`               | training log every 10 steps (losses, gradient norms, per-layer routing statistics)                |
| `eval.jsonl`                  | validation every 100 steps on a fixed 262k-token subset; the final one on the full validation set |
| `sweep_eval.json`             | the trained model re-evaluated with other top-p thresholds and top-k values                       |
| `routing_dump.npz`            | per-token loss and per-layer expert masks on the first 131k validation tokens                     |
| `summary.json`                | final results; written last, so its presence means the run is complete                            |
| `train.log`, `events.jsonl` | console log; start, resume, interrupt and finish events                                           |
| `ckpt/final.pt`               | final weights (`ckpt/last.pt` holds the resume state while a run is in progress)                |

## Reproducing our results

1. `python prepare_data.py` and check that all checksums match.
2. `python run.py train --all` (or group by group).
3. `python analysis/reproduce_analysis.py`

Step 3 runs every analysis script on the finished runs in `runs/` and writes their console output
(`results/<step>.txt`) and figures to `results/`. Use `--only` or `--skip` with step names, and `--runs` to analyse
runs stored elsewhere. Every step except `checkpoint_probe`, which loads `ckpt/final.pt`, needs only the small log
files of each run.

| Step                                | Script                           | Covers                                                                                                                                             |
| ----------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tuning_lr`, `tuning_alpha`     | `analysis/tuning_report.py`    | the two 3M-token sweeps                                                                                                                            |
| `analyze_main`, `analyze_scale` | `analysis/analyze.py`          | results tables, compute-matched gaps, inference-time sweep, health flags, figures, for 10M and 30M runs                                            |
| `summary_stats`                   | `analysis/summary_stats.py`    | paired same-seed differences, seed spreads, learning curves, routing and load, training health                                                     |
| `compute_matched`                 | `analysis/compute_matched.py`  | gaps below the top-k curve at matched inference compute (final experts per token) and matched training compute (total training FLOPs), 10M and 30M |
| `entropy_sweep`                   | `analysis/entropy_sweep.py`    | the `entropy-10m` group                                                                                                                          |
| `tokens_10m`, `tokens_30m`      | `analysis/analyze_tokens.py`   | token-level analysis of expert counts                                                                                                              |
| `probe_layers`                    | `analysis/probe_layers.py`     | per-layer probes: token determinism, early training, final routing                                                                                 |
| `checkpoint_probe`                | `analysis/checkpoint_probe.py` | forward-only probes of the final checkpoints                                                                                                       |
| `figures`                         | `analysis/figures.py`          | summary figures across groups, including loss against experts per token (`fig1`) and against total training FLOPs (`fig6`)                     |

To check that runs were trained with the registered configurations (for example, runs you trained, or ours if you
downloaded them):

```bash
python run.py verify --runs runs
```

It compares each run's `config.json` with the registry, including `config_hash`, a hash of every setting that can
affect results.

### How exact is a rerun?

- **Data and configurations** are identical: the checksums and `run.py verify` check this.
- **CPU** training is deterministic: rerunning a configuration and seed reproduces the weights, logs and outputs bit
  for bit, also across an interrupt and resume.
- **Apple MPS** (what we used) is not deterministic over a full run: rerunning the same configuration and seed changes
  the final validation loss by about 0.003. Early training matches exactly. Expect individual numbers to differ at
  about this level; compare seed means and seed spreads, not single values.
- GPUs and other library versions add their own numerical differences.

## Tests

```bash
python tests/test_moe.py      # routing rules, losses, gradients, FLOP and parameter counts
python tests/test_train.py    # training loop, resume, registry and runner
python tests/test_tokens.py   # token-level statistics on planted data
```

## Repository layout

```
moe.py               model: MoE transformer, top-k / top-p routing, auxiliary losses, FLOP accounting
train.py             trains one run
experiments.py       registry of all configurations, groups and seeds
run.py               master runner: list, train, verify
prepare_data.py      download and tokenize TinyStories
tokenizer/           the BPE tokenizer
analysis/            analysis scripts; reproduce_analysis.py runs them all
paper/               research manuscript (PDF)
tests/               tests
```
