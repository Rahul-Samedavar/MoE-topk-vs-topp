#!/usr/bin/env python3
"""prepare_data.py - download TinyStories and build the token files that train.py reads.

    python prepare_data.py                      # about 2 GB download; writes data/tinystories and data/tinystories30m
    python prepare_data.py --retrain-tokenizer  # also retrain the BPE tokenizer instead of using the shipped one
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DATASET = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
VOCAB_SIZE = 4096
EXPECTED_SHA256 = {
    "tokenizer/tinystories_bpe_4096/tokenizer.json": "4ae86bf5b176b1abc4ba3eb302b835c341d622e4ab37cbdc49f174a4cf6c77aa",
    "tinystories/train.bin": "b7f6cd439c6c617a8647adb2bfd9a0611e3a68e37775ce6b169328a5fd23b2f9",
    "tinystories/val.bin": "1746cf28677a637b0854a77a95880e71a5d87ab88f244f33d8734de959b40c09",
    "tinystories30m/train.bin": "8f2d72f6a868c543d6599896425a025ca7d667f3fbdd5772e57c145c1073f74a",
    "tinystories30m/val.bin": "1746cf28677a637b0854a77a95880e71a5d87ab88f244f33d8734de959b40c09",
}
BUDGETS = {"tinystories": 10_000_000, "tinystories30m": 30_000_000}
MAX_VAL_TOKENS = 1_000_000
SEED = 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dataset_dir: Path):
    if (dataset_dir / "dataset_dict.json").exists():
        print(f"{dataset_dir} exists, skipping download")
        return
    from datasets import load_dataset
    print(f"downloading {DATASET} at revision {REVISION} ...")
    ds = load_dataset(DATASET, revision=REVISION)
    ds.save_to_disk(str(dataset_dir))
    print({k: len(v) for k, v in ds.items()}, "stories")


def train_tokenizer(dataset_dir: Path, path: Path):
    """Byte-level BPE with 4,096 tokens."""
    from datasets import load_from_disk
    from tokenizers import Tokenizer, models, trainers
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.pre_tokenizers import ByteLevel

    dataset = load_from_disk(str(dataset_dir))
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=["<unk>", "<s>", "</s>"],
                                  initial_alphabet=ByteLevel.alphabet(), show_progress=True)

    def batch_iterator(data, batch_size=1000):
        for start in range(0, len(data), batch_size):
            yield from data[start:start + batch_size]["text"]

    print(f"\nTraining {VOCAB_SIZE}-token BPE tokenizer...")
    tokenizer.train_from_iterator(batch_iterator(dataset["train"]), trainer=trainer, length=len(dataset["train"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    print("Vocabulary size:", tokenizer.get_vocab_size(), " saved at:", path)


def encode_split(ds, tok, eos_id, max_tokens, seed, truncate, batch=1000):
    order = np.random.default_rng(seed).permutation(len(ds)) if seed is not None else np.arange(len(ds))
    chunks, total, n_stories, n_bytes = [], 0, 0, 0
    for i in range(0, len(order), batch):
        texts = [t.strip() for t in ds.select(order[i:i + batch].tolist())["text"]]
        texts = [t for t in texts if t]
        for text, enc in zip(texts, tok.encode_batch(texts)):
            chunks.append(np.array(enc.ids + [eos_id], dtype=np.uint16))
            total += len(enc.ids) + 1
            n_bytes += len(text.encode("utf-8"))
            n_stories += 1
        if total >= max_tokens:
            break
    if total < max_tokens:
        print(f"  warning: split has only {total:,} tokens, wanted {max_tokens:,}")
    arr = np.concatenate(chunks)
    if truncate:
        arr = arr[:max_tokens]
    return arr, {"stories": n_stories, "tokens": int(len(arr)), "bytes_per_token": n_bytes / total}


def pre_tokenize(dataset_dir: str, tokenizer_path: str, out: Path, max_train_tokens: int, eos="</s>"):
    """dataset_dir and tokenizer_path are kept as given (e.g. "./data/tinystories"); they are recorded in meta.json."""
    from datasets import load_from_disk
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    eos_id, unk_id = tok.token_to_id(eos), tok.token_to_id("<unk>")
    assert eos_id is not None, f"{eos} not in tokenizer"
    assert tok.get_vocab_size() < 65536
    ds = load_from_disk(str(dataset_dir))
    print(f"tokenizer vocab {tok.get_vocab_size()}, separator {eos}={eos_id}; "
          f"train {len(ds['train']):,} stories, val {len(ds['validation']):,} stories")

    print("encoding train ..."); tr, s_tr = encode_split(ds["train"], tok, eos_id, max_train_tokens, SEED, True)
    print("encoding val ...");   va, s_va = encode_split(ds["validation"], tok, eos_id, MAX_VAL_TOKENS, None, False)
    for name, arr in (("train", tr), ("val", va)):
        assert arr.max() < tok.get_vocab_size()
        if unk_id is not None:
            print(f"  {name}: <unk> tokens = {int((arr == unk_id).sum())}")
    out.mkdir(parents=True, exist_ok=True)
    tr.tofile(out / "train.bin"); va.tofile(out / "val.bin")

    bad = 0  # round trip: decode(encode(text)) == text on a few validation stories
    for t in ds["validation"].select(range(min(200, len(ds["validation"]))))["text"]:
        t = t.strip()
        bad += tok.decode(tok.encode(t).ids) != t
    meta = {"dataset": str(dataset_dir), "vocab_size": tok.get_vocab_size(), "eos_id": eos_id, "dtype": "uint16",
            "train_tokens": int(len(tr)), "val_tokens": int(len(va)), "train_stats": s_tr, "val_stats": s_va,
            "roundtrip_mismatches_of_200": int(bad), "tokenizer": str(tokenizer_path), "seed": SEED}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("vocab_size", "eos_id", "train_tokens", "val_tokens",
                                           "roundtrip_mismatches_of_200")}),
          f"\nval bytes/token {s_va['bytes_per_token']:.3f}\nwrote {out}/train.bin val.bin meta.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="./data", help="output folder (train.py expects ./data)")
    ap.add_argument("--tokenizer", default="./tokenizer/tinystories_bpe_4096/tokenizer.json")
    ap.add_argument("--retrain-tokenizer", action="store_true",
                    help="train the tokenizer again and write it to --tokenizer (overwrites the shipped file)")
    a = ap.parse_args()
    root, tok_path = Path(a.root), Path(a.tokenizer)
    dataset_dir = f"{a.root.rstrip('/')}/tinystories"
    
    download(Path(dataset_dir))
    if a.retrain_tokenizer or not tok_path.exists():
        train_tokenizer(Path(dataset_dir), tok_path)
    for name, budget in BUDGETS.items():
        if (root / name / "meta.json").exists():
            print(f"{root / name}/meta.json exists, skipping")
            continue
        print(f"\n== {name}: {budget:,} training tokens ==")
        pre_tokenize(dataset_dir, a.tokenizer, root / name, budget)

    print("\n== checksums ==")
    ok = True
    for rel, want in EXPECTED_SHA256.items():
        p = tok_path if rel.startswith("tokenizer/") else root / rel
        got = sha256(p) if p.exists() else "missing"
        ok &= got == want
        print(f"{'OK      ' if got == want else 'MISMATCH'} {p}")
    print("all files match the reference checksums" if ok else
          "some files differ from the reference files: results will not be directly comparable (see README)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
