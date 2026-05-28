#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download all 8 evaluation datasets used by MoEGuard's downstream eval into
a local directory tree that can be tarballed and shipped to an air-gapped
GPU host.

Run this on a workstation **with internet access** (e.g. your Mac):

    pip install -U datasets huggingface_hub
    python3 download_datasets.py --out ./data

It will:
  1. snapshot_download each dataset repo from HuggingFace Hub
  2. also call datasets.load_dataset(...) once with cache_dir=./data/hf_cache
     so that an arrow-format cache exists for fast offline load
  3. write a manifest.json listing every (name, repo, config, split) combo

To ship to the GPU host:

    cd /Users/zds/bsr/eval && tar czf eval_data.tar.gz data/
    scp eval_data.tar.gz user@gpu:/mnt/ais-c1/dataset/zds/eval/
    ssh user@gpu 'cd /mnt/ais-c1/dataset/zds/eval && tar xzf eval_data.tar.gz'

On the GPU host the evaluation driver (run_eval.sh) sets
HF_DATASETS_CACHE / HF_DATASETS_OFFLINE=1 so no network is touched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------
# Dataset registry.
# Each entry is (name, repo_id, config, splits, allow_patterns).
# allow_patterns prevents snapshot_download from pulling unrelated configs
# (e.g. super_glue ships 11 sub-tasks but we only need boolq).
# ----------------------------------------------------------------------
DATASETS = [
    # name           repo_id                config            splits                              allow_patterns
    ("boolq",       "aps/super_glue",      "boolq",          ["train", "validation"],            ["boolq/*", "README.md"]),
    ("winogrande",  "allenai/winogrande",  "winogrande_xl",  ["train", "validation"],            ["winogrande_xl/*", "README.md"]),
    # EleutherAI/race no longer ships the 'middle' config in parquet
    # form; ehovy/race is the maintained parquet mirror with high+middle.
    ("race_middle", "ehovy/race",          "middle",         ["train", "validation", "test"],    ["middle/*", "README.md"]),
    # math_qa is special-cased below (raw zip, no datasets loader).
    ("swag",        "allenai/swag",        "regular",        ["train", "validation"],            None),  # repo is small
    # ybisk/piqa is a loading script; lighteval/piqa is the parquet mirror.
    ("piqa",        "lighteval/piqa",      "plain_text",     ["train", "validation", "test"],    ["plain_text/*", "README.md"]),
    ("arc_easy",    "allenai/ai2_arc",     "ARC-Easy",       ["train", "validation", "test"],    ["ARC-Easy/*", "README.md"]),
    ("openbookqa",  "allenai/openbookqa",  "main",           ["train", "validation", "test"],    ["main/*", "README.md"]),
]

# math_qa's HF repo is a loading script; new datasets (>=3) rejects loading
# scripts. We mirror the upstream zip directly and write the splits as
# parquet so the GPU host can read them with HF_DATASETS_OFFLINE=1.
MATHQA_URL = "https://math-qa.github.io/math-QA/data/MathQA.zip"


def download_one(repo_id: str, out_dir: Path, allow_patterns=None) -> Path:
    """Mirror the dataset repo's raw files (parquet / json / loader script)."""
    from huggingface_hub import snapshot_download

    local_dir = out_dir / "repos" / repo_id.replace("/", "__")
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [snapshot] {repo_id}  ->  {local_dir}  (patterns={allow_patterns})")
    # huggingface_hub >=0.23 dropped local_dir_use_symlinks; pass kwargs
    # conditionally so this script works on both old (0.20-) and new (1.x) APIs.
    import inspect
    sig = inspect.signature(snapshot_download)
    kwargs = dict(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )
    if "local_dir_use_symlinks" in sig.parameters:
        kwargs["local_dir_use_symlinks"] = False
    snapshot_download(**kwargs)
    return local_dir


def warm_cache(repo_id: str, config: str | None, splits: list[str], cache_dir: Path) -> dict:
    """Trigger a real load so the arrow cache is materialised."""
    from datasets import load_dataset
    import inspect as _inspect

    info = {}
    sig = _inspect.signature(load_dataset)
    extra = {}
    # trust_remote_code was removed in datasets>=3; only pass it on old versions.
    if "trust_remote_code" in sig.parameters:
        extra["trust_remote_code"] = True

    for split in splits:
        try:
            print(f"  [cache ] {repo_id} cfg={config} split={split}")
            ds = load_dataset(
                repo_id,
                name=config,
                split=split,
                cache_dir=str(cache_dir),
                **extra,
            )
            info[split] = len(ds)
        except Exception as e:
            print(f"    !! cache miss for split={split}: {e}", file=sys.stderr)
            info[split] = f"ERROR: {e!r}"
    return info


def download_mathqa(out_dir: Path, cache_dir: Path) -> dict:
    """math_qa is a loading script on HF (rejected by datasets>=3).
    Mirror the upstream zip and convert to parquet under a synthetic
    repo dir so lm-eval can load it via load_dataset('parquet', data_files=...).
    """
    import urllib.request
    import zipfile
    import json as _json

    print("\n==== mathqa  (raw zip from math-qa.github.io) ====")
    local_repo = out_dir / "repos" / "mathqa"
    local_repo.mkdir(parents=True, exist_ok=True)
    zip_path = local_repo / "MathQA.zip"
    if not zip_path.exists():
        print(f"  [fetch ] {MATHQA_URL}")
        urllib.request.urlretrieve(MATHQA_URL, zip_path)

    extract_dir = local_repo / "raw"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    # Upstream zip ships train.json / dev.json / test.json (json arrays).
    # Convert to parquet via datasets.Dataset.from_list for HF compatibility.
    from datasets import Dataset

    splits_info = {}
    for src_name, hf_name in [("train.json", "train"),
                              ("dev.json", "validation"),
                              ("test.json", "test")]:
        # The zip extracts to a top-level dir; find the json wherever it lives.
        matches = list(extract_dir.rglob(src_name))
        if not matches:
            print(f"    !! missing {src_name} in zip", file=sys.stderr)
            splits_info[hf_name] = "ERROR: file not in zip"
            continue
        src = matches[0]
        with open(src) as fh:
            rows = _json.load(fh)
        ds = Dataset.from_list(rows)
        out_pq = local_repo / f"{hf_name}.parquet"
        ds.to_parquet(str(out_pq))
        splits_info[hf_name] = len(ds)
        print(f"  [convert] {src_name} -> {out_pq.name}  ({len(ds)} rows)")

    # Warm the parquet loader cache as well.
    try:
        from datasets import load_dataset
        load_dataset(
            "parquet",
            data_files={k: str(local_repo / f"{k}.parquet")
                        for k in splits_info if isinstance(splits_info[k], int)},
            cache_dir=str(cache_dir),
        )
    except Exception as e:
        print(f"    !! parquet warm failed: {e}", file=sys.stderr)

    return {
        "name": "mathqa",
        "repo": "math-qa.github.io (raw zip)",
        "config": None,
        "local_repo_dir": str(local_repo.relative_to(out_dir)),
        "splits": splits_info,
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("./data"),
                        help="output root (will contain repos/ and hf_cache/)")
    parser.add_argument("--skip-warm", action="store_true",
                        help="skip arrow-cache warmup (snapshot only)")
    args = parser.parse_args()

    out_dir: Path = args.out.resolve()
    cache_dir = out_dir / "hf_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Point HF libs at our local cache so warm_cache writes there.
    os.environ["HF_HOME"] = str(out_dir / "hf_home")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    manifest = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "datasets": []}

    for name, repo_id, config, splits, patterns in DATASETS:
        print(f"\n==== {name}  ({repo_id}{' / ' + config if config else ''}) ====")
        try:
            local = download_one(repo_id, out_dir, allow_patterns=patterns)
        except Exception as e:
            print(f"  XX snapshot failed: {e}", file=sys.stderr)
            manifest["datasets"].append({
                "name": name, "repo": repo_id, "config": config,
                "status": f"snapshot_failed: {e!r}",
            })
            continue

        split_info = {}
        if not args.skip_warm:
            split_info = warm_cache(repo_id, config, splits, cache_dir)

        manifest["datasets"].append({
            "name": name,
            "repo": repo_id,
            "config": config,
            "local_repo_dir": str(local.relative_to(out_dir)),
            "splits": split_info,
            "status": "ok",
        })

    # math_qa: raw-zip fallback (no HF parquet mirror exists).
    try:
        manifest["datasets"].append(download_mathqa(out_dir, cache_dir))
    except Exception as e:
        print(f"  XX mathqa failed: {e}", file=sys.stderr)
        manifest["datasets"].append({"name": "mathqa", "status": f"failed: {e!r}"})

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nmanifest written -> {manifest_path}")
    print(f"total size (du): run `du -sh {out_dir}` to check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
