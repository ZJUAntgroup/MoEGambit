#!/usr/bin/env python3
"""
Patch HuggingFace datasets cache for backward compatibility.

Why: download_datasets.py was run with datasets>=4.x, which writes
features with `"_type": "List"`. Older datasets versions (<=3.x) used
`"_type": "Sequence"` and do not recognize `"List"`, raising:

    TypeError: must be called with a dataclass type or instance

This script rewrites `"_type": "List"` -> `"_type": "Sequence"` in every
dataset_info.json under EVAL_DATA_ROOT/hf_cache, preserving the original
file as `dataset_info.json.bak`. Idempotent: re-runs become no-ops.

Usage:
    EVAL_DATA_ROOT=/mnt/ais-c1/dataset/zds/evaldata \
        python eval/scripts/patch_dataset_cache.py
"""

import json
import os
import shutil
import sys
from pathlib import Path


def patch_features(node):
    """Recursively rewrite _type: List -> Sequence in a features tree."""
    changed = 0
    if isinstance(node, dict):
        if node.get("_type") == "List":
            node["_type"] = "Sequence"
            changed += 1
        for v in node.values():
            changed += patch_features(v)
    elif isinstance(node, list):
        for v in node:
            changed += patch_features(v)
    return changed


def patch_file(info_path: Path) -> int:
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    features = info.get("features")
    if not isinstance(features, dict):
        return 0
    n = patch_features(features)
    if n == 0:
        return 0
    bak = info_path.with_suffix(info_path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(info_path, bak)
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
    return n


def main() -> int:
    root = os.environ.get(
        "EVAL_DATA_ROOT", "/mnt/ais-c1/dataset/zds/evaldata"
    )
    cache = Path(root) / "hf_cache"
    if not cache.is_dir():
        print(f"[patch] cache dir not found: {cache}", file=sys.stderr)
        return 2

    total_files = 0
    total_changes = 0
    for info_path in cache.rglob("dataset_info.json"):
        n = patch_file(info_path)
        if n > 0:
            print(f"[patch] {info_path}: rewrote {n} List -> Sequence")
        total_files += 1
        total_changes += n

    print(
        f"[patch] scanned {total_files} dataset_info.json files, "
        f"rewrote {total_changes} List -> Sequence entries"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
