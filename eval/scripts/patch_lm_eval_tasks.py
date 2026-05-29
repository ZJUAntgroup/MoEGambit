#!/usr/bin/env python3
"""
Patch lm-eval-harness installed task YAMLs so that ``dataset_path``
matches what is actually present in our local hf_cache (offline use).

Why this exists
---------------
``download_datasets.py`` was run against a different set of HuggingFace
mirrors than what lm-eval ships in its default task YAMLs, e.g.:

    lm-eval expects                  | local cache actually has
    ---------------------------------|------------------------------------
    super_glue                       | aps___super_glue
    openbookqa                       | allenai___openbookqa
    piqa                             | lighteval___piqa
    EleutherAI/race                  | ehovy___race
    swag                             | allenai___swag
    winogrande                       | allenai___winogrande
    math_qa                          | repos/mathqa/{train,val,test}.parquet

Without patching, lm_eval ends in offline mode with
    ConnectionError: Couldn't reach 'super_glue' on the Hub (OfflineModeIsEnabled)

This script rewrites the dataset_path (and where needed dataset_kwargs)
in lm-eval's *installed* YAML files so the offline cache hits. It is
**idempotent**: it only edits if the line still says the old value, and
keeps a .bak copy of the file the first time it modifies it.

Run env:
    EVAL_DATA_ROOT=/mnt/ais-c1/dataset/zds/evaldata \
        python eval/scripts/patch_lm_eval_tasks.py

Set SKIP_LM_EVAL_TASK_PATCH=1 to bypass (run_eval.sh respects it).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path


# Mapping: task name -> (relative yaml path inside lm_eval/tasks/,
#                       old dataset_path, new dataset_path).
# For mathqa we additionally inject dataset_kwargs (handled separately).
SIMPLE_PATCHES = [
    # task             yaml relative path                      old              new
    ("boolq",         "super_glue/boolq/default.yaml",       "super_glue",    "aps/super_glue"),
    ("openbookqa",    "openbookqa/openbookqa.yaml",          "openbookqa",    "allenai/openbookqa"),
    ("piqa",          "piqa/piqa.yaml",                      "piqa",          "lighteval/piqa"),
    ("race",          "race/race.yaml",                      "EleutherAI/race", "ehovy/race"),
    ("swag",          "swag/swag.yaml",                      "swag",          "allenai/swag"),
    ("winogrande",    "winogrande/default.yaml",             "winogrande",    "allenai/winogrande"),
]

# mathqa needs special handling: convert the loader from the broken HF
# 'math_qa' loading-script repo to a local parquet load via the parquet
# builder. The data files were materialised by download_datasets.py at
# $EVAL_DATA_ROOT/repos/mathqa/{train,validation,test}.parquet.
MATHQA_YAML_REL = "mathqa/mathqa.yaml"


def _backup(p: Path) -> None:
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(p, bak)


def _replace_dataset_path(text: str, old: str, new: str) -> tuple[str, int]:
    """Replace lines like 'dataset_path: <old>' (yaml-style) with new value.
    Returns (text, n_replacements).
    Only matches when the path is exactly <old> (avoid partial matches).
    """
    pattern = re.compile(
        rf"^([ \t]*dataset_path:[ \t]*){re.escape(old)}[ \t]*$",
        re.MULTILINE,
    )
    new_text, n = pattern.subn(rf"\g<1>{new}", text)
    return new_text, n


def find_lm_eval_tasks_dir() -> Path:
    try:
        import lm_eval  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"lm_eval not importable: {e}")
    tasks_dir = Path(lm_eval.__file__).resolve().parent / "tasks"
    if not tasks_dir.is_dir():
        raise SystemExit(f"lm_eval tasks dir not found: {tasks_dir}")
    return tasks_dir


def patch_simple(tasks_dir: Path) -> int:
    """Run all SIMPLE_PATCHES. Returns total number of lines modified."""
    total = 0
    for task, rel, old, new in SIMPLE_PATCHES:
        yaml = tasks_dir / rel
        if not yaml.is_file():
            # Some lm-eval versions split YAMLs differently; try a glob.
            cand = list(tasks_dir.glob(rel.replace("/default.yaml", "/*.yaml")))
            if not cand:
                print(f"[patch] WARN: yaml not found for {task}: {yaml}", file=sys.stderr)
                continue
            yaml = cand[0]
        text = yaml.read_text(encoding="utf-8")
        new_text, n = _replace_dataset_path(text, old, new)
        if n == 0:
            # Already patched, or different value -> skip.
            continue
        _backup(yaml)
        yaml.write_text(new_text, encoding="utf-8")
        print(f"[patch] {task}: {yaml.name}  dataset_path: {old!r} -> {new!r}  ({n} line)")
        total += n
    return total


def patch_mathqa(tasks_dir: Path, eval_data_root: Path) -> int:
    yaml = tasks_dir / MATHQA_YAML_REL
    if not yaml.is_file():
        print(f"[patch] WARN: mathqa yaml not found: {yaml}", file=sys.stderr)
        return 0

    repo_dir = eval_data_root / "repos" / "mathqa"
    train = repo_dir / "train.parquet"
    val = repo_dir / "validation.parquet"
    test = repo_dir / "test.parquet"
    if not (train.is_file() and val.is_file() and test.is_file()):
        print(
            f"[patch] WARN: mathqa parquet files missing under {repo_dir}; "
            f"skipping mathqa patch (task will fail if you try to run it)",
            file=sys.stderr,
        )
        return 0

    text = yaml.read_text(encoding="utf-8")

    # Idempotency marker: if our marker line is already there, skip.
    marker = "# patched by patch_lm_eval_tasks.py"
    if marker in text:
        return 0

    # We rewrite the whole file to a deterministic form. Preserve the
    # rest of the original config by parsing lines and dropping any
    # dataset_path / dataset_name / dataset_kwargs at the top level,
    # then prepending our overrides.
    lines = text.splitlines()
    kept: list[str] = []
    skip_until_dedent = False
    for line in lines:
        stripped = line.lstrip()
        if skip_until_dedent:
            # We are inside a dataset_kwargs: block (indented). Skip
            # lines that are still indented; stop at the first line
            # that is non-indented and non-empty.
            if line.startswith((" ", "\t")) or stripped == "":
                continue
            skip_until_dedent = False
        if stripped.startswith("dataset_path:") or stripped.startswith("dataset_name:"):
            continue
        if stripped.startswith("dataset_kwargs:"):
            skip_until_dedent = True
            continue
        kept.append(line)

    header = [
        marker,
        "dataset_path: parquet",
        "dataset_kwargs:",
        "  data_files:",
        f"    train: {train}",
        f"    validation: {val}",
        f"    test: {test}",
    ]
    new_text = "\n".join(header + [""] + kept).rstrip() + "\n"
    _backup(yaml)
    yaml.write_text(new_text, encoding="utf-8")
    print(f"[patch] mathqa: {yaml.name}  rewired to local parquet under {repo_dir}")
    return 1


def main() -> int:
    eval_data_root = Path(os.environ.get(
        "EVAL_DATA_ROOT", "/mnt/ais-c1/dataset/zds/evaldata"
    )).resolve()
    if not eval_data_root.is_dir():
        print(f"[patch] EVAL_DATA_ROOT not found: {eval_data_root}", file=sys.stderr)
        return 2

    tasks_dir = find_lm_eval_tasks_dir()
    print(f"[patch] lm_eval tasks dir : {tasks_dir}")
    print(f"[patch] EVAL_DATA_ROOT    : {eval_data_root}")

    n_simple = patch_simple(tasks_dir)
    n_mathqa = patch_mathqa(tasks_dir, eval_data_root)
    print(f"[patch] done. simple={n_simple} edits, mathqa={n_mathqa} edits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
