#!/usr/bin/env python3
"""
Patch lm-eval-harness installed task YAMLs so that ``dataset_path``
matches what is actually present in our local hf_cache (offline use).

Also short-circuits the datasets free-disk-space pre-check that fires
as a false positive on NFS / mounted volumes.

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

Additionally, even after fixing dataset_path, datasets>=2.x calls
``has_sufficient_disk_space`` inside DatasetBuilder.download_and_prepare
*even when no download is needed*. On NFS volumes shutil.disk_usage()
sometimes returns 0, producing a spurious
    OSError: Not enough disk space. Needed: Unknown size
This script also disables that guard.

All edits are **idempotent** and keep .bak copies.

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


def patch_datasets_disk_check() -> int:
    """Disable datasets' bogus free-disk-space pre-check.

    datasets>=2.x calls ``has_sufficient_disk_space(needed, cache_dir)``
    inside ``DatasetBuilder.download_and_prepare`` *even when the data
    is already on disk and no download is needed*. On NFS / mounted
    volumes where ``shutil.disk_usage`` returns 0 or raises, this fires
    as a false positive:

        OSError: Not enough disk space. Needed: Unknown size
        (download: Unknown size, generated: Unknown size,
         post-processed: Unknown size)

    The check is genuinely useful for real downloads but useless for
    our fully-cached offline setup, so we short-circuit it by injecting
    a guard in builder.py: any call ``has_sufficient_disk_space(...)``
    becomes ``True`` regardless of the underlying disk_usage result.

    The patch is **idempotent** (a marker comment prevents re-edit) and
    keeps a .bak of the original builder.py.
    """
    try:
        import datasets  # noqa: F401
    except ImportError as e:
        print(f"[patch] datasets not importable: {e}", file=sys.stderr)
        return 0

    builder_py = Path(datasets.__file__).resolve().parent / "builder.py"
    if not builder_py.is_file():
        print(f"[patch] datasets/builder.py not found: {builder_py}", file=sys.stderr)
        return 0

    text = builder_py.read_text(encoding="utf-8")

    marker = "# disk-space check disabled by patch_lm_eval_tasks.py"
    if marker in text:
        return 0  # already patched

    # Replace
    #   if not has_sufficient_disk_space(...):
    # with
    #   if False and not has_sufficient_disk_space(...):  # <marker>
    # so the original call still appears (for grep / future audit) but
    # never triggers the OSError branch.
    pattern = re.compile(
        r"^([ \t]*)if not has_sufficient_disk_space\(",
        re.MULTILINE,
    )

    def _sub(m: "re.Match[str]") -> str:
        indent = m.group(1)
        return f"{indent}if False and not has_sufficient_disk_space("

    new_text, n = pattern.subn(_sub, text)
    if n == 0:
        print(
            "[patch] WARN: no has_sufficient_disk_space() guard found in "
            f"{builder_py}; datasets internal API may have changed",
            file=sys.stderr,
        )
        return 0

    # Append marker as a top-of-file comment so re-runs are no-ops.
    new_text = f"{marker}\n" + new_text

    bak = builder_py.with_suffix(builder_py.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(builder_py, bak)
    builder_py.write_text(new_text, encoding="utf-8")
    print(
        f"[patch] datasets: {builder_py.name}  "
        f"short-circuited {n} has_sufficient_disk_space() call(s)"
    )
    return n


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
    n_disk = patch_datasets_disk_check()
    print(f"[patch] done. simple={n_simple} edits, mathqa={n_mathqa} edits, "
          f"datasets_disk_check={n_disk} edits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
