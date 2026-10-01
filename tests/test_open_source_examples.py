from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_installation_and_both_frameworks():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "python -m pip install -e '.[dev]'" in readme
    assert "examples/megatron/run_hot_spare.sh" in readme
    assert "examples/deepspeed/run_hot_spare.sh" in readme
    assert "DRY_RUN=1" in readme
    assert "Environment requirements" in readme
    assert "Network ports" in readme


def test_example_scripts_are_executable_and_use_validated_entrypoints():
    expectations = {
        "megatron/run_hot_spare.sh": "test_hotspare_replace.sh",
        "deepspeed/run_hot_spare.sh": "test_deepspeed_hotspare_replace.sh",
    }
    for relative_path, entrypoint in expectations.items():
        script = ROOT / "examples" / relative_path
        source = script.read_text(encoding="utf-8")
        assert os.access(script, os.X_OK)
        assert entrypoint in source
        assert "DRY_RUN" in source
        assert "/shared/moegambit/" in source
        assert re.search(r"/mnt/[^/]+/dataset/[^/]+", source) is None


def test_dense_examples_dry_run_training_and_spare_commands():
    for framework in ("megatron", "deepspeed"):
        script = ROOT / "examples" / framework / "run_dense.sh"
        assert os.access(script, os.X_OK)
        for node_rank in ("0", "2"):
            result = subprocess.run(
                ["bash", str(script)],
                cwd=ROOT,
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "DRY_RUN": "1",
                    "NODE_RANK": node_rank,
                },
                capture_output=True,
                text=True,
                check=True,
            )
            assert "MOEGAMBIT_MODEL_KIND=dense" in script.read_text(encoding="utf-8")
            expected_entrypoint = (
                "elastic_watcher.py" if node_rank == "2" else "elastic_launcher.py"
            )
            assert expected_entrypoint in result.stdout
            assert "--num-experts" not in result.stdout


def test_framework_neutral_package_has_one_source_root():
    package_roots = [
        path
        for path in ROOT.rglob("moegambit/__init__.py")
        if "Megatron-LM" not in path.parts and "DeepSpeed" not in path.parts
    ]

    assert package_roots == [ROOT / "src" / "moegambit" / "__init__.py"]


def test_root_elastic_entrypoints_are_adapter_dispatchers():
    launcher = (ROOT / "elastic_launcher.py").read_text(encoding="utf-8")
    watcher = (ROOT / "elastic_watcher.py").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "moegambit.cli.elastic_compat" in launcher
    assert "moegambit.cli.elastic_compat" in watcher
    assert "--adapter deepspeed" in readme
    assert "compat_launcher.py" not in launcher
    assert "run_spare_single_rank.sh" not in watcher
