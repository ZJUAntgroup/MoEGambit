from __future__ import annotations

import os
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
        assert "/mnt/ais-c1/dataset/zds/" in source


def test_framework_neutral_package_has_one_source_root():
    package_roots = [
        path
        for path in ROOT.rglob("moegambit/__init__.py")
        if "Megatron-LM" not in path.parts and "DeepSpeed" not in path.parts
    ]

    assert package_roots == [ROOT / "src" / "moegambit" / "__init__.py"]
