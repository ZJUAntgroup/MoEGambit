# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

try:
    #  This is populated by setup.py
    from .git_version_info_installed import *  # noqa: F401 # type: ignore
except ModuleNotFoundError:
    from pathlib import Path

    version_file = Path(__file__).resolve().parents[1] / 'version.txt'
    if version_file.is_file():
        # MoEGambit can launch directly from its pinned source snapshot.
        version = version_file.read_text(encoding='utf-8').strip() + '+0c36f6d'
    else:
        version = "0.0.0"
    git_hash = '0c36f6d'
    git_branch = 'v0.19.3'

    from .ops.op_builder.all_ops import ALL_OPS
    installed_ops = dict.fromkeys(ALL_OPS.keys(), False)
    accelerator_name = ""
    torch_info = {'version': "0.0", "cuda_version": "0.0", "hip_version": "0.0"}

# compatible_ops list is recreated for each launch
from .ops.op_builder.all_ops import ALL_OPS

compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)
for op_name, builder in ALL_OPS.items():
    op_compatible = builder.is_compatible()
    compatible_ops[op_name] = op_compatible
    compatible_ops["deepspeed_not_implemented"] = False
