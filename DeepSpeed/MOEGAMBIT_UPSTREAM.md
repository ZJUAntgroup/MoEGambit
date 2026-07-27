# MoEGambit DeepSpeed Baseline

This directory is based on the official
[`deepspeedai/DeepSpeed`](https://github.com/deepspeedai/DeepSpeed) repository:

- Tag: `v0.19.3`
- Commit: `0c36f6d3efb806c07a44e5c2c9b18a81d204b821`
- License: Apache-2.0

MoEGambit keeps the upstream runtime and build sources required to install
DeepSpeed. Large upstream blogs, generated documentation, CI metadata,
examples, and the full upstream test corpus are omitted from this repository.
The runtime integration patch in `deepspeed/__init__.py` is deliberately
limited to configuration validation and an optional
post-engine-construction hook. Both are inactive when the MoEGambit feature
switches are off. `setup.py` and `deepspeed/git_version_info.py` also pin the
snapshot's upstream commit instead of inheriting metadata from the enclosing
MoEGambit repository. `deepspeed/launcher/runner.py` forwards MoEGambit
feature and recovery variables to remote workers alongside `PYTHONPATH`.
