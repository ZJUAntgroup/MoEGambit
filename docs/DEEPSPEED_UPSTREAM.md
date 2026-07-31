# MoEGambit DeepSpeed Baseline

The bundled `DeepSpeed/` directory is based on the official
[`deepspeedai/DeepSpeed`](https://github.com/deepspeedai/DeepSpeed) repository:

- Tag: `v0.19.3`
- Commit: `0c36f6d3efb806c07a44e5c2c9b18a81d204b821`
- License: Apache-2.0

MoEGambit keeps the upstream runtime and build sources required to install
DeepSpeed. Large upstream blogs, generated documentation, CI metadata,
examples, and the full upstream test corpus are omitted from this repository.
The runtime integration patch is limited to calls through
`moegambit.adapters.deepspeed.hooks`. Recovery coordination, ordered process
group creation, packed expert checkpoints, and launcher recovery decisions
live in the installable MoEGambit package. `setup.py` and
`deepspeed/git_version_info.py` pin the snapshot's upstream commit instead of
inheriting metadata from the enclosing repository. The DeepSpeed launcher
forwards MoEGambit feature and recovery variables to remote workers alongside
`PYTHONPATH`.

`deepspeed/runtime/pipe/engine.py` contains one compatibility correction for
the v0.19.3 torch-style backward hooks. Non-final pipeline stages use one
`torch.autograd.backward()` call and leave backward timers and optimizer
prologue/epilogue ownership to the output hook manager. This avoids starting
the same timer twice and preserves one backward state-machine transition for
pipeline outputs containing multiple differentiable tensors.
