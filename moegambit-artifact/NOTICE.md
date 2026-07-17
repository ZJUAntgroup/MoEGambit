# Notices

MoEGambit is a research artifact for contract-based hybrid recovery in
Mixture-of-Experts training.

This repository includes a complete `src/Megatron-LM/` source tree with
MoEGambit patches applied for reproducibility. Megatron-LM and its bundled
third-party code retain their original notices and licenses in
`src/Megatron-LM/LICENSE` and in file-level headers.

The retained logs under `data/logs/evaluation/` are anonymized evaluation logs.
Large training logs, checkpoints, datasets, scheduler records, machine-local
paths, and other site-local outputs are intentionally excluded from the public
artifact.

Generated experiment outputs should be written to `runs/` or another
site-local directory and should not be committed unless they have been reviewed
for public release.
