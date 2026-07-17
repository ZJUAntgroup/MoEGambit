# Contributing

Thanks for helping improve MoEGambit.

## Scope

This repository is primarily an artifact package. Contributions should keep the
implementation, scripts, logs, and documentation aligned with the accompanying
paper and should avoid broad rewrites of upstream Megatron-LM code unless they
are necessary for the recovery mechanism.

## Development Guidelines

- Keep generated outputs, checkpoints, large datasets, and site-local logs out
  of Git.
- Do not commit secrets, private hostnames, scheduler records, internal paths,
  access tokens, or raw cluster logs that have not been reviewed and anonymized.
- Prefer environment variables for local paths such as `DATA_PATH`, `CKPT_DIR`,
  `TRAIN_LOG_DIR`, and `ARTIFACT_RUN_ROOT`.
- Preserve upstream license headers and third-party notices in
  `src/Megatron-LM/`.
- Include focused tests or reproduction notes when changing recovery behavior,
  policy admission logic, reintegration, or logging semantics.

## Suggested Checks

Before opening a pull request, run the smallest relevant checks for your change:

```bash
python -m py_compile scripts/*.py src/elastic/*.py
python scripts/parse_ablation.py --help
python scripts/analyze_moegambit_log.py --help
```

Full end-to-end reproduction requires a multi-node GPU environment and is not
expected for small documentation or analysis-script changes.
