# MoEGambit Artifact

This artifact accompanies the paper **"MoEGambit: Contract-Based Hybrid Recovery for Mixture-of-Experts Training"**.

MoEGambit is a recovery-side framework for sparse Mixture-of-Experts (MoE) training.  When the runtime contract admits hybrid repair, the replacement rank restores dense-DP-replicated non-expert state from a healthy peer at the current step and restores only rank-local expert state from checkpoint.  This avoids replay while bounding the stale-expert exposure introduced by partial repair.

The artifact contains:

- Megatron-LM patches for safe-point repair, hybrid restore, the R2 staleness-density policy, two-phase optimizer recovery, and reintegration.
- Fault-injection and recovery scripts used for the paper experiments.
- Selected anonymized evaluation logs, analysis scripts, and plotting scripts for inspecting the paper results.
- Pre-generated PDF figures used by the paper.

## Paper Alignment

The artifact is aligned with the paper terminology and claims:

- **R1 Safe point:** optimizer commits are guarded while a failed rank is repaired.
- **R2 Staleness-density policy:** hybrid recovery is admitted only when
  `Phi'(t) = (S(t) + |E_new| * Delta) / (N_expert * W_exp) <= Phi_max`.
- **R3 Reintegration and logs:** decisions, guard inputs, latency segments, and reintegration transitions are logged for audit.
- **Hybrid restore:** non-expert state is pulled from a dense-DP peer; rank-local expert state is loaded from checkpoint when EDP=1.
- **Full-peer restore:** when EDP>1 and an expert peer is available, expert state can also be pulled from a peer and contributes zero stale-expert debt.
- **Two-phase protocol:** expert weights are restored first; optimizer state is restored later under an update barrier.

The paper reports raw recovery latency reductions of **20.6%--55.0%** and a **36.9x** replay-inclusive speedup for a 100-iteration replay gap.  The artifact follows the paper's latency scope and does not add stronger end-to-end claims.

## Directory Structure

```text
moegambit-artifact/
├── src/
│   ├── elastic/
│   │   ├── elastic_watcher.py
│   │   └── elastic_launcher.py
│   └── Megatron-LM/
│       ├── megatron/training/
│       │   ├── arguments.py
│       │   ├── checkpointing.py
│       │   ├── initialize.py
│       │   ├── training.py
│       │   └── ft_integration.py
│       └── megatron/core/transformer/moe/
│           ├── recovery_controller.py
│           ├── gap_aware_recovery_policy.py
│           ├── rank_exposure_tracker.py
│           ├── optimizer_commit_guard.py
│           ├── dense_param_sync.py
│           ├── stale_expert_restore.py
│           ├── deferred_optimizer_load.py
│           ├── two_phase_recovery.py
│           ├── reintegration_barrier.py
│           ├── dispatch_topology_refresh.py
│           ├── expert_directory.py
│           ├── group_rebuild.py
│           ├── fault_injection.py
│           ├── fault_injection_framework.py
│           └── moc_pec_emulation.py
├── scripts/
│   ├── run_main_exp_moeguard.sh
│   ├── run_main_exp_mocsystem.sh
│   ├── run_main_exp_baseline.sh
│   ├── run_moe64_hotspare.sh
│   ├── run_moe64_baseline.sh
│   ├── run_moe64_ablation.sh
│   ├── bench_moe64_tp1pp8ep4.sh
│   ├── bench_moe64_tp2pp4ep4.sh
│   ├── bench_moe64_tp2pp4ep8.sh
│   ├── run_moe128.sh
│   ├── bench_moe128.sh
│   ├── bench_dsv2lite.sh
│   ├── find_multi_fault.sh
│   ├── analyze_moegambit_log.py
│   ├── moegambit_timing_report.py
│   ├── moegambit_ablation_report.py
│   ├── parse_ablation.py
│   ├── plot_train_loss.py
│   ├── plot_fault_heatmap.py
│   ├── plot_parallelism_sensitivity.py
│   └── plot_scalability.py
├── data/
│   ├── logs/
│   │   └── evaluation/
│   │       ├── eval_moegambit.log
│   │       ├── eval_baseline.log
│   │       └── eval_mocsystem.log
└── figures/
    ├── moegambit_runtime_architecture.pdf
    ├── edp_distribution.pdf
    ├── train_loss.pdf
    ├── fault_heatmap_compact.pdf
    ├── scalability.pdf
    └── parallelism_sensitivity.pdf
```

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- NCCL distributed backend
- Megatron-LM dependencies
- NumPy, pandas, and matplotlib for analysis scripts
- A multi-node GPU cluster for full-scale reproduction

The paper experiments used Qwen3-30B-A3B on 64 H20 GPUs as the main setting and DeepSeek-V2-Lite as the cross-model setting.

## Recovery Policy Parameters

The paper R2 policy is exposed through the backward-compatible policy name `rank_exposure_guarded_hybrid`.

| Parameter | CLI flag | Paper default | Meaning |
| --- | --- | ---: | --- |
| `Delta_min` | `--moe-moegambit-delta-time-min-gap` | 1 | Minimum replay gap for hybrid recovery |
| `Delta_max` | `--moe-moegambit-max-single-gap` | 200 | Maximum single-event checkpoint gap |
| `W_exp` | `--moe-moegambit-exposure-window-steps` | 2000 | Sliding exposure window |
| `Phi_max` | `--moe-moegambit-max-rank-stale-exposure` | 0.1 | Maximum projected `Phi'(t)` |
| `N_expert` | Megatron `--num-experts` | 128 for Qwen3 | Routed experts in the exposure domain |

The flag name `--moe-moegambit-max-rank-stale-exposure` is retained for compatibility with earlier scripts, but its current meaning is the paper's `Phi_max`.  Structured decision logs include both the legacy field and the paper-facing fields:

- `num_affected_experts`
- `num_experts`
- `stale_expert_debt_before`
- `stale_expert_debt_after`
- `phi_prime_before`
- `phi_prime_after`
- `phi_max`

## Recovery Protocol

For a fail-stop rank event `<r,t,c>`:

1. The controller marks the failed rank and its experts as `RECOVERING`.
2. R1 installs an optimizer commit guard and discards any in-flight iteration.
3. R2 evaluates the guarded policy:
   - restart if no dense-DP peer is available;
   - restart if `Delta < Delta_min`;
   - restart if `Delta > Delta_max`;
   - restart if projected `Phi'(t) > Phi_max`;
   - otherwise admit hybrid recovery.
4. Hybrid recovery runs the two state paths:
   - Path P: pull dense-DP-replicated non-expert state from a healthy peer at step `t`.
   - Path C: load the failed rank's expert shard from checkpoint step `c`.
5. The two-phase protocol restores weights first, resumes under an update barrier, and attaches optimizer state before releasing the barrier.
6. R3 advances the replacement rank through `RECOVERING -> REPAIRED -> BARRIER -> HEALTHY` and emits structured logs.

## Key CLI Flags

All MoEGambit flags are prefixed with `--moe-moegambit-` and are defined in `src/Megatron-LM/megatron/training/arguments.py`.

| Flag | Purpose |
| --- | --- |
| `--moe-moegambit-enable` | Master switch for MoEGambit patches |
| `--moe-moegambit-recovery-controller` | Enable the recovery controller and state machine |
| `--moe-moegambit-hot-spare-pool` | Enable spare-rank replacement support |
| `--moe-moegambit-fault-injection` | Enable built-in fault injection |
| `--moe-moegambit-expert-directory` | Track expert placement and recovery state |
| `--moe-moegambit-dense-param-sync` | Enable dense/non-expert peer restore |
| `--moe-moegambit-stale-expert-restore` | Enable rank-local expert checkpoint restore |
| `--moe-moegambit-defer-optimizer-load` | Enable optimizer-later recovery |
| `--moe-moegambit-weights-first-recovery` | Restore weights before optimizer state |
| `--moe-moegambit-gap-aware-recovery` | Enable policy-based path selection |
| `--moe-moegambit-recovery-policy-type rank_exposure_guarded_hybrid` | Use the paper R2 policy |
| `--moe-moegambit-full-peer-recovery` | Use expert-peer restore when EDP>1 |
| `--moe-moegambit-force-checkpoint-restart` | Force the checkpoint-restart baseline |
| `--moe-moegambit-reintegration-barrier` | Enable guarded reintegration |

## Minimal Paper-Policy Example

```bash
python src/Megatron-LM/pretrain_gpt.py \
  --moe-moegambit-enable \
  --moe-moegambit-recovery-controller \
  --moe-moegambit-hot-spare-pool \
  --moe-moegambit-expert-directory \
  --moe-moegambit-dense-param-sync \
  --moe-moegambit-stale-expert-restore \
  --moe-moegambit-weights-first-recovery \
  --moe-moegambit-defer-optimizer-load \
  --moe-moegambit-gap-aware-recovery \
  --moe-moegambit-recovery-policy-type rank_exposure_guarded_hybrid \
  --moe-moegambit-delta-time-min-gap 1 \
  --moe-moegambit-max-single-gap 200 \
  --moe-moegambit-exposure-window-steps 2000 \
  --moe-moegambit-max-rank-stale-exposure 0.1 \
  --moe-moegambit-reintegration-barrier \
  --moe-moegambit-fault-injection \
  [standard Megatron args...]
```

## Fault Injection

Fault injection is controlled either by CLI flags or by environment variables:

| Variable | Description |
| --- | --- |
| `MOEGAMBIT_FAULT_INJECT` | Enable fault injection (`1` = on) |
| `MOEGAMBIT_FAULT_INJECT_STEP` | Training step to inject a fault |
| `MOEGAMBIT_FAULT_INJECT_RANK` | Global rank to fail |
| `MOEGAMBIT_FAULT_INJECT_TYPE` | Fault type (`kill`, `hang`, `oom`, or `restart_in_place`) |
| `MOEGAMBIT_FAULT_INJECT_INTERVAL` | Inject periodically every N steps |
| `MOEGAMBIT_FAULT_INJECT_SEED` | Random seed for fault schedules |
| `MOEGAMBIT_FAULT_INJECT_PLAN` | Pre-planned fault schedule |
| `MOEGAMBIT_FORCE_CHECKPOINT_RESTART` | Force checkpoint restart for baseline/emulation runs |

Example:

```bash
export MOEGAMBIT_FAULT_INJECT=1
export MOEGAMBIT_FAULT_INJECT_STEP=200
export MOEGAMBIT_FAULT_INJECT_RANK=7
export MOEGAMBIT_FAULT_INJECT_TYPE=restart_in_place
```

## Logs and Data

Only downstream evaluation logs are included as raw logs. Large training,
fault-injection, parallelism, scalability logs, and some intermediate CSV/JSON
summaries are omitted from the review-time package because they contain
internal operational data, including cluster-local paths, hostnames, scheduler
records, and non-public run-management metadata. The retained package provides
the anonymized implementation, experiment scripts, selected evaluation logs,
analysis scripts, and pre-generated figures; reviewers can inspect the
mechanisms and regenerate summaries when running the scripts on their own
environment.

The retained raw logs are under `data/logs/evaluation/`:

- `eval_moegambit.log`
- `eval_baseline.log`
- `eval_mocsystem.log`

Included result files rewrite cluster-local paths to `/anon/...` and private
control-plane addresses to non-identifying placeholders. Reproduction scripts
default to writing under `runs/`; set `ARTIFACT_RUN_ROOT`, `DATA_PATH`,
`CKPT_DIR`, or `TRAIN_LOG_DIR` to use site-local storage. Generated logs and
CSV/JSON summaries should be treated as site-local outputs and are therefore
not bundled when they cannot be safely anonymized.

## Reproducing Paper Analyses

Main recovery and baseline runs regenerate raw training/recovery logs under
`runs/` unless `ARTIFACT_RUN_ROOT` or `TRAIN_LOG_DIR` is set:

```bash
bash scripts/run_moe64_hotspare.sh
bash scripts/run_main_exp_baseline.sh
bash scripts/run_main_exp_mocsystem.sh
```

Mechanism decomposition regenerates per-mode logs and summaries:

```bash
bash scripts/run_moe64_ablation.sh
python scripts/parse_ablation.py
python scripts/moegambit_ablation_report.py <log_dir>
```

Multiple-failure and R2 stress tests regenerate sweep logs:

```bash
bash scripts/find_multi_fault.sh
python scripts/plot_fault_heatmap.py
```

Parallelism and scaling regenerate benchmark logs:

```bash
bash scripts/bench_moe64_tp1pp8ep4.sh
bash scripts/bench_moe64_tp2pp4ep4.sh
bash scripts/bench_moe64_tp2pp4ep8.sh
bash scripts/bench_moe128.sh
bash scripts/bench_dsv2lite.sh
python scripts/plot_parallelism_sensitivity.py
python scripts/plot_scalability.py
```

Downstream summaries can be inspected from the retained evaluation logs.
Training/recovery log analyzers are intended for newly generated logs:

```bash
python scripts/plot_train_loss.py
python scripts/analyze_train_log.py <new_log_file>
python scripts/analyze_moegambit_log.py <new_log_file>
```

## Notes for Reviewers

- `rank_exposure_guarded_hybrid` is a legacy CLI name.  The implementation now uses the paper R2 expert-weighted `Phi'(t)` guard.
- `max_rank_stale_exposure` is retained as a legacy flag name for `Phi_max`.
- Full reproduction requires the same multi-node GPU setting and dataset paths used by the paper.  The package retains raw evaluation logs only; training and recovery log-analysis scripts should be run on regenerated logs.
- `src/Megatron-LM/` is a complete Megatron-LM tree with MoEGambit patches applied.

## License and Citation

This artifact is distributed under multiple open-source licenses.  MoEGambit-specific files outside `src/Megatron-LM/` are released under Apache-2.0; the bundled Megatron-LM tree retains the upstream Megatron-LM and third-party licenses in `src/Megatron-LM/LICENSE`.

If you use this artifact, please cite the accompanying MoEGambit paper.  A starter `CITATION.cff` is included and should be updated with the final publication metadata before archival release.
