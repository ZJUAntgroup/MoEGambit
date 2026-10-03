# Input provenance and comparison labels

These are supplied manuscript exports, not generated observations. Numerical
precision in CSV/JSON preserves the original export; displayed values use the
paper's rounded precision. CSV line endings are normalized to LF; numerical
values are unchanged.

| Files | Source and retained scope |
| --- | --- |
| `recovery_scaling.csv` | Rounded recovery cell means supplied for the paper's GPU/layout figure. The historical `restart_seconds` column was renamed to `full_load_seconds`: this arm restores the replacement rank and continues at the current step; it does not include whole-job rollback/replay. |
| `checkpoint_results.csv`, `expert_count_results.csv`, `summary.json` | Publication-selected full-Hybrid/Restart comparisons. On 2026-10-02 the authors confirmed that all reruns restore old expert weights and optimizer state, use whole-job Restart, and retain the supplied aggregate values. Existing actual-Restart rows take precedence over duplicate NoFault rows. `original_reference_arm` records archival provenance; `reference_arm` / `reference_policy` describes the confirmed publication comparison. Per-parameter state verification and rerun raw logs are not included. |
| `risk_summary.json`, `risk_trajectory.csv` | Four supplied seed-1234 selected histories to step 10,000, including repeated faults. Their prefixes are shared. `risk_calibrated=false` refers to this trajectory study, not the separate R2 audit. The exported evaluation grid starts at the first relevant fault; pre-fault validation samples are not included. |
| `architecture_summary.json`, `architecture_quality_results.csv` | Six supplied full-state Hybrid branches with separate Restart/NoFault references and five validation points per pair. GQA and DeepSeek-style MLA profiles also differ in shared experts, routing and MoE layer placement. These are window comparisons, not independent run-risk trials. |
| `r2_audit_aggregate.json` | Author-reported new fixed-size 200-run audit and explicit independent/frozen-policy protocol confirmation on 2026-10-03. Earlier 50 tests are excluded. No per-run records, predictor class/package, calibration method or training/calibration sample counts are inferred. |
| `moc_e2e_aggregate.csv` | Author-supplied physical MoC mechanism-port results: one controlled-restart window per arm, including recovery and replay. These are separate from the four replay-excluded layout means. |

The user-approved additional downstream plot uses the current manuscript's
zero-shot table (`downstream_accuracy.csv`). Each task's `acc`/`acc_norm` metric
and eight-task equal weighting are retained. Other additional table-derived
conclusion plots have been withdrawn; no raw samples are added.

The main README's 35.6× recovery speedup and 20-repeat overhead summary come from
the authors' reported manuscript measurements. This bundle contains no raw
per-event timing records or per-repeat overhead samples from which to compute
new confidence intervals. The training-loss PDF is the supplied publication
figure; its raw three-arm logs must be supplied to the optional redraw script.

The 200-run audit and all short/shared-prefix quality cases remain separate.
The accompanying scripts compute source hashes, reject incompatible grids or
conflicting records, and never synthesize missing trajectories.
