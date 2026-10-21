# Input provenance and comparison labels

These are supplied experiment exports, not generated observations. Numerical
precision in CSV/JSON preserves the original export; displayed values use the
displayed rounded precision. CSV line endings are normalized to LF; numerical
values are unchanged.

| Files | Source and retained scope |
| --- | --- |
| `recovery_scaling.csv` | Rounded recovery cell means supplied for the GPU/layout comparison. The historical `restart_seconds` column was renamed to `full_load_seconds`: this arm restores the replacement rank and continues at the current step; it does not include whole-job rollback/replay. |
| `checkpoint_results.csv`, `expert_count_results.csv`, `summary.json` | Selected benchmark full-Hybrid/Restart comparisons. On 2026-10-02 the authors confirmed that all reruns restore old expert weights and optimizer state, use whole-job Restart, and retain the supplied aggregate values. Existing actual-Restart rows take precedence over duplicate NoFault rows. `original_reference_arm` records archival provenance; `reference_arm` / `reference_policy` describes the confirmed benchmark comparison. Per-parameter state verification and rerun raw logs are not included. |
| `risk_summary.json`, `risk_trajectory.csv` | Four supplied seed-1234 selected histories to step 10,000, including repeated faults. Their prefixes are shared. `risk_calibrated=false` refers to this trajectory study, not the separate R2 audit. The exported evaluation grid starts at the first relevant fault; pre-fault validation samples are not included. |
| `architecture_summary.json`, `architecture_quality_results.csv` | Six supplied full-state Hybrid branches with separate Restart/NoFault references and five validation points per pair. GQA and DeepSeek-style MLA profiles also differ in shared experts, routing and MoE layer placement. These are window comparisons, not independent run-risk trials. |
| `r2_audit_aggregate.json` | Author-reported new fixed-size 200-run audit and explicit independent/frozen-policy protocol confirmation on 2026-10-03. Earlier 50 tests are excluded. No per-run records, predictor class/package, calibration method or training/calibration sample counts are inferred. |
| `moc_e2e_aggregate.csv` | Author-supplied physical MoC mechanism-port results: one controlled-restart window per arm, including recovery and replay. These are separate from the four replay-excluded layout means. |

The user-approved additional downstream plot uses the
zero-shot accuracy export (`downstream_accuracy.csv`). Each task's `acc`/`acc_norm` metric
and eight-task equal weighting are retained. Other additional table-derived
conclusion plots have been withdrawn; no raw samples are added.

The main README's 35.6× recovery speedup and 20-repeat overhead summary come from
the reported benchmark measurements. This bundle contains no raw
per-event timing records or per-repeat overhead samples from which to compute
new confidence intervals. The training-loss PDF is the supplied benchmark
figure; its raw three-arm logs must be supplied to the optional redraw script.

The 200-run audit and all short/shared-prefix quality cases remain separate.
The accompanying scripts compute source hashes, reject incompatible grids or
conflicting records, and never synthesize missing trajectories.


## Web update exports and R2 protocol correction (2026-10-21)

The original `1020result/update_pipeline.log` timing export and the sanitized
`1020result/summary.log` plan are preserved byte-for-byte in
`archive/web_update_20261020/`. The original plan had only the private
`original_quality_archive` path removed. Their historical `r2_enabled=false`
flag is superseded by the author's confirmation that the Web rerun enabled R2
admission **before Hybrid restoration** and retained the reported workflow and
quality outcomes.

The active `web_update_pipeline.csv` changes only the R2 flag and adds the arm's
R2 role. Every timing, GPU-hour, replay-count, endpoint-loss and speedup value is
unchanged. The active plan records the corrected protocol and its normalized
SHA-256. Its original cluster `config_hash` remains an archival identifier, not
proof of the corrected runtime configuration. `web_protocol_correction.json`
records the author confirmation, both original file hashes, corrected file
hashes, and the unchanged FiQA export hash. `scripts/web_protocol.py` verifies
that numerical measurements and non-R2 scientific parameters were preserved.

`fiqa_quality.csv` remains a byte-for-byte copy of
`1020result/webquality.log`. Its reported 2,000-replicate paired-query intervals
are retained rather than reconstructed from aggregate means. No query-level
rows, timing repeats, predictor parameters, risk scores, or action counts are
added by this correction. The forced Restart reference arm must not be counted
as an R2 rejection. The Web outcome study remains separate from the independent
200-run risk audit. See the bundle README for protocol and measurement scope.
