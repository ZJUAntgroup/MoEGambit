# Quality-risk admission and experiment evidence

This release adds an opt-in quality-risk policy, exact state inventory audits,
per-rank completion verification, and compact evidence exports. These tools do
not fit a predictor or establish an unconditional statistical guarantee for a
new model. Existing engine compatibility recovery paths remain available.

## Installation and a CPU smoke run

```bash
python -m pip install -e .
python examples/audit_smoke.py --log-dir /personal/moegambit_audit_smoke
moegambit-audit verify-run \
  --root /personal/moegambit_audit_smoke \
  --output /personal/moegambit_audit_smoke/verification.json
moegambit-audit collect \
  --root /personal/moegambit_audit_smoke \
  --output /personal/moegambit_audit_smoke_bundle
```

Use a fresh log directory. The smoke run contains synthetic scalar state and
synthetic completion records; it checks the tooling, not GPU recovery or model
quality. Output paths are explicit and portable. For the FSE cluster, keep all
logs/results under `/personal`; checkpoint payloads can use a separate
`CKPT_DIR` on `/zds`. No audit command copies checkpoint tensors.

Without installation, replace `moegambit-audit` with
`PYTHONPATH=src python -m moegambit.cli.audit` from the repository root.

## Which policy executes?

| Entry point | Admission behavior |
| --- | --- |
| Common watcher `--policy peer` (default) | Complete compatible live restore, otherwise complete checkpoint fallback |
| Common watcher `--policy quality-risk --quality-mode audit-only` | Validate/log the new rule; never execute a candidate hybrid restore |
| Common watcher `--policy quality-risk --quality-mode enforce` | Execute a candidate hybrid only with qualified, supported, context-bound whole-run risk evidence |
| Common watcher `--policy moe-hybrid` | Legacy expert-staleness density policy; not the new paper's quality-risk rule |
| Vendored Megatron/DeepSpeed compatibility examples and watchers | Existing adapter-specific policies; not automatically migrated by selecting the common watcher policy |

All common policies retain source completeness, version consistency and adapter
capability checks. The common coordinator currently accepts one failed logical
rank per recovery epoch. This addition does not extend that execution boundary.
A complete current live copy uses the peer path; the quality gate applies to a
candidate hybrid restore that would introduce stale expert state. A checkpoint
fallback is a plan decision; whole-job relaunch requires the existing fallback
controller/launcher contract. FullLoad of a replacement rank must not be
reported as whole-job restart.

The new policy does **not** use checkpoint-age or expert-density admission
cutoffs. Age may be a predictor feature. It does not set or relax the quality
budget. The default targets are fixed across supported models:

```text
eta_final = 0.005
eta_peak  = 0.01
alpha_run = 0.05
Y = max(final relative loss degradation / eta_final,
        maximum predeclared evaluation-step relative degradation / eta_peak)
R = qualified upper bound on Pr(Y > 1 for the complete run | recovery history)
    / alpha_run
admit candidate hybrid iff R <= 1
```

Loss comparisons must use the same validation examples, committed training
steps and predeclared evaluation schedule. Across tokenizers, define a common
text-normalized metric (such as loss per byte). The final/peak tolerances and
alpha are predeclared targets, not thresholds fitted from the existing 15
checkpoint experiments.

### Coordinator-owned evidence provider

`QualityRiskPolicy` accepts an injected `RiskEvidenceProvider.evaluate(facts)`.
A worker cannot authorize hybrid simply by sending a probability. The watcher
owns the provider. An adapter may implement the optional
`state.quality_recovery_context(query)` hook to provide JSON containing:

```json
{
  "model_id": "registered-model-configuration",
  "telemetry_version": "telemetry-v1",
  "policy_version": "frozen-policy-v1",
  "features": {},
  "run_history": []
}
```

In production, features should include normalized routed mass, sensitivity,
parameter and optimizer drift summaries, route change, learning rate and
training stage. Retain global sums and worst local effects; averaging by layer
count can hide accumulated effects. Record previous recoveries and violations
for the **entire run**, including after a fallback. Mark an observed peak
quality violation with `peak_violation_observed: true` in its history entry;
the policy forces the whole-run risk bound to 1 because a later restart cannot
erase a historical maximum. The external predictor must
independently reconcile this history with authoritative run records. A flag in
a worker request or a reset history is not evidence of safety.

`risk_context(facts)` and `risk_context_digest(facts)` expose the canonical
conditioning snapshot for a predictor. The coordinator supplies authoritative
job/attempt, recovery epoch, topology generation and group manifest. The digest
also binds failed ranks, step, all candidate source versions/locators,
capabilities, features, and recovery/exposure history. Ranks submitting different
quality contexts for the same frozen epoch are rejected. A custom provider can
compute an estimate synchronously from this snapshot; the file provider is a
simple bridge for an external process that publishes predictions atomically.

For the file provider, both artifacts must be managed by the coordinator's
operator (not a worker-writable probability file):

```bash
moegambit-watcher \
  --rendezvous-host "$MASTER_ADDR" --rendezvous-port 29502 \
  --policy quality-risk --quality-mode enforce \
  --quality-calibration "$LOG_DIR/calibration.json" \
  --quality-evidence "$LOG_DIR/current_prediction.json"
```

Calibration is pinned when the watcher starts; each new candidate decision
reads a fresh prediction. An already frozen recovery plan stays frozen.
Authentication, job-token and control-store options follow the normal watcher
configuration. Protect these files with the same ownership/access boundary as
the coordinator configuration.

Calibration schema (version 1):

| Field | Required value/meaning |
| --- | --- |
| `schema_version` | Integer 1 |
| `expires_at` | ISO timestamp with timezone |
| `bound_scope` | `whole_run_conditional`; per-fault or marginal bounds are rejected for this gate |
| `qualification` | `operator_reviewed` only after external methodological review |
| `independent_audit` | Boolean true, declaring an audit unused for fitting/tuning |
| `method`, `assumptions` | Nonempty description of the external bound and its justified conditions |
| `audit_report_sha256` | Lowercase 64-character supporting audit report digest |
| `outcome_definition` | `paired-whole-run-final-and-peak-v1` |
| `eta_final`, `eta_peak`, `alpha_run` | Exactly match the policy targets |
| `support` | Arrays under `model_id`, `telemetry_version`, `policy_version` |

Prediction schema (version 1):

| Field | Required value/meaning |
| --- | --- |
| `schema_version` | Integer 1 |
| `calibration_digest` | `moegambit.audit.io.digest(calibration)` |
| `context_digest` | `risk_context_digest(facts)` for this exact candidate |
| `expires_at` | Unexpired timestamp with timezone |
| `risk_upper` | Finite number in [0,1] |
| `support_checked`, `in_support`, `run_history_checked` | Boolean true, attested by the trusted predictor |

The file provider attaches the pinned `calibration` object to this prediction.
Custom providers must return that object themselves. An unqualified/expired
artifact, missing history, unseen model, unsupported features, mismatched
snapshot or malformed bound selects checkpoint fallback, or aborts if there is
no complete checkpoint. With no provider, audit-only still records why evidence
is missing. Enforce configuration without a provider fails at startup.

**Contract checks are not a mathematical certificate.** A schema-valid file or
`operator_reviewed` label cannot prove conditional coverage. The operator must
justify the bound's statistical method, assumptions, calibration protocol and
support checks. Do not relabel a marginal conformal bound or an empirical failure
frequency as a conditional whole-run bound. New architectures can retain the
same targets but need support validation and independently qualified evidence.
No trained/calibrated provider or cross-architecture guarantee is bundled here.

## Exact state audits

Capture the actual values through `capture_catalog(catalog)`, then publish with
`moegambit.audit.io.write_json`. Tensor capture needs optional PyTorch and hashes
CPU contiguous raw bytes, including BF16. Scalar accessors use canonical JSON;
provide a JSON encoding for custom RNG/scheduler objects. Non-finite floating
values are rejected. This diagnostic may synchronize/copy device tensors, so run
it at a quiescent audit boundary, not on every training step.

An expected manifest must be captured **independently from the selected source
payloads**, remapped to logical destination ownership and expected provenance.
Do not derive expected state from the just-restored destination. For a hybrid
restore, expert entries may have checkpoint versions while unaffected entries
have current versions. All required identities must be explicitly inventoried:
parameters, FP32 master weights, Adam first/second moments, step counters,
loss scale, persistent buffers/router bias, scheduler, RNG and data progress.
Current framework catalogs do not universally enumerate every one of these;
extend/configure your adapter inventory before claiming complete recovery.

```bash
moegambit-audit state \
  --expected "$LOG_DIR/expected_state.json" \
  --observed "$LOG_DIR/observed_state.json" \
  --before "$LOG_DIR/before_state.json" \
  --affected 'layer.1.expert.7.weight' \
  --output "$LOG_DIR/state_audit.json"
```

`--affected` may be repeated. The command compares exact digest, dtype, shape,
kind, placement, owner and all provenance-version fields. Missing, unexpected
or duplicate identities fail. With `--before`, every unaffected identity must
retain content and committed/optimizer versions; recovery epoch may advance.
The audit is exact, not a floating-tolerance or model-quality test. It proves
agreement with the supplied independent inventory, not its completeness.

## Per-rank completion and recovery records

Use a shared result directory and the same run manifest on every rank:

```python
from moegambit.audit.run import RunEvidenceWriter

manifest = {
    "schema_version": 1,
    "job_id": "experiment-001", "attempt_id": "attempt-001",
    "code_commit": "actual-git-commit", "world_size": world_size,
    "final_step": planned_final_step,
    "required_recovery_epochs": [1],
    "required_files": ["paired_quality.json", "state_audit.json"],
}
writer = RunEvidenceWriter(log_dir, manifest, rank)
# For the public common runtime, at construction:
# moegambit.initialize(adapter, config, event_sink=writer.record_recovery)
# Run the training/recovery loop; finish required evaluation and audit work.
writer.complete(actual_final_step, exit_code=0)
```

The writer refuses a second process for the same rank. Use a fresh attempt and
result directory when restarting; never manufacture completion files to make an
interrupted run pass. Recovery records are published after the runtime commits
a complete post-recovery iteration, plus terminal fallback/abort records.
Publication failures include a traceback in the runtime logger; the verifier
then cannot find the required commit evidence. Call `complete` only after the
loop/evaluations actually finish; this is an instrumentation contract, not
independent process attestation. Completion files are published atomically.

```bash
moegambit-audit verify-run --root "$LOG_DIR" \
  --output "$LOG_DIR/verification.json"
```

Verification requires all logical ranks to report the exact target step,
commit, manifest, attempt and successful exit; it also requires a committed
full-step recovery record from every rank for every declared epoch. Duplicate
records, unresolved provisional recoveries, malformed/truncated files,
fallbacks/aborts and rank disagreements fail. Required evidence files must be
present and nonempty; their scientific content needs the separate state/quality
audits. The verifier does not infer completion from low GPU utilization, the
last printed step, or one master success line. It does not prove survivor PID
continuity or measure end-to-end speedup by itself.

## Compact evidence export

```bash
moegambit-audit collect --root "$QUALITY_DIR" \
  --output /personal/fse_evidence_bundle
```

The collector recognizes existing FSE summary CSVs, paired-quality JSONs,
training-loss/replay diagnostics and selected case manifests, plus independent state manifests and the new
rank/recovery evidence. Older pipelines without the new run manifest can still
be exported; verification is explicitly marked incomplete. Checkpoint trees,
binary payloads, hidden folders and symlinks are excluded. Each copied file has
a size and SHA-256 in `evidence_manifest.json`. Existing destinations are never
overwritten. Default data budget is 100 MiB; oversized files are omitted with a
reason. `--max-mib` and `--max-files` bound output.

Logs are excluded by default. Use `--include-logs` only for sanitized logs;
check them for tokens, private dataset paths or other information before sharing.
The tool does not export environment variables or credentials and does not
promise to remove secrets already embedded in allowed scientific result files.

## Independent whole-run statistical audit

```bash
moegambit-audit risk --trials 59 --violations 0 --confidence 0.95 \
  --alpha-run 0.05 --output "$LOG_DIR/risk_audit.json"
```

This computes the one-sided exact Clopper--Pearson upper bound without scipy.
With 59 independent complete runs and zero violations, the 95% upper bound is
below 5%; 58 runs are insufficient. Fifteen zero-violation runs give an upper
bound of about 18.1%, not 5%. Checkpoint splices from one seed/training trajectory
are correlated and do not count as 59 independent runs. Freeze policy/support
before auditing and predeclare how trials are sampled. If the estimand is risk
among admitted runs, select and count independent admitted whole runs under
that frozen sampling protocol; do not discard violations or failed evaluations.

This is a **marginal** whole-run audit under independence and the declared
sampling distribution. It is not evidence of a conditional guarantee for every
fault state and cannot alone qualify the enforce policy's conditional bound.
It is not robust to distribution shift or audit reuse for tuning. Report the
assumptions and missing evaluations with the results.
