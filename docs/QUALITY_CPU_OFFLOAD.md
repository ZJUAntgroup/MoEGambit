# Quality features after a rank loses its state

Quality admission uses telemetry captured **before failure**. Training produces
compact feature tensors at a committed-step boundary. `AsyncQualityOffloader`
freezes those tensors, copies CUDA values to pinned CPU memory on a transfer
stream, and publishes JSON records from a background thread. An independent
watcher retains the acknowledged records in `CPUQualityFeatureStore`.

The CPU copy inside a training worker alone would disappear when that process
exits. Place the watcher in a separate process; for training-node failure,
place it on a different host. The RAM store survives worker loss, not watcher
or watcher-host loss. If the watcher restarts without the required snapshots,
the quality gate selects checkpoint fallback. This feature does not persist
parameter checkpoints or replace the existing whole-job relaunch contract.

## What is captured, and when?

| Input | Producer before failure | Retained form |
| --- | --- | --- |
| Routing mass `p` and routing changes | Router counters, normalized using the complete layer's routed-token total | Arrays with stable layer/expert IDs; global normalization must already be available |
| Sensitivity `s` | The configured predictor's sensitivity estimator | Estimates plus their estimator/telemetry version; load alone is not sensitivity |
| Parameter drift `d_theta` | Training telemetry relative to the selected durable expert checkpoint | Compact scale-normalized drift summaries with the checkpoint step |
| First/second-moment drift | Optimizer telemetry using the same expert/checkpoint reference | `d_momentum` and `d_variance` summaries, not replacement values for optimizer state |
| Learning rate and training stage | Scheduler and committed-step progress | Scalars or arrays using the predeclared stage definition |
| Prior recoveries and quality violations | Run-history recorder | An append-only whole-run history, including events before checkpoint rollback |

The publisher accepts nested JSON and dense CPU/CUDA tensors. It does not
invent sensitivity estimates, infer optimizer drift from checkpoint age, or
fit a predictor. A feature producer must supply the estimator-specific inputs
used by its calibrated model, including any required checkpoint references.
The predictor consumes `quality_context.features.by_rank`; it uses the
affected-rank set and stable layer/expert ownership to select exposed experts,
and preserves global sums and worst local effects rather than silently
averaging over layers. It must qualify this schema/version in calibration.

Each record binds job/attempt, owner rank, model/policy/telemetry identity,
committed step, source recovery epoch/topology generation/manifest, and checkpoint reference.
The recovery request separately names the **next** topology generation.
The coordinator freezes one complete retained context with its recovery plan;
later uploads cannot change the frozen decision. Snapshot digests are part of
the predictor's existing context digest.

Every logical rank publishes a record, including ranks with no local experts.
For that case, publish explicit empty expert arrays alongside its relevant
non-expert telemetry. All ranks must use the same history and reference.
Missing ranks, conflicting records, topology mismatch, a different checkpoint
reference, and missing exact-step telemetry fail closed. An old record is not
silently presented as current. The retention store can hold records for
multiple failed ranks, but the common recovery coordinator still supports one
failed logical rank per epoch; this change does not extend execution support.

## Enable the independent watcher

Add CPU retention to the common quality-policy watcher. Keep the watcher and
worker job/attempt/token settings aligned, and use the same control endpoint:

```bash
moegambit-watcher \
  --bind-host "$QUALITY_HOST" --port "$QUALITY_PORT" \
  --job-token "$MOEGAMBIT_JOB_TOKEN" --require-token \
  --rendezvous-host "$MASTER_ADDR" --rendezvous-port 29502 \
  --policy quality-risk --quality-mode audit-only \
  --quality-cpu-features --quality-retain-steps 4 \
  --quality-max-bytes 67108864 --quality-max-scopes 8
```

Audit-only does not execute candidate Hybrid restores. To enable enforcement,
use the qualified provider/artifacts described in [the risk-policy guide](ARTIFACT_AUDIT.md).
A custom coordinator-owned `RiskEvidenceProvider.evaluate(facts)` receives the
assembled snapshot directly. The external file-provider bridge must generate
fresh evidence for that same complete context, including its retained-record
digests. Old predictions without these bindings are rejected.

Defaults retain four distinct steps per scope, cap each record at 256 KiB and
cap aggregate encoded record payloads at 64 MiB (Python/control-server object
overhead is additional). Job/attempt/topology/model identities form a scope;
the default maximum is eight scopes per watcher. Configure these bounds for
the expected run and topology changes. Bounds reject additional data instead
of silently discarding the current snapshot or authorizing a partial one.

## Connect the training loop

Create one uploader per logical rank using a dedicated `ControlClient` so
background publication cannot hold the recovery coordinator client's lock.

```python
from moegambit.quality import AsyncQualityOffloader
from moegambit.runtime.client import ControlClient, ControlClientConfig

uploader = AsyncQualityOffloader(
    ControlClient(ControlClientConfig(
        quality_host, quality_port, job_id, attempt_id,
        {"global_rank": global_rank}, job_token=job_token,
        request_timeout_s=5.0,
    )),
    model_id=model_id, telemetry_version="quality-telemetry-v1",
    policy_version="frozen-risk-policy-v1", max_pending=2,
)

# RecoveryRuntime's commit_iteration hook calls this provider only after a
# committed optimizer step and after record_checkpoint has registered a
# durable checkpoint. Providers supply existing telemetry; no GPU state is
# read from a failed worker during recovery.
runtime.configure_quality_offload(
    uploader,
    feature_provider=lambda step, checkpoint_step: telemetry.compact_features(
        step=step, checkpoint_step=checkpoint_step),
    history_provider=lambda: run_history.snapshot(),
)
```

`telemetry` and `run_history` above are application-owned producers, not
bundled trained estimators. `runtime` is the common `RecoveryRuntime` instance.
Its existing Megatron training facade calls the optimizer/iteration hooks;
attach this configuration to that same instance, not to a second runtime.
When the coordinator is `WatcherRecoveryCoordinator`, configuration also
installs the model/telemetry/policy identity in recovery requests. An adapter's
optional `quality_recovery_context` must agree with those identities.
The watcher replaces worker-supplied features/history with retained records.

For a custom Megatron or DeepSpeed loop outside the common runtime, call the
same uploader at the loop's **proven global commit boundary**:

```python
pending = uploader.submit(
    telemetry.compact_features(step=step, checkpoint_step=checkpoint_step),
    committed_step=step, checkpoint_step=checkpoint_step,
    topology_generation=topology.generation,
    group_manifest_hash=topology.manifest_hash,
    world_size=world_size, recovery_epoch=recovery_epoch,
    run_history=run_history.snapshot(),
)
```

An engine step counter alone does not prove the distributed commit. The custom
loop must provide its existing device-completion/commit-fence proof first,
and use a common coordinator configured with `quality_identity=uploader.identity`.
Legacy engine recovery controllers are not automatically migrated by adding
a publisher. Register the durable checkpoint only after every shard/manifest
is complete; a partially saved checkpoint must never be the drift reference.

`submit` returns immediately after immutable capture and queuing. It does not
wait for transfer completion or the network ACK on the training thread.
A Future succeeds only after watcher acknowledgement. A full bounded queue
returns `None`; publication failures are logged and recorded in `stats()`.
Missing telemetry forces recovery fallback. Keep the full history across
relaunches, even when optimizer/data progress rolls back to an older checkpoint.

CUDA cloning, tensor reductions in the producer, allocation and submission
still have costs on the training path; asynchronous publication does not
establish zero overhead. Measure the complete path on the target cluster.
Read `uploader.stats()` periodically and keep stderr/application logs under
`LOG_DIR`. `RecoveryRuntime.describe()` includes these counters. Close the
uploader during normal shutdown; after relaunch create a new client/uploader
for the new attempt rather than mutating one with pending uploads.

## CPU smoke and validation

```bash
python examples/quality_features/cpu_retention_smoke.py \
  --log-dir /personal/moegambit_quality_cpu_smoke
python -m pytest tests/unit/test_quality_offload.py tests/unit/test_quality_risk.py -q
```

The smoke starts a real local control server, publishes synthetic features
from two subprocesses, abruptly exits one worker after its ACK, and writes
`result.json` plus worker logs. It confirms retained inputs and rejection of
an unpublished next step. It does not establish model quality or a risk bound.
The test suite also checks immutable capture, bounded queues, histories,
reference mismatches, conflicts, and admission/fallback integration. Tensor
tests require PyTorch; the CUDA stream-ordering test requires a CUDA device.
