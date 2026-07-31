"""Stable hook surface imported by patched Megatron source files.

Megatron must not import recovery implementation modules directly.  This
facade is the only supported dependency from Megatron into MoEGambit; the
implementation remains owned by the adapter package.
"""

from .elastic_client import (
    elastic_align_resume_state,
    elastic_clear_post_rebuild_trace,
    elastic_client_update_step,
    elastic_configure_recovery_nccl_transport,
    elastic_create_rebuild_store,
    elastic_expert_sidecar_available,
    elastic_is_post_rebuild_trace_active,
    elastic_on_nccl_error,
    elastic_post_rebuild_iteration_barrier,
    elastic_prearm_standby_cuda_runtime,
    elastic_refresh_prearmed_standby_assignment,
    elastic_replacement_sync_params,
    elastic_report_recovery_phase,
    elastic_restore_expert_sidecar,
    elastic_sanitize_recovery_env_for_startup,
    elastic_save_expert_sidecar,
    elastic_trace_post_rebuild_phase,
    elastic_wait_for_ordinal_barrier,
    elastic_wait_for_recovery_phase_count,
    is_rebuild_mode,
)
from .integration import (
    after_optimizer_step,
    before_optimizer_step,
    bootstrap_control_plane,
    commit_iteration,
    finalize_replacement,
    get_runtime,
    initialize_runtime,
    iteration_boundary,
    on_distributed_error,
)
from .moe.preferential_routing import get_all_preferential_routing_managers
from .native_hooks import megatron_hooks
from .moe_integration import (
    maybe_initialize_moegambit_moe,
    moegambit_advance_iteration,
    moegambit_after_iteration,
    moegambit_announce_replacement_ready,
    moegambit_before_iteration,
    moegambit_clear_checkpoint_restart,
    moegambit_complete_replay,
    moegambit_exceeded_max_replays,
    moegambit_get_checkpoint_restart_decision,
    moegambit_get_reintegration_summary,
    moegambit_has_pending_replacements,
    moegambit_is_checkpoint_restart_requested,
    moegambit_is_current_iteration_invalid,
    moegambit_is_initialized,
    moegambit_is_reintegration_pending,
    moegambit_is_replay_pending,
    moegambit_is_waiting_for_replacement,
    moegambit_mark_optimizer_committed,
    moegambit_mark_optimizer_skipped,
    moegambit_pipeline_begin_iteration,
    moegambit_pipeline_complete_replay,
    moegambit_pipeline_initiate_rollback,
    moegambit_pipeline_is_in_rollback,
    moegambit_pipeline_on_failure,
    moegambit_post_load_checkpoint,
    moegambit_pre_save_checkpoint,
    moegambit_query_replacement_status,
    moegambit_report_hard_failure,
    moegambit_rollback_iteration,
    moegambit_save_manifest,
    moegambit_should_commit_optimizer,
    moegambit_snapshot_iteration,
)

__all__ = [
    name
    for name in globals()
    if (
        name.startswith("elastic_")
        or name.startswith("moegambit_")
        or name
        in {
            "after_optimizer_step",
            "before_optimizer_step",
            "bootstrap_control_plane",
            "commit_iteration",
            "finalize_replacement",
            "get_all_preferential_routing_managers",
            "get_runtime",
            "initialize_runtime",
            "is_rebuild_mode",
            "iteration_boundary",
            "maybe_initialize_moegambit_moe",
            "megatron_hooks",
            "on_distributed_error",
        }
    )
]
