"""Run the timing suite after native checkpoint loading, then 5 real smoke steps."""
from pathlib import Path
import os
import runpy
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def main():
    sys.path.insert(0, str(ROOT / "Megatron-LM"))
    from torch.distributed.elastic.multiprocessing.errors import record
    from moc_timing_plan import Plan
    from moc_timing_launch import atomic_json
    plan = Plan.from_env()
    @record
    def execute():
        import torch
        import torch.distributed as dist
        from megatron.training import training, get_args
        from moc_timing_runtime import Suite
        setup = training.setup_model_and_optimizer
        step = training.train_step
        smoke = []
        def with_benchmark(*args, **kwargs):
            model, optimizer, scheduler = setup(*args, **kwargs)
            settings = get_args()
            if int(settings.iteration) != plan.step:
                raise ValueError("native checkpoint did not load the requested baseline step")
            atomic_json(Path(plan.result_dir) / "worker_errors" / f"environment.rank_{dist.get_rank():03d}.json",
                        {"rank": dist.get_rank(), "torch": torch.__version__, "cuda": torch.version.cuda,
                         "device": torch.cuda.get_device_name(),
                         "checkpoint_step": settings.iteration,
                         "num_experts": settings.num_experts, "num_layers": settings.num_layers,
                         "use_distributed_optimizer": settings.use_distributed_optimizer,
                         "micro_batch_size": settings.micro_batch_size})
            Suite(plan, model, optimizer, settings).run()
            return model, optimizer, scheduler
        def checked_step(*args, **kwargs):
            t0 = time.perf_counter()
            result = step(*args, **kwargs)
            torch.cuda.synchronize()
            if result[1] or result[3]:
                raise RuntimeError("smoke step was skipped or requested early exit")
            for value in result[0].values():
                if not torch.isfinite(torch.as_tensor(value)).all():
                    raise RuntimeError("nonfinite smoke training loss")
            smoke.append({"step": int(get_args().curr_iteration) + 1,
                          "local_elapsed_s": time.perf_counter() - t0})
            return result
        training.setup_model_and_optimizer = with_benchmark
        training.train_step = checked_step
        runpy.run_path(str(ROOT / "Megatron-LM" / "pretrain_gpt.py"), run_name="__main__")
        rank = dist.get_rank()
        if len(smoke) != plan.smoke_steps:
            raise RuntimeError(f"expected {plan.smoke_steps} smoke steps, observed {len(smoke)}")
        atomic_json(Path(plan.result_dir) / "ranks" / f"smoke.rank_{rank:03d}.json",
                    {"rank": rank, "count": len(smoke), "rows": smoke,
                     "interpretation": "runnability check after final fixed-state restore; no quality claim"})
        dist.barrier()
        if rank == 0:
            from moc_timing_summarize import summarize
            summarize(Path(plan.result_dir))
        dist.barrier()
    try:
        execute()
    except BaseException:
        error = {"global_rank": os.environ.get("RANK"), "local_rank": os.environ.get("LOCAL_RANK"),
                 "time": time.time(), "traceback": traceback.format_exc()}
        try:
            directory = Path(plan.result_dir) / "worker_errors"
            directory.mkdir(exist_ok=True)
            atomic_json(directory / f"FAILED.rank_{os.environ.get('RANK', 'unknown')}.json", error)
        except OSError as e:
            print(f"[moc-timing] failure cannot be recorded: {e}; {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
