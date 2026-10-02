"""Keep unused validation loaders independent of a resumed checkpoint cursor."""
from functools import wraps


def install_timing_data_hook(training, get_args):
    original = training.build_train_valid_test_data_loaders

    @wraps(original)
    def build(*args, **kwargs):
        settings = get_args()
        eval_iters = settings.eval_iters
        counts = eval_iters if isinstance(eval_iters, (list, tuple)) else [eval_iters]
        if any(count != 0 for count in counts) or getattr(settings, "full_validation", False):
            return original(*args, **kwargs)
        # Megatron constructs validation samplers even with eval-iters=0.
        # The checkpoint cursor may exceed this timing run's validation set.
        # Only the unused sampler starts at zero; retain the saved counter for
        # checkpoint/replay metadata. Never reset consumed_train_samples.
        saved = settings.consumed_valid_samples
        settings.consumed_valid_samples = 0
        try:
            if saved:
                training.print_rank_0(
                    f"[moc-data] validation disabled: initialize unused validation "
                    f"sampler at 0; preserve checkpoint consumed_valid_samples={saved}"
                )
            return original(*args, **kwargs)
        finally:
            settings.consumed_valid_samples = saved

    training.build_train_valid_test_data_loaders = build
