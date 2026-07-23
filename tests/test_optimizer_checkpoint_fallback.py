import ast
from pathlib import Path


OPTIMIZER_PATH = (
    Path(__file__).parents[1]
    / "Megatron-LM"
    / "megatron"
    / "core"
    / "optimizer"
    / "optimizer.py"
)


class _Logger:
    def warning(self, *args, **kwargs):
        pass


class _Torch:
    saved = []

    @classmethod
    def save(cls, states, filename):
        cls.saved.append((states, filename))


class _Group:
    def __init__(self, rank):
        self._rank = rank

    def rank(self):
        return self._rank


class _DistributedOptimizer:
    def __init__(self, rank, state, *, use_gloo=False):
        self.data_parallel_group = _Group(rank)
        self.data_parallel_group_gloo = object() if use_gloo else None
        self.state = state
        self.calls = []

    def get_parameter_state_dp_zero(self, **kwargs):
        self.calls.append(kwargs)
        return self.state


def _build_chained_optimizer(optimizers):
    source = OPTIMIZER_PATH.read_text()
    tree = ast.parse(source)
    chained = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ChainedOptimizer"
    )
    save_method = next(
        node
        for node in chained.body
        if isinstance(node, ast.FunctionDef) and node.name == "save_parameter_state"
    )
    harness = ast.ClassDef(
        name="_Harness",
        bases=[],
        keywords=[],
        body=[save_method],
        decorator_list=[],
    )
    module = ast.Module(body=[harness], type_ignores=[])
    namespace = {"logger": _Logger(), "torch": _Torch}
    exec(compile(ast.fix_missing_locations(module), str(OPTIMIZER_PATH), "exec"), namespace)
    instance = namespace["_Harness"]()
    instance.chained_optimizers = optimizers
    return instance


def test_nccl_fallback_discards_nonroot_state_without_asserting():
    _Torch.saved = []
    nonroot = _DistributedOptimizer(rank=1, state={"all_gathered": True})
    chained = _build_chained_optimizer([nonroot, nonroot])

    chained.save_parameter_state("optimizer.pt")

    assert _Torch.saved == []
    assert nonroot.calls == [
        {"use_gloo_comm": False, "return_on_all_ranks": True},
        {"use_gloo_comm": False, "return_on_all_ranks": True},
    ]


def test_nccl_fallback_saves_only_states_owned_by_this_dp_root():
    _Torch.saved = []
    dense_root = _DistributedOptimizer(rank=0, state={"dense": "state"})
    expert_nonroot = _DistributedOptimizer(rank=1, state={"expert": "temporary"})
    chained = _build_chained_optimizer([dense_root, expert_nonroot])

    chained.save_parameter_state("optimizer.pt")

    assert _Torch.saved == [
        ([{"dense": "state"}, None], "optimizer.pt"),
    ]
