# MoEGambit 包隔离约定

## 当前结论

此前的双包隔离已经结束：

- `src/moegambit` 是唯一通用源码根；
- `deepspeed_adapter/moegambit` 已迁移并删除；
- `deepspeed_adapter/moegambit_deepspeed` 只包含 DeepSpeed 私有 adapter；
- 测试在同一 Python 进程中同时加载根包和可选 adapter；
- wheel 仍只从 `src` 收集通用包。

仓库源码运行时使用：

```bash
PYTHONPATH=/path/to/repository/src:/path/to/repository/deepspeed_adapter
```

该组合现在合法：第一个目录提供唯一 `moegambit`，第二个目录只提供
`moegambit_deepspeed`。

## 约束

不得在框架 adapter 下重新创建顶层 `moegambit`。新增公共协议、watcher、
transport、配置或 CLI 必须进入 `src/moegambit`；DeepSpeed 和 Megatron 私有
对象只能进入各自 adapter。

完整依赖方向和恢复流程见
`docs/design/UNIFIED_RECOVERY_ARCHITECTURE.md`。
