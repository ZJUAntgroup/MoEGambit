# MoEGambit 包隔离约定

## 结论

`runtime` 分支不删除 `main` 已有的 `DeepSpeed/` 或
`deepspeed_adapter/`。删除这些受同事维护的目录会形成 Git 删除提交，
后续合并时可能误删对方代码。

当前采用逻辑隔离：

- 通用核心的唯一正式源码根目录是 `src/moegambit`；
- 默认测试只将 `src` 放入 Python 导入路径；
- wheel 只从 `src` 收集 `moegambit*`，不会打包 `deepspeed_adapter`；
- DeepSpeed 测试必须使用独立 Python 进程和独立 package profile；
- 核心包发现另一个顶层 `moegambit` 时立即拒绝导入，不再依赖路径顺序猜测。

## 核心开发和测试

默认 profile 是 `core`：

```bash
python -m pytest -q
```

该 profile 会屏蔽 `tests/test_deepspeed_real_adapter.py`，并在测试会话开始时
确认 `moegambit` 实际来自 `src/moegambit/__init__.py`。

## 单独验证 DeepSpeed 目录

需要验证同事目录时，必须启动新的 Python 进程：

```bash
MOEGAMBIT_TEST_PACKAGE_PROFILE=deepspeed \
python -m pytest tests/test_deepspeed_real_adapter.py -q
```

当前 DeepSpeed 兼容包使用 Python 3.10 以上的类型语法，因此该命令需要
Python 3.10 或更高版本。不要在同一 pytest 进程中先运行核心测试，再切换
到 DeepSpeed 包。

## 禁止的路径组合

以下配置仍然是不合法的：

```bash
PYTHONPATH=/path/to/repository/src:/path/to/repository/deepspeed_adapter
```

两处都包含顶层 `moegambit`，Python 原生导入机制无法把它们当成两个不同
实现。核心包会检测这种情况并抛出 `PackageIdentityError`，防止静默导入错误
实现。

## 最终集成要求

逻辑隔离消除了当前分支开发、测试和 wheel 发布中的随机导入，但没有改变
同事目录仍含第二份 `moegambit` 的事实。最终合并双方方案时必须选择一种
根治方式：

1. 将 `deepspeed_adapter/moegambit` 的通用能力迁入 `src/moegambit`，然后
   删除第二份核心；或
2. 把同事目录中的兼容核心改成唯一名称，只保留
   `moegambit_deepspeed` 作为显式插件。

在完成其中一种方式前，不允许同时把两个源码根目录加入 `PYTHONPATH`。
