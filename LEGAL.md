# Legal Notices

This document records the licensing and third-party composition currently
visible in this repository. It is informational, is not legal advice, and does
not itself grant a license. The license text that applies to MoEGambit-owned
code is in the root `LICENSE` file.

## 1. Repository composition

This repository is a composite research repository. A root license must not be
interpreted as replacing the licenses retained by bundled upstream projects or
as establishing provenance for generated and cached research material.

| Path | Composition | Governing notice |
|---|---|---|
| `src/moegambit/` | Framework-neutral MoEGambit Python package | Apache-2.0; root `LICENSE` |
| `deepspeed_adapter/`, project launchers, tests, scripts and MoEGambit documentation | Project integration code, unless a file states otherwise | Apache-2.0; root `LICENSE` |
| `DeepSpeed/` | Microsoft DeepSpeed source tree with repository-local modifications | `DeepSpeed/LICENSE` and file-level notices |
| `DeepSpeed/deepspeed/inference/v2/kernels/cutlass_ops/` | Bundled kernel component | Its nested `LICENSE` and file-level notices |
| `Megatron-LM/` | NVIDIA Megatron-LM source tree with repository-local modifications and bundled third-party code | `Megatron-LM/LICENSE` and file-level notices |
| `moegambit-artifact/` | Packaged research artifact with its own bundled Megatron-LM copy | `moegambit-artifact/LICENSE`, `moegambit-artifact/NOTICE.md`, and nested notices |
| Logs, figures, PDFs, datasets, cached evaluation repositories and generated experiment outputs | Research inputs and outputs with mixed or unverified provenance | Not automatically covered by the root license; review before redistribution |

## 2. Installable MoEGambit package

The distribution declared by `pyproject.toml` collects packages only from
`src/` and declares the license expression `Apache-2.0`. The build configuration
includes both `LICENSE` and this `LEGAL.md` in source and wheel distributions.

The core package currently declares no required runtime dependency. Optional,
non-vendored dependencies are:

| Dependency | Purpose | Upstream license |
|---|---|---|
| PyTorch (`torch>=2.0`) | Tensor operations and distributed communication | BSD-3-Clause |
| pytest (`pytest>=7.0`) | Development and testing | MIT |

Dependency license names above are informational summaries. Upstream package
metadata and license files govern.

## 3. DeepSpeed subtree

`DeepSpeed/` contains an upstream DeepSpeed source tree and local changes. Its
root `DeepSpeed/LICENSE` contains the Apache License 2.0. The nested CUTLASS
kernel directory also retains its own `LICENSE`. Existing copyright,
attribution, trademark, patent and file-level notices must be preserved.

Apache-2.0 section 4(b) requires modified files distributed as derivative works
to carry prominent notices stating that they were changed. Before external
redistribution, identify locally modified DeepSpeed files and verify those
notices rather than relying only on this summary.

## 4. Megatron-LM subtree

`Megatron-LM/` is derived from NVIDIA Megatron-LM and has been modified by this
project. `Megatron-LM/LICENSE` begins with NVIDIA's BSD-3-Clause-style terms and
also records licenses for bundled code from other projects. File-level notices
remain authoritative.

Before redistribution, preserve upstream notices and verify modification
notices for locally changed files. The separate copy under
`moegambit-artifact/src/Megatron-LM/` is governed by the artifact's own license
and notice files together with its nested Megatron-LM license.

## 5. Architectural references

The generalized package structure is informed by publicly described design
patterns from DeepEP (a reusable communication/core package) and DualPipe (an
explicit training-loop integration). This repository's design documents record
architectural influence only; no copied DeepEP or DualPipe source code has been
identified in the generalized package implementation.

## 6. Open items before external distribution

The following items remain unresolved and must not be silently inferred from
the presence of `LICENSE` or `LEGAL.md`:

1. The copyright holder for project-authored files has not been formally
   recorded in this repository.
2. Most project source files do not carry copyright or SPDX headers.
3. Modification notices for locally changed `DeepSpeed/` and `Megatron-LM/`
   files have not been comprehensively audited.
4. Cached evaluation material, datasets, logs, figures, PDFs and other research
   artifacts have not received a complete provenance and redistribution audit.
5. Internal paths, addresses, tokens, hostnames and operational data must be
   reviewed before any public release.

These are release-governance blockers, not runtime failures. Adding this file
does not by itself make the full composite repository ready for public release.

## 7. Contact

Direct licensing questions to the repository owner. For bundled upstream
subtrees, their retained license and file-level notices govern and take
precedence over this summary.
