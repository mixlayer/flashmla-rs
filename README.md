# flashmla-rs

Rust bindings for DeepSeek's [FlashMLA](https://github.com/deepseek-ai/FlashMLA) focused on sparse MLA
prefill and decode without depending on libtorch, pybind11, or Python extension loading.

## Requirements

- Rust with Cargo, edition 2024 support.
- NVIDIA CUDA Toolkit with `nvcc` available through `CUDA_HOME`, `CUDA_ROOT`, `CUDA_PATH`, or `PATH`.
- An SM90/Hopper or SM100/Blackwell datacenter GPU for kernel execution.
- CUDA 12.8 or newer for SM90; the upstream `sm_100f` sources require CUDA 12.9 or newer.
- Git submodules initialized, including FlashMLA's nested CUTLASS submodule.
- Linux build environment with a C++20-capable host compiler, `ar`, and CUDA driver/runtime
  libraries.

SM120/SM121 are not supported by upstream FlashMLA sparse MLA and should use FlashInfer instead.

## Setup

Initialize the vendored FlashMLA source:

```bash
git submodule update --init --recursive
```

Build and test the workspace:

```bash
cargo test --workspace
CUDA_COMPUTE_CAP=90 cargo test --workspace
CUDA_COMPUTE_CAP=100 cargo test --workspace
```

When neither `CUDA_COMPUTE_CAP` nor `FLASHMLA_ARCHS` is set, the build queries all GPUs reported
by `nvidia-smi` and selects their common compute capability. A mixed-architecture host is rejected
because one `flashmla-sys` artifact contains exactly one architecture. CI, cross-compilation, and
container builds without a visible GPU should continue to set the target explicitly.

Run the GPU smoke tests explicitly on the matching architecture:

```bash
CUDA_COMPUTE_CAP=90 cargo test -p candle-flashmla -- --ignored --nocapture
CUDA_COMPUTE_CAP=100 cargo test -p candle-flashmla \
  sparse_prefill_sm100_smoke -- --ignored --nocapture
CUDA_COMPUTE_CAP=100 cargo test -p candle-flashmla \
  sparse_decode_sm100_smoke -- --ignored --nocapture
```

Run only one smoke test:

```bash
CUDA_COMPUTE_CAP=90 cargo test -p candle-flashmla sparse_prefill_sm90_smoke -- --ignored --nocapture
CUDA_COMPUTE_CAP=90 cargo test -p candle-flashmla sparse_decode_sm90_smoke -- --ignored --nocapture
```

Use an external FlashMLA checkout instead of the vendored submodule:

```bash
FLASHMLA_ROOT=/path/to/FlashMLA CUDA_COMPUTE_CAP=100 cargo test --workspace
```

Useful build variables:

- `CUDA_COMPUTE_CAP=90` or `100`: deterministically selects the SM90a or SM100f build path.
- `FLASHMLA_ARCHS=sm90a` or `sm100f`: equivalent explicit single-architecture selection.
- `FLASHMLA_ROOT=/path/to/FlashMLA`: overrides the vendored submodule.
- `FLASHMLA_BUILD_DIR=/path/to/cache`: controls where CUDA object files and the static archive are
  written.
- `CANDLE_NVCC_CCBIN=/path/to/compiler`: passes a custom host compiler to `nvcc`.
- `FLASHMLA_NO_CCACHE=1`: disables `ccache` if present.
- `NVCC_THREADS=N`: passes `--threads N` to `nvcc`.
- `FLASHMLA_PTXAS_VERBOSE=1`: enables verbose ptxas output.

### Unsupported architectures

When the selected architecture has no implemented FlashMLA path, the build script emits a Cargo
warning, enables the `unsupported_arch` configuration, and skips CUDA source compilation and native
linking. This allows every crate in the workspace to compile on targets such as SM120 and SM121.
`flashmla-sys` provides ABI-compatible stubs in this configuration: kernel and device-query calls
return `FLASHMLA_STATUS_UNSUPPORTED_ARCH`, and `flashmla_last_error()` reports that FlashMLA is
unavailable on the selected architecture. The stubs do not provide a kernel fallback; callers
should use another implementation such as FlashInfer.

### `flashmla-sys`

Raw FFI and CUDA build crate.

- Discovers FlashMLA from `FLASHMLA_ROOT` or `crates/flashmla-sys/vendor/FlashMLA`.
- Compiles a static `libflashmla.a` from selected upstream CUDA sources plus the local C ABI wrapper.
- Exposes C-compatible status codes, device query, sparse prefill, sparse decode planning, and
  sparse decode launch symbols.
- Does not depend on Candle, cudarc, libtorch, pybind11, or Python.

### `flashmla`

Candle-independent Rust wrapper.

- Converts raw FFI status codes into Rust errors.
- Provides typed parameter structs for prefill and decode.
- Performs shape, stride, workspace, and architecture-related validation before entering FFI.

### `candle-flashmla`

Candle integration crate.

- Depends on Candle with CUDA enabled.
- Validates Candle tensor dtype, shape, device, and layout.
- Extracts CUDA stream/device pointers while preserving cudarc synchronization guards until launches
  are scheduled.
- Allocates output tensors and sparse decode scheduler/workspace tensors.
- Provides the model-facing `sparse_prefill`, `sparse_decode_plan`, and `sparse_decode` APIs.

## Implementation Status

Status values:

- `Done`: implemented and covered by a GPU smoke test.
- `Partial`: scaffolding or source compilation exists, but the full public path is not complete.
- `Planned`: expected, but not wired yet.
- `Unsupported`: intentionally not supported for this project/upstream combination.

| Kernel | SM90 / Hopper | SM100 / Blackwell | SM120 / SM121 |
| --- | --- | --- | --- |
| Sparse BF16 prefill | Done: C ABI, Rust wrapper, Candle API, smoke test | Done: architecture-specific dispatch, validation, and B200 smoke test | Unsupported |
| Sparse BF16-query / FP8-cache decode | Done: C ABI, Rust wrapper, Candle plan/run API, smoke test | Done: both planning branches, decode dispatch, combine, and graph-capture B200 smoke test | Unsupported |
| Dense prefill | Planned | Planned | Unsupported |
| Dense decode | Planned | Planned | Unsupported |

## Design Notes

- The C ABI intentionally avoids upstream FlashMLA's `csrc/api/*.h` and `api.cpp` paths because they
  include libtorch-facing helpers.
- C++ exceptions are caught at the ABI boundary and returned as `flashmla_status_t` plus
  `flashmla_last_error()`.
- Tensor memory is caller-owned. The FFI functions enqueue CUDA work on the provided stream and do
  not synchronize.
- Sparse decode uses explicit caller-owned scheduler metadata, `num_splits`, `lse_accum`, and
  `o_accum` buffers allocated by the Candle integration layer.
- SM100 V32 cache blocks use a byte pitch divisible by 656. MODEL1 keeps its 584-byte logical row
  layout but each cache block must be padded so the block pitch is divisible by the upstream TMA
  stride of 576 bytes.
- SM90a and SM100f are separate compile-time targets. The C ABI rejects a current device whose
  exact compute capability does not match the compiled target before launching a kernel.
