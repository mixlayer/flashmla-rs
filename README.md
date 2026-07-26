# flashmla-rs

Rust bindings for DeepSeek's [FlashMLA](https://github.com/deepseek-ai/FlashMLA) focused on sparse MLA
prefill and decode without depending on libtorch, pybind11, or Python extension loading.

## Requirements

- Rust with Cargo, edition 2024 support.
- NVIDIA CUDA Toolkit with `nvcc` available through `CUDA_HOME`, `CUDA_ROOT`, `CUDA_PATH`, or `PATH`.
- An SM90 CUDA GPU for the implemented smoke-tested kernels.
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
CUDA_COMPUTE_CAP=90 cargo test --workspace
```

Run the SM90 GPU smoke tests explicitly:

```bash
CUDA_COMPUTE_CAP=90 cargo test -p candle-flashmla -- --ignored --nocapture
```

Run only one smoke test:

```bash
CUDA_COMPUTE_CAP=90 cargo test -p candle-flashmla sparse_prefill_sm90_smoke -- --ignored --nocapture
CUDA_COMPUTE_CAP=90 cargo test -p candle-flashmla sparse_decode_sm90_smoke -- --ignored --nocapture
```

Use an external FlashMLA checkout instead of the vendored submodule:

```bash
FLASHMLA_ROOT=/path/to/FlashMLA CUDA_COMPUTE_CAP=90 cargo test --workspace
```

Useful build variables:

- `CUDA_COMPUTE_CAP=90`: selects the current SM90 build path.
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
| Sparse BF16 prefill | Done: C ABI, Rust wrapper, Candle API, smoke test | Planned: upstream sources identified, dispatch/build not wired | Unsupported |
| Sparse BF16-query / FP8-cache decode | Done: C ABI, Rust wrapper, Candle plan/run API, smoke test | Planned: upstream sources identified, dispatch/build not wired | Unsupported |
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
- SM100 support should be added by making the build source set and runtime C ABI dispatch
  architecture-aware before adding the SM100 kernel calls.
