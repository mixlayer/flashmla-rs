# FlashMLA Rust Bindings Plan

Date: 2026-06-26

## Goal

Create an external Rust workspace at `~/flashmla-rs` that provides FlashMLA bindings usable by Mixlayer/modeld without depending on libtorch or Python extension loading.

Primary modeld use case:

- GLM 5.2 sparse MLA execution on production SM90 and later SM100.
- SM120/121 development should continue to use FlashInfer sparse MLA because upstream FlashMLA does not support SM120/121.
- Initial validation should target single-layer output parity against vLLM/FlashMLA on an SM90 node.

## Proposed Workspace

```text
~/flashmla-rs/
  Cargo.toml
  crates/
    flashmla-sys/
      Cargo.toml
      build.rs
      csrc/
        flashmla_c_api.h
        flashmla_c_api.cu
      src/
        lib.rs
      vendor/
        FlashMLA/              # git submodule, or fallback to FLASHMLA_ROOT
    flashmla/
      Cargo.toml
      src/
        lib.rs
        error.rs
        arch.rs
        sparse_prefill.rs
        sparse_decode.rs
        workspace.rs
    candle-flashmla/
      Cargo.toml
      src/
        lib.rs
        error.rs
        tensor.rs
        sparse_prefill.rs
        sparse_decode.rs
        workspace.rs
```

The `flashmla` crate is optional but recommended. If time is tight, collapse its safe wrapper layer into `candle-flashmla` first, but keep `flashmla-sys` strictly raw FFI.

## Crate Responsibilities

### `flashmla-sys`

- Owns CUDA/C++ compilation and linking.
- Exposes a small stable C ABI.
- Does not depend on Candle, cudarc, libtorch, pybind11, or Python.
- Uses caller-owned input/output/workspace pointers.
- Returns status codes and error strings instead of throwing C++ exceptions across FFI.

### `flashmla`

- Provides safe Rust parameter structs over `flashmla-sys`.
- Encodes architecture, shape, dtype, and workspace validation that is independent of Candle.
- Can expose cudarc-oriented APIs later, but should not require Candle.

### `candle-flashmla`

- Owns Candle tensor validation and allocation.
- Extracts CUDA stream/device pointers from Candle tensors.
- Allocates outputs and workspaces.
- Implements vLLM-compatible padding/slicing behavior for unsupported head counts.
- Provides modeld-facing APIs for sparse prefill and sparse decode.

## Important Upstream Constraint

Do not bind FlashMLA's Python/PyTorch extension API directly.

Upstream `csrc/api/api.cpp` exposes pybind functions that allocate `at::Tensor`s internally. The upstream API headers also include libtorch via `csrc/api/common.h`. The Rust binding should instead add a separate C ABI wrapper that includes lower-level FlashMLA headers and launches kernels with raw pointers.

The C wrapper should include lower-level headers such as:

- `params.h`
- `sm90/prefill/sparse/phase1.h`
- `sm100/prefill/sparse/fwd/head64/phase1.h`
- `sm100/prefill/sparse/fwd/head128/phase1.h`
- `sm100/prefill/sparse/fwd_for_small_topk/head128/phase1.h`
- `sm90/decode/sparse_fp8/splitkv_mla.h`
- `sm100/decode/head64/kernel.h`
- `smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h`
- `smxx/decode/combine/combine.h`

Avoid including:

- `csrc/api/api.cpp`
- `csrc/api/common.h`
- `csrc/api/sparse_fwd.h`
- `csrc/api/sparse_decode.h`

Those are useful as reference implementations, but they pull in libtorch.

## Build Strategy

### Source Location

Support both:

- Vendored upstream as `crates/flashmla-sys/vendor/FlashMLA`.
- External checkout via `FLASHMLA_ROOT=/home/zackangelo/FlashMLA`.

Build script precedence:

1. `FLASHMLA_ROOT`, if set.
2. Vendored submodule path.
3. Fail with a clear message.

### Architecture Support

Initial supported targets:

- `sm_90a` for Hopper.
- `sm_100f` for Blackwell datacenter.

Explicitly reject:

- `sm_120` and `sm_121`.

Reason: upstream FlashMLA support matrix is SM90/SM100 only, and local kernels contain guards such as `__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200` for SM100 paths.

Build environment variables:

- `CUDA_COMPUTE_CAP=90` or `100` for explicit target selection.
- `FLASHMLA_ARCHS=sm90a,sm100f` for multi-arch builds later.
- `FLASHMLA_ROOT=/home/zackangelo/FlashMLA` for external source checkout.
- `FLASHMLA_BUILD_DIR=...` for cached build artifacts.
- `CANDLE_NVCC_CCBIN=...` for custom host compiler.
- `FLASHMLA_NO_CCACHE=1` to disable `ccache`.
- `NVCC_THREADS=...` to mirror upstream build parallelism.

### NVCC Flags

Mirror upstream FlashMLA flags:

- `-O3`
- `-std=c++20`
- `-DNDEBUG`
- `-D_USE_MATH_DEFINES`
- `-U__CUDA_NO_HALF_OPERATORS__`
- `-U__CUDA_NO_HALF_CONVERSIONS__`
- `-U__CUDA_NO_HALF2_OPERATORS__`
- `-U__CUDA_NO_BFLOAT16_CONVERSIONS__`
- `--expt-relaxed-constexpr`
- `--expt-extended-lambda`
- `--use_fast_math`
- `-lineinfo`
- `-Xcompiler=-fPIC`
- `-Xcompiler=-fvisibility=hidden`

Use `--ptxas-options=-v` behind an opt-in environment flag because it is noisy.

### Include Paths

Use:

- `${FLASHMLA_ROOT}/csrc`
- `${FLASHMLA_ROOT}/csrc/kerutils/include`
- `${FLASHMLA_ROOT}/csrc/sm90`
- `${FLASHMLA_ROOT}/csrc/cutlass/include`
- `${FLASHMLA_ROOT}/csrc/cutlass/tools/util/include`
- `${CUDA_ROOT}/include`

Validate that FlashMLA's Cutlass submodule exists before compiling.

### Source Sets

Common decode utilities:

- `csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu`
- `csrc/smxx/decode/combine/combine.cu`

SM90 sparse prefill:

- `csrc/sm90/prefill/sparse/fwd.cu`
- `csrc/sm90/prefill/sparse/instantiations/phase1_k512.cu`
- `csrc/sm90/prefill/sparse/instantiations/phase1_k512_topklen.cu`
- `csrc/sm90/prefill/sparse/instantiations/phase1_k576.cu`
- `csrc/sm90/prefill/sparse/instantiations/phase1_k576_topklen.cu`

SM90 sparse decode:

- `csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h64.cu`
- `csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h128.cu`
- `csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h64.cu`
- `csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h128.cu`

SM100 sparse prefill:

- `csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k512.cu`
- `csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k576.cu`
- `csrc/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k512.cu`
- `csrc/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k576.cu`
- `csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_prefill_k512.cu`

SM100 sparse decode:

- `csrc/sm100/decode/head64/instantiations/v32.cu`
- `csrc/sm100/decode/head64/instantiations/model1.cu`
- `csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512.cu`

Defer dense decode/prefill bindings until sparse GLM path is working.

## C ABI Design

### Status and Errors

Expose:

```c
typedef enum flashmla_status_t {
  FLASHMLA_STATUS_SUCCESS = 0,
  FLASHMLA_STATUS_INVALID_ARGUMENT = 1,
  FLASHMLA_STATUS_UNSUPPORTED_ARCH = 2,
  FLASHMLA_STATUS_CUDA_ERROR = 3,
  FLASHMLA_STATUS_INTERNAL_ERROR = 4,
} flashmla_status_t;

const char* flashmla_last_error(void);
```

Do not let C++ exceptions or `TORCH_CHECK` behavior cross FFI.

### Device Query

Expose:

```c
flashmla_status_t flashmla_get_device_info(
  int device_id,
  int* major,
  int* minor,
  int* num_sms
);
```

This lets Rust validate SM90/SM100 before launch and compute decode workspace sizing.

### Sparse Prefill

Expose a raw BF16 sparse prefill entry point:

```c
flashmla_status_t flashmla_sparse_prefill_bf16(
  const flashmla_sparse_prefill_params_t* params
);
```

Parameter struct should contain:

- `s_q`, `s_kv`, `h_q`, `h_kv`, `d_qk`, `d_v`, `topk`
- `sm_scale`
- raw pointers for `q`, `kv`, `indices`, optional `attn_sink`, optional `topk_length`
- strides for `q`, `kv`, and `indices`
- caller-owned output pointers for `out`, `max_logits`, and `lse`
- `num_sm`
- `cudaStream_t`

Validation:

- `q` and `kv` are BF16.
- `indices` and `topk_length` are I32.
- `attn_sink`, `max_logits`, and `lse` are F32.
- `d_qk` is `512` or `576`.
- `d_v` is `512`.
- `h_q` is `64` or `128` after padding.
- `h_kv` is `1` for MQA-style GLM/DeepSeek MLA.

Dispatch behavior:

- SM90: call `sm90::fwd::run_fwd_phase1_kernel`.
- SM100 with `h_q=64`: call `sm100::fwd::head64::run_fwd_phase1_kernel`.
- SM100 with `h_q=128` and small top-k: call `sm100::fwd_for_small_topk::head128::run_fwd_for_small_topk_phase1_kernel`.
- SM100 with `h_q=128` and regular top-k: call `sm100::fwd::head128::run_fwd_phase1_kernel`.

Mirror upstream selection logic, but implement it without `csrc/api/common.h`.

### Sparse Decode

Expose separate planning and run calls:

```c
flashmla_status_t flashmla_sparse_decode_plan(
  const flashmla_sparse_decode_plan_params_t* params,
  flashmla_sparse_decode_plan_result_t* result
);

flashmla_status_t flashmla_sparse_decode_bf16_fp8(
  const flashmla_sparse_decode_params_t* params
);
```

Planning should compute:

- `num_sm_parts`
- `fixed_overhead_num_blocks`
- `block_size_topk`
- scheduler metadata I32 length
- `num_splits` length
- `lse_accum` element count
- `o_accum` element count

Run should consume caller-owned:

- output `out`
- output `lse`
- scheduler metadata
- `num_splits`
- `lse_accum`
- `o_accum`

Run should perform:

1. Optional schedule metadata generation, or require precomputed metadata.
2. Sparse decode kernel launch.
3. Combine kernel launch.

Validation:

- `q` is BF16 with shape `[batch, s_q, h_q, d_qk]`.
- packed KV cache is FP8/u8-compatible with FlashMLA's expected per-token byte layout.
- `indices` is I32 with shape `[batch, s_q, topk]`.
- `d_qk` is `512` or `576`.
- `d_v` is `512`.
- `h_q` is `64` or `128` after padding.
- `h_kv` is `1`.

For GLM/vLLM compatibility, expose `d_v=512` even if model attention output is later projected/sliced differently. vLLM's FlashMLA sparse path hardcodes `head_dim_v=512` and returns latent output shaped by `kv_lora_rank`.

## Candle Integration

### Tensor Helpers

`candle-flashmla/src/tensor.rs` should mirror the style used by `modeld-flashinfer`:

- `ensure_rank`
- `ensure_contiguous` or `ensure_last_dim_contiguous`
- `ensure_same_device`
- CUDA stream/device extraction
- typed pointer extraction for BF16, F32, I32, U8/F8E4M3
- stride conversion with overflow checks

### Sparse Prefill API

Suggested high-level API:

```rust
pub struct SparsePrefillConfig {
    pub softmax_scale: f32,
    pub d_v: usize,
    pub pad_heads: bool,
}

pub fn sparse_prefill(
    q: &Tensor,
    kv: &Tensor,
    indices: &Tensor,
    topk_length: Option<&Tensor>,
    attn_sink: Option<&Tensor>,
    config: SparsePrefillConfig,
) -> Result<SparsePrefillOutput>;
```

Output:

- `out: Tensor`
- `max_logits: Tensor`
- `lse: Tensor`

Padding:

- On SM90, pad `h_q` to `64` when needed.
- On SM100, pad `h_q` to `128` when needed.
- Slice output back to the original number of heads after launch.

### Sparse Decode API

Suggested high-level API:

```rust
pub struct SparseDecodeConfig {
    pub softmax_scale: f32,
    pub d_v: usize,
    pub pad_heads: bool,
}

pub struct SparseDecodePlan {
    pub scheduler_metadata: Tensor,
    pub num_splits: Tensor,
    pub lse_accum: Tensor,
    pub o_accum: Tensor,
    pub meta: SparseDecodePlanMeta,
}

pub fn sparse_decode_plan(...) -> Result<SparseDecodePlan>;

pub fn sparse_decode(
    q: &Tensor,
    kv_cache: &Tensor,
    indices: &Tensor,
    topk_length: Option<&Tensor>,
    attn_sink: Option<&Tensor>,
    extra_kv_cache: Option<&Tensor>,
    extra_indices: Option<&Tensor>,
    extra_topk_length: Option<&Tensor>,
    plan: &mut SparseDecodePlan,
    config: SparseDecodeConfig,
) -> Result<SparseDecodeOutput>;
```

Output:

- `out: Tensor`
- `lse: Tensor`

## SM90 Bring-Up Plan

### Phase 1: Workspace Scaffold

Create the Cargo workspace and crates:

- root `Cargo.toml`
- `crates/flashmla-sys`
- `crates/flashmla`
- `crates/candle-flashmla`

Do not wire into modeld yet.

### Phase 2: `flashmla-sys` Build

Tasks:

1. Add `FLASHMLA_ROOT` source discovery.
2. Validate Cutlass submodule paths.
3. Add SM90-only source list first.
4. Add static library build via `nvcc` plus `ar`, similar to `modeld-kernels/build.rs`.
5. Emit link directives for `cudart`, `cuda`, and `stdc++`.
6. Add `cargo:rerun-if-changed` for wrapper files and selected upstream source roots.

Initial SM90 command:

```bash
cd ~/flashmla-rs
FLASHMLA_ROOT=/home/zackangelo/FlashMLA CUDA_COMPUTE_CAP=90 cargo check -p flashmla-sys
```

### Phase 3: Raw Sparse Prefill FFI

Tasks:

1. Define `flashmla_c_api.h` status enum and prefill params.
2. Implement `flashmla_sparse_prefill_bf16`.
3. Reimplement SM90 prefill dispatch without libtorch.
4. Add a Rust `extern "C"` declaration in `flashmla-sys/src/lib.rs`.
5. Add a tiny build-only test that links the symbol.

SM90 validation:

```bash
FLASHMLA_ROOT=/home/zackangelo/FlashMLA CUDA_COMPUTE_CAP=90 cargo test -p flashmla-sys -- --nocapture
```

### Phase 4: Candle Sparse Prefill Smoke Test

Tasks:

1. Add `candle-flashmla` tensor pointer helpers.
2. Allocate Candle CUDA tensors for a small synthetic prefill case.
3. Launch sparse prefill.
4. Compare output against upstream Python FlashMLA or vLLM on the same SM90 node.

Start with small shapes:

- `s_q=16`
- `s_kv=128`
- `h_q=64`
- `h_kv=1`
- `d_qk=576`
- `d_v=512`
- `topk=32` or `64`

Only increase toward GLM values after basic correctness.

### Phase 5: Raw Sparse Decode FFI

Tasks:

1. Define decode plan/result structs.
2. Reimplement decode metadata selection from upstream `sparse_decode.h`.
3. Add C ABI wrapper for schedule metadata generation.
4. Add C ABI wrapper for sparse decode + combine.
5. Add Rust workspace allocation helpers.

Decode is more complex than prefill because upstream Python allocates scheduler metadata, split-KV accumulators, and output tensors internally.

### Phase 6: GLM Single-Layer Parity Harness

Goal: validate one GLM layer against vLLM without needing a full-model GPU.

Required artifacts from vLLM:

- Layer input hidden states.
- Q/latent projections after GLM attention projections.
- Concatenated query used by FlashMLA.
- Packed FP8 KV cache or BF16 prefill workspace.
- Sparse DSA/top-k indices.
- Optional `topk_length`.
- FlashMLA output before the output projection.
- Final attention output after output projection if available.

Harness flow:

1. Run vLLM on SM90 for one layer and save tensors.
2. Load tensors in Rust/Candle.
3. Run `candle-flashmla` sparse prefill or decode.
4. Compare FlashMLA latent output and LSE.
5. Then compare post-attention output once modeld GLM projection plumbing exists.

Suggested tolerances:

- BF16 prefill: start with `rtol=1e-2`, `atol=1e-2`.
- FP8 decode: expect looser tolerance; measure vLLM observed error first.

### Phase 7: SM100 Enablement

After SM90 is working:

1. Add SM100 source list.
2. Require CUDA 12.9+ for SM100 compilation.
3. Add `sm_100f` gencode.
4. Validate SM100 prefill and decode on a B200/GB200 node.
5. Keep SM90 and SM100 artifacts separately cacheable.

## Known Gaps and Risks

- FlashMLA does not support SM120/121, so local desktop development must use FlashInfer sparse MLA or CPU/reference fallbacks.
- FlashMLA consumes sparse indices; it does not implement GLM DSA index generation.
- The initial C ABI must avoid upstream libtorch-dependent headers.
- Decode workspace sizing must match upstream exactly or split-KV combine will be wrong.
- GLM/vLLM pads head counts for FlashMLA kernels; Candle wrapper must reproduce that behavior.
- Packed FP8 KV cache layout must match FlashMLA `MODEL1`/`V32` expectations exactly.
- We need a clear license review before vendoring FlashMLA into a reusable crate.

## Definition of Done

SM90 milestone:

- `flashmla-sys` builds on SM90 without libtorch.
- `flashmla_sparse_prefill_bf16` links and runs.
- `candle-flashmla` can run sparse prefill on synthetic tensors.
- Synthetic output parity is measured against upstream FlashMLA.
- GLM single-layer prefill or decode parity is measured against vLLM.

Production-ready milestone:

- Sparse prefill and sparse decode are both bound.
- Workspace planning is explicit and reusable.
- SM90 and SM100 build paths are validated.
- modeld can choose FlashMLA for SM90/SM100 and FlashInfer for SM120/121.

