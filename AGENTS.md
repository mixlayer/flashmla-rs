# Repository Guidelines

## Scope

These instructions apply to the entire `flashmla-rs` workspace.

## Project Shape

- Keep the root manifest as a Cargo workspace.
- Keep `flashmla-sys` as the raw FFI and CUDA build crate.
- Keep `flashmla` as the Candle-independent safe wrapper crate.
- Keep `candle-flashmla` as the Candle integration crate.
- Do not bind FlashMLA's Python, PyTorch, or libtorch-facing APIs.
- Prefer the vendored FlashMLA submodule by default, while preserving `FLASHMLA_ROOT` override support.

## Rust API Rules

- Any public Rust struct or function must include a `///` doc block.
- Public FFI declarations should document ownership, pointer validity, dtype, shape, stride, stream, and synchronization expectations.
- Keep raw `unsafe extern "C"` bindings in `flashmla-sys`; higher-level validation belongs in `flashmla` or `candle-flashmla`.
- Preserve C-compatible layouts with `#[repr(C)]` for exported or mirrored ABI structs.

## CUDA and FFI Rules

- C ABI entry points must return `flashmla_status_t`; do not let C++ exceptions cross FFI.
- Use `flashmla_last_error()` for human-readable error details after non-success status returns.
- Keep `flashmla-sys` free of Candle, cudarc, Python, PyTorch, pybind11, and libtorch dependencies.
- Validate architecture and shapes before launching kernels where practical.
- SM120 and SM121 are unsupported by upstream FlashMLA and should be rejected explicitly.

## Build and Test

- Format with `cargo fmt --all`.
- For CUDA build checks, use `CUDA_COMPUTE_CAP=90 cargo check -p flashmla-sys`.
- For targeted FFI checks, use `CUDA_COMPUTE_CAP=90 cargo test -p flashmla-sys`.
- For full workspace validation, use `CUDA_COMPUTE_CAP=90 cargo test --workspace`.

## Editing Discipline

- Keep changes narrowly scoped to the current phase in `PLAN.md`.
- Do not revert unrelated user changes.
- Prefer existing workspace patterns over introducing new abstractions.
