use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

const COMMON_DECODE_SOURCES: &[&str] = &[
    "csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
    "csrc/smxx/decode/combine/combine.cu",
];

const SM90_SOURCES: &[&str] = &[
    "csrc/sm90/prefill/sparse/fwd.cu",
    "csrc/sm90/prefill/sparse/instantiations/phase1_k512.cu",
    "csrc/sm90/prefill/sparse/instantiations/phase1_k512_topklen.cu",
    "csrc/sm90/prefill/sparse/instantiations/phase1_k576.cu",
    "csrc/sm90/prefill/sparse/instantiations/phase1_k576_topklen.cu",
    "csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h64.cu",
    "csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h128.cu",
    "csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h64.cu",
    "csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h128.cu",
];

const SM100_SOURCES: &[&str] = &[
    "csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k512.cu",
    "csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k576.cu",
    "csrc/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k512.cu",
    "csrc/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k576.cu",
    "csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_prefill_k512.cu",
    "csrc/sm100/decode/head64/instantiations/v32.cu",
    "csrc/sm100/decode/head64/instantiations/model1.cu",
    "csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512.cu",
];

fn main() {
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );

    println!("cargo:rerun-if-env-changed=FLASHMLA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=FLASHMLA_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=FLASHMLA_ARCHS");
    println!("cargo:rerun-if-env-changed=CANDLE_NVCC_CCBIN");
    println!("cargo:rerun-if-env-changed=FLASHMLA_NO_CCACHE");
    println!("cargo:rerun-if-env-changed=NVCC_THREADS");
    println!("cargo:rerun-if-env-changed=FLASHMLA_PTXAS_VERBOSE");
    println!("cargo:rerun-if-changed=csrc/flashmla_c_api.h");
    println!("cargo:rerun-if-changed=csrc/flashmla_c_api.cu");

    let arch = match selected_arch() {
        Ok(arch) => arch,
        Err(warning) => {
            let flashmla_root = configured_flashmla_root(&manifest_dir);
            println!("cargo:warning={warning}; enabling the unsupported_arch feature");
            println!("cargo:rustc-cfg=feature=\"unsupported_arch\"");
            println!(
                "cargo:rustc-env=FLASHMLA_SOURCE_ROOT={}",
                flashmla_root.display()
            );
            println!("cargo:metadata=source_root={}", flashmla_root.display());
            return;
        }
    };

    let flashmla_root = discover_flashmla_root(&manifest_dir);
    let cuda_root = discover_cuda_root();
    let nvcc = cuda_root.join("bin").join("nvcc");
    let ar = find_program("ar").unwrap_or_else(|| PathBuf::from("ar"));

    for source in COMMON_DECODE_SOURCES.iter().chain(arch.sources()) {
        println!(
            "cargo:rerun-if-changed={}",
            flashmla_root.join(source).display()
        );
    }
    println!(
        "cargo:rustc-env=FLASHMLA_SOURCE_ROOT={}",
        flashmla_root.display()
    );
    println!("cargo:metadata=source_root={}", flashmla_root.display());

    let sources = selected_sources(&manifest_dir, &flashmla_root, arch);
    for source in &sources {
        require_file(source);
    }

    let build_dir = build_dir(&arch);
    fs::create_dir_all(&build_dir).unwrap_or_else(|error| {
        panic!(
            "failed to create FlashMLA build directory {}: {error}",
            build_dir.display()
        )
    });

    let include_dirs = include_dirs(&manifest_dir, &flashmla_root, &cuda_root, &arch);
    let mut objects = Vec::with_capacity(sources.len());
    for (index, source) in sources.iter().enumerate() {
        let object = build_dir.join(format!("{index:02}_{}.o", object_stem(source)));
        compile_cuda_source(&nvcc, source, &object, &include_dirs, &arch);
        objects.push(object);
    }

    let archive = build_dir.join("libflashmla.a");
    archive_objects(&ar, &archive, &objects);

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=flashmla");
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_root.join("lib64").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_root.join("lib64").join("stubs").display()
    );
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

fn configured_flashmla_root(manifest_dir: &Path) -> PathBuf {
    match env::var_os("FLASHMLA_ROOT") {
        Some(root) if root.is_empty() => panic!("FLASHMLA_ROOT is set but empty"),
        Some(root) => PathBuf::from(root),
        None => manifest_dir.join("vendor").join("FlashMLA"),
    }
}

fn discover_flashmla_root(manifest_dir: &Path) -> PathBuf {
    let source = if env::var_os("FLASHMLA_ROOT").is_some() {
        "FLASHMLA_ROOT"
    } else {
        "vendored FlashMLA submodule"
    };
    validate_flashmla_root(configured_flashmla_root(manifest_dir), source)
}

fn validate_flashmla_root(path: PathBuf, source: &str) -> PathBuf {
    let root = path.canonicalize().unwrap_or_else(|error| {
        panic!(
            "{source} does not point to a readable FlashMLA checkout at {}: {error}",
            path.display()
        )
    });

    require_path(&root, "csrc/params.h", source);
    require_path(&root, "csrc/kerutils/include", source);
    require_path(&root, "csrc/cutlass/include", source);
    require_path(
        &root,
        "csrc/cutlass/tools/util/include",
        "FlashMLA CUTLASS utilities",
    );

    root
}

fn require_path(root: &Path, relative: &str, source: &str) {
    let path = root.join(relative);
    if path.exists() {
        return;
    }

    if relative.starts_with("csrc/cutlass") {
        panic!(
            "{source} FlashMLA checkout is missing {relative}. Initialize nested submodules with `git submodule update --init --recursive crates/flashmla-sys/vendor/FlashMLA`."
        );
    }

    panic!(
        "{source} does not look like a FlashMLA checkout: missing {}",
        path.display()
    );
}

fn discover_cuda_root() -> PathBuf {
    for var in ["CUDA_HOME", "CUDA_ROOT", "CUDA_PATH"] {
        if let Some(root) = env::var_os(var) {
            if root.is_empty() {
                panic!("{var} is set but empty");
            }
            return validate_cuda_root(PathBuf::from(root), var);
        }
    }

    let nvcc = find_program("nvcc").unwrap_or_else(|| {
        panic!(
            "could not find nvcc. Set CUDA_HOME, CUDA_ROOT, or CUDA_PATH to a CUDA Toolkit install."
        )
    });
    let cuda_root = nvcc.parent().and_then(Path::parent).unwrap_or_else(|| {
        panic!(
            "could not derive CUDA root from nvcc path {}",
            nvcc.display()
        )
    });
    validate_cuda_root(cuda_root.to_path_buf(), "nvcc")
}

fn validate_cuda_root(path: PathBuf, source: &str) -> PathBuf {
    let root = path.canonicalize().unwrap_or_else(|error| {
        panic!(
            "{source} does not point to a readable CUDA Toolkit at {}: {error}",
            path.display()
        )
    });
    require_file(&root.join("bin").join("nvcc"));
    require_file(&root.join("include").join("cuda_runtime_api.h"));
    root
}

fn selected_arch() -> Result<Arch, String> {
    if let Some(archs) = env::var_os("FLASHMLA_ARCHS") {
        let archs = archs.to_string_lossy();
        let parsed: Vec<_> = archs
            .split(',')
            .map(str::trim)
            .filter(|arch| !arch.is_empty())
            .collect();
        if parsed.len() != 1 {
            panic!(
                "FLASHMLA_ARCHS={archs:?} is not supported yet. Use exactly one of sm90a or sm100f."
            );
        }
        return parse_arch(parsed[0], "FLASHMLA_ARCHS");
    }

    match env::var("CUDA_COMPUTE_CAP") {
        Ok(value) if !value.trim().is_empty() => parse_arch(value.trim(), "CUDA_COMPUTE_CAP"),
        Ok(_) => panic!("CUDA_COMPUTE_CAP is set but empty"),
        Err(env::VarError::NotPresent) => detect_arch_with_nvidia_smi(),
        Err(env::VarError::NotUnicode(_)) => panic!("CUDA_COMPUTE_CAP is not valid Unicode"),
    }
}

fn detect_arch_with_nvidia_smi() -> Result<Arch, String> {
    let Some(nvidia_smi) = find_program("nvidia-smi") else {
        return Err(
            "CUDA architecture auto-detection requires `nvidia-smi`, but it was not found in PATH. Set CUDA_COMPUTE_CAP=90/100 or FLASHMLA_ARCHS=sm90a/sm100f explicitly"
                .to_string(),
        );
    };
    let output = Command::new(&nvidia_smi)
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .map_err(|error| {
            format!(
                "failed to run {} for CUDA architecture auto-detection: {error}. Set CUDA_COMPUTE_CAP=90/100 or FLASHMLA_ARCHS=sm90a/sm100f explicitly",
                nvidia_smi.display()
            )
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "{} failed during CUDA architecture auto-detection with status {}: {}. Set CUDA_COMPUTE_CAP=90/100 or FLASHMLA_ARCHS=sm90a/sm100f explicitly",
            nvidia_smi.display(),
            output.status,
            stderr.trim()
        ));
    }
    let stdout = String::from_utf8(output.stdout).map_err(|error| {
        format!(
            "{} returned non-UTF-8 compute capability output: {error}. Set CUDA_COMPUTE_CAP=90/100 or FLASHMLA_ARCHS=sm90a/sm100f explicitly",
            nvidia_smi.display()
        )
    })?;
    parse_detected_archs(&stdout)
}

fn parse_detected_archs(output: &str) -> Result<Arch, String> {
    let mut capabilities = output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let Some(first_capability) = capabilities.next() else {
        return Err(
            "nvidia-smi reported no visible GPUs during CUDA architecture auto-detection. Set CUDA_COMPUTE_CAP=90/100 or FLASHMLA_ARCHS=sm90a/sm100f explicitly"
                .to_string(),
        );
    };
    let selected = parse_arch(first_capability, "nvidia-smi compute capability")?;
    for capability in capabilities {
        let arch = parse_arch(capability, "nvidia-smi compute capability")?;
        if arch != selected {
            return Err(format!(
                "nvidia-smi reported mixed GPU architectures ({first_capability} and {capability}), but flashmla-rs builds exactly one architecture. Set CUDA_COMPUTE_CAP=90/100 or FLASHMLA_ARCHS=sm90a/sm100f explicitly"
            ));
        }
    }
    println!(
        "cargo:warning=auto-detected FlashMLA target {} from nvidia-smi",
        selected.name()
    );
    Ok(selected)
}

fn parse_arch(value: &str, source: &str) -> Result<Arch, String> {
    let normalized: String = value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|ch| *ch != '_' && *ch != '.')
        .collect();

    match normalized.as_str() {
        "90" | "90a" | "sm90" | "sm90a" | "compute90a" => Ok(Arch::Sm90a),
        "100" | "100f" | "sm100" | "sm100f" | "compute100f" => Ok(Arch::Sm100f),
        "120" | "121" | "sm120" | "sm121" => Err(format!(
            "{source}={value:?} is unsupported by upstream FlashMLA sparse MLA"
        )),
        _ => Err(format!(
            "{source}={value:?} is unsupported. Use CUDA_COMPUTE_CAP=90/100 or FLASHMLA_ARCHS=sm90a/sm100f"
        )),
    }
}

fn selected_sources(manifest_dir: &Path, flashmla_root: &Path, arch: Arch) -> Vec<PathBuf> {
    let mut sources = Vec::with_capacity(COMMON_DECODE_SOURCES.len() + arch.sources().len() + 1);
    sources.push(manifest_dir.join("csrc").join("flashmla_c_api.cu"));
    sources.extend(
        COMMON_DECODE_SOURCES
            .iter()
            .chain(arch.sources())
            .map(|source| flashmla_root.join(source)),
    );
    sources
}

fn include_dirs(
    manifest_dir: &Path,
    flashmla_root: &Path,
    cuda_root: &Path,
    arch: &Arch,
) -> Vec<PathBuf> {
    vec![
        manifest_dir.join("csrc"),
        flashmla_root.join("csrc"),
        flashmla_root.join("csrc").join("kerutils").join("include"),
        flashmla_root.join("csrc").join(arch.source_dir()),
        flashmla_root.join("csrc").join("cutlass").join("include"),
        flashmla_root
            .join("csrc")
            .join("cutlass")
            .join("tools")
            .join("util")
            .join("include"),
        cuda_root.join("include"),
    ]
}

fn build_dir(arch: &Arch) -> PathBuf {
    match env::var_os("FLASHMLA_BUILD_DIR") {
        Some(path) if path.is_empty() => panic!("FLASHMLA_BUILD_DIR is set but empty"),
        Some(path) => PathBuf::from(path).join(arch.name()),
        None => PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set by Cargo"))
            .join("flashmla-build")
            .join(arch.name()),
    }
}

fn compile_cuda_source(
    nvcc: &Path,
    source: &Path,
    object: &Path,
    include_dirs: &[PathBuf],
    arch: &Arch,
) {
    let mut command = nvcc_command(nvcc);
    command.arg("-c").arg(source).arg("-o").arg(object);
    command.args([
        "-O3",
        "-std=c++20",
        "-DNDEBUG",
        "-D_USE_MATH_DEFINES",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-lineinfo",
        "-Xcompiler=-fPIC",
        "-Xcompiler=-fvisibility=hidden",
    ]);
    command.arg("-gencode").arg(arch.gencode());
    command.arg(arch.target_define());
    if env_truthy("FLASHMLA_PTXAS_VERBOSE") {
        command.arg("--ptxas-options=-v");
    }
    if let Some(ccbin) = env::var_os("CANDLE_NVCC_CCBIN") {
        if ccbin.is_empty() {
            panic!("CANDLE_NVCC_CCBIN is set but empty");
        }
        command.arg("-ccbin").arg(ccbin);
    }
    if let Some(threads) = env::var_os("NVCC_THREADS") {
        if threads.is_empty() {
            panic!("NVCC_THREADS is set but empty");
        }
        command.arg("--threads").arg(threads);
    }
    for include_dir in include_dirs {
        command.arg("-I").arg(include_dir);
    }

    run_command(command, "nvcc", source);
}

fn nvcc_command(nvcc: &Path) -> Command {
    if env_truthy("FLASHMLA_NO_CCACHE") {
        return Command::new(nvcc);
    }

    if let Some(ccache) = find_program("ccache") {
        let mut command = Command::new(ccache);
        command.arg(nvcc);
        command
    } else {
        Command::new(nvcc)
    }
}

fn archive_objects(ar: &Path, archive: &Path, objects: &[PathBuf]) {
    if archive.exists() {
        fs::remove_file(archive).unwrap_or_else(|error| {
            panic!(
                "failed to remove stale archive {}: {error}",
                archive.display()
            )
        });
    }

    let mut command = Command::new(ar);
    command.arg("crs").arg(archive);
    command.args(objects);
    run_command(command, "ar", archive);
}

fn run_command(mut command: Command, program: &str, input: &Path) {
    let status = command.status().unwrap_or_else(|error| {
        panic!(
            "failed to run {program} while building {}: {error}",
            input.display()
        )
    });
    if !status.success() {
        panic!(
            "{program} failed while building {} with status {status}",
            input.display()
        );
    }
}

fn require_file(path: &Path) {
    if !path.is_file() {
        panic!("required file is missing: {}", path.display());
    }
}

fn find_program(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

fn object_stem(source: &Path) -> String {
    let name = source
        .file_name()
        .unwrap_or_else(|| panic!("source path has no file name: {}", source.display()))
        .to_string_lossy();
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn env_truthy(name: &str) -> bool {
    env::var_os(name).is_some_and(is_truthy)
}

fn is_truthy(value: OsString) -> bool {
    matches!(
        value.to_string_lossy().to_ascii_lowercase().as_str(),
        "1" | "true" | "y" | "yes" | "on"
    )
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Arch {
    Sm90a,
    Sm100f,
}

#[cfg(test)]
mod tests {
    use super::{Arch, parse_detected_archs};

    #[test]
    fn detects_homogeneous_sm100_gpus() {
        assert_eq!(parse_detected_archs("10.0\n10.0\n").unwrap(), Arch::Sm100f);
    }

    #[test]
    fn detects_homogeneous_sm90_gpus() {
        assert_eq!(parse_detected_archs("9.0\n9.0\n").unwrap(), Arch::Sm90a);
    }

    #[test]
    fn rejects_mixed_gpu_architectures() {
        let error = parse_detected_archs("9.0\n10.0\n").unwrap_err();
        assert!(error.contains("mixed GPU architectures"));
    }

    #[test]
    fn rejects_empty_gpu_query() {
        let error = parse_detected_archs("\n").unwrap_err();
        assert!(error.contains("no visible GPUs"));
    }
}

impl Arch {
    fn name(self) -> &'static str {
        match self {
            Self::Sm90a => "sm90a",
            Self::Sm100f => "sm100f",
        }
    }

    fn source_dir(self) -> &'static str {
        match self {
            Self::Sm90a => "sm90",
            Self::Sm100f => "sm100",
        }
    }

    fn sources(self) -> &'static [&'static str] {
        match self {
            Self::Sm90a => SM90_SOURCES,
            Self::Sm100f => SM100_SOURCES,
        }
    }

    fn gencode(self) -> &'static str {
        match self {
            Self::Sm90a => "arch=compute_90a,code=sm_90a",
            Self::Sm100f => "arch=compute_100f,code=sm_100f",
        }
    }

    fn target_define(self) -> &'static str {
        match self {
            Self::Sm90a => "-DFLASHMLA_TARGET_SM90=1",
            Self::Sm100f => "-DFLASHMLA_TARGET_SM100=1",
        }
    }
}
