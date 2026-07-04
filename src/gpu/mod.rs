mod cuda;
mod hip;

use std::env;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context as _, Result, anyhow, bail};
use libloading::Library;

use crate::fingerprint::MAX_PREFIXES;
use crate::{HexPrefix, HexPrefixSet};

const BENCHMARK_PREFIX_HEX: &str = "FEDCBA9876543210";
const BENCHMARK_PUBLIC_KEY: [u8; 32] = [0x42; 32];
const BENCHMARK_START_TIMESTAMP: u32 = 0x1357_9BDF;
const BATCH_ITERATIONS: u64 = 512;
const KERNEL_NAME: &str = "search_v4_ed25519";
const FULL_BATCH_KERNEL_NAME: &str = "search_v4_ed25519_full";
const KERNEL_SOURCE: &str = include_str!("../gpu_kernel.cu");
const BLOCK_SIZE_CANDIDATES: [u32; 6] = [32, 64, 128, 256, 512, 1024];
// The kernel currently evaluates four timestamps per inner iteration, so
// larger poll intervals would skip work instead of just reducing polling.
const POLL_INTERVAL_CANDIDATES: [u32; 1] = [4];
const GLOBAL_WORK_SIZE_CANDIDATES: [u32; 5] = [1 << 19, 1 << 20, 1 << 21, 1 << 22, 1 << 23];
const FINALIST_COUNT: usize = 2;
const SUSTAINED_BENCHMARK_DURATION: Duration = Duration::from_secs(5);
const NOT_FOUND_TIMESTAMP: u32 = u32::MAX;
const PREFIX_WORDS: usize = 4;
const PREFIXES_BUFFER_BYTES: usize = MAX_PREFIXES * PREFIX_WORDS * std::mem::size_of::<u32>();

pub fn gpu_search_available() -> bool {
    GpuSearchEngine::new().is_ok()
}

pub(crate) enum GpuSearchEngine {
    Hip(hip::HipSearchEngine),
    Cuda(cuda::CudaSearchEngine),
}

impl GpuSearchEngine {
    pub(crate) fn new() -> Result<Self> {
        match backend_override()? {
            Some(Backend::Hip) => return hip::HipSearchEngine::new().map(Self::Hip),
            Some(Backend::Cuda) => return cuda::CudaSearchEngine::new().map(Self::Cuda),
            None => {}
        }

        let hip_error = match hip::HipSearchEngine::new() {
            Ok(engine) => return Ok(Self::Hip(engine)),
            Err(error) => error,
        };
        let cuda_error = match cuda::CudaSearchEngine::new() {
            Ok(engine) => return Ok(Self::Cuda(engine)),
            Err(error) => error,
        };

        Err(anyhow!(
            "no usable GPU backend found\n  HIP: {hip_error:#}\n  CUDA: {cuda_error:#}"
        ))
    }

    pub(crate) fn backend_name(&self) -> &'static str {
        match self {
            Self::Hip(_) => "HIP",
            Self::Cuda(_) => "CUDA",
        }
    }

    pub(crate) fn device_name(&self) -> &str {
        match self {
            Self::Hip(engine) => engine.device_name(),
            Self::Cuda(engine) => engine.device_name(),
        }
    }

    pub(crate) fn batch_size(&self) -> u64 {
        match self {
            Self::Hip(engine) => engine.batch_size(),
            Self::Cuda(engine) => engine.batch_size(),
        }
    }

    pub(crate) fn prepare_search(&mut self, base_words: &[u32; 16]) -> Result<()> {
        match self {
            Self::Hip(engine) => engine.prepare_search(base_words),
            Self::Cuda(engine) => engine.prepare_search(base_words),
        }
    }

    pub(crate) fn prepare_prefixes(&mut self, prefixes: &HexPrefixSet) -> Result<()> {
        match self {
            Self::Hip(engine) => engine.prepare_prefixes(prefixes),
            Self::Cuda(engine) => engine.prepare_prefixes(prefixes),
        }
    }

    pub(crate) fn search_batch(&mut self, start_timestamp: u32, count: u32) -> Result<Option<u32>> {
        match self {
            Self::Hip(engine) => engine.search_batch(start_timestamp, count),
            Self::Cuda(engine) => engine.search_batch(start_timestamp, count),
        }
    }
}

enum Backend {
    Hip,
    Cuda,
}

fn backend_override() -> Result<Option<Backend>> {
    let Some(value) = env::var_os("PGP_VANITY_GPU_BACKEND") else {
        return Ok(None);
    };

    match value.to_string_lossy().to_ascii_lowercase().as_str() {
        "hip" => Ok(Some(Backend::Hip)),
        "cuda" => Ok(Some(Backend::Cuda)),
        other => bail!(
            "unrecognized PGP_VANITY_GPU_BACKEND value {other:?}; expected \"hip\" or \"cuda\""
        ),
    }
}

fn benchmark_prefix() -> HexPrefix {
    HexPrefix::parse(BENCHMARK_PREFIX_HEX).expect("benchmark prefix must be valid")
}

fn push_finalist<Variant>(
    finalists: &mut Vec<(Variant, f64)>,
    variant: Variant,
    hashes_per_second: f64,
) {
    let insert_at = finalists
        .iter()
        .position(|(_, best_hashes_per_second)| hashes_per_second > *best_hashes_per_second)
        .unwrap_or(finalists.len());
    finalists.insert(insert_at, (variant, hashes_per_second));
    finalists.truncate(FINALIST_COUNT);
}

fn tuning_cache_path(file_name: &str) -> PathBuf {
    let base = env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| Path::new(&home).join(".cache")))
        .or_else(|| env::var_os("LOCALAPPDATA").map(PathBuf::from))
        .unwrap_or_else(env::temp_dir);
    base.join("pgp-vanity").join(file_name)
}

unsafe fn load_symbol<T: Copy>(library: &Library, symbol: &[u8]) -> Result<T> {
    Ok(*unsafe { library.get::<T>(symbol) }.with_context(|| {
        format!(
            "missing required symbol {}",
            String::from_utf8_lossy(symbol)
        )
    })?)
}

fn round_up_u32(value: u32, multiple: u32) -> u32 {
    if multiple == 0 {
        return value;
    }

    let remainder = value % multiple;
    if remainder == 0 {
        value
    } else {
        value + (multiple - remainder)
    }
}
