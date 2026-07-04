use std::env;
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as AnyhowContext, Result, anyhow, bail, ensure};
use libloading::Library;

use super::{
    BATCH_ITERATIONS, BENCHMARK_PUBLIC_KEY, BENCHMARK_START_TIMESTAMP, BLOCK_SIZE_CANDIDATES,
    FINALIST_COUNT, FULL_BATCH_KERNEL_NAME, GLOBAL_WORK_SIZE_CANDIDATES, KERNEL_NAME,
    KERNEL_SOURCE, NOT_FOUND_TIMESTAMP, POLL_INTERVAL_CANDIDATES, PREFIX_WORDS,
    PREFIXES_BUFFER_BYTES, SUSTAINED_BENCHMARK_DURATION, benchmark_prefix, load_symbol,
    push_finalist, round_up_u32, tuning_cache_path,
};
use crate::HexPrefixSet;
use crate::fingerprint::{FingerprintSearch, MAX_PREFIXES};

// `0` means "no explicit cap"; positive values pass `--maxrregcount=N` to
// NVRTC. The unrolled SHA-1 kernel benefits from a high register budget on
// architectures with a small per-thread register file (Pascal in particular),
// where the default ptxas heuristic spills heavily.
const MAXRREGCOUNT_CANDIDATES: [u32; 4] = [0, 96, 128, 255];
const TUNING_CACHE_VERSION: u32 = 3;
const TUNING_CACHE_FILE: &str = "cuda-tuning.txt";

type CuResult = c_int;
type NvrtcResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuStream = *mut c_void;
type CuDevicePtr = u64;
type NvrtcProgram = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;
const NVRTC_SUCCESS: NvrtcResult = 0;

// CUdevice_attribute values from cuda.h.
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

type SharedCudaApi = Arc<CudaApi>;

struct CudaApi {
    _cuda_library: Library,
    _nvrtc_library: Library,
    cu_init: unsafe extern "C" fn(c_uint) -> CuResult,
    cu_device_get_count: unsafe extern "C" fn(*mut c_int) -> CuResult,
    cu_device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
    cu_device_get_name: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
    cu_device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult,
    cu_ctx_create: unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> CuResult,
    cu_ctx_destroy: unsafe extern "C" fn(CuContext) -> CuResult,
    cu_mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult,
    cu_mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
    cu_memcpy_htod: unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult,
    cu_module_load_data: unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult,
    cu_module_unload: unsafe extern "C" fn(CuModule) -> CuResult,
    cu_module_get_function:
        unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
    cu_launch_kernel: unsafe extern "C" fn(
        CuFunction,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        CuStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    cu_ctx_synchronize: unsafe extern "C" fn() -> CuResult,
    cu_get_error_string: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
    nvrtc_create_program: unsafe extern "C" fn(
        *mut NvrtcProgram,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> NvrtcResult,
    nvrtc_compile_program:
        unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult,
    nvrtc_destroy_program: unsafe extern "C" fn(*mut NvrtcProgram) -> NvrtcResult,
    nvrtc_get_program_log_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    nvrtc_get_program_log: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    nvrtc_get_ptx_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    nvrtc_get_ptx: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    nvrtc_get_error_string: unsafe extern "C" fn(NvrtcResult) -> *const c_char,
}

impl CudaApi {
    fn load() -> Result<Self> {
        let cuda_library = load_cuda_driver()
            .context("failed to load the CUDA driver library (libcuda.so / nvcuda.dll)")?;
        let nvrtc_library =
            load_nvrtc().context("failed to load NVRTC; install the CUDA toolkit")?;

        Ok(Self {
            // Driver API uses _v2 ABI symbols on 64-bit platforms for the
            // memory-management and context entry points. Loading the _v2
            // names directly avoids relying on the macro renaming that the
            // cuda.h headers do.
            cu_init: unsafe { load_symbol(&cuda_library, b"cuInit\0")? },
            cu_device_get_count: unsafe { load_symbol(&cuda_library, b"cuDeviceGetCount\0")? },
            cu_device_get: unsafe { load_symbol(&cuda_library, b"cuDeviceGet\0")? },
            cu_device_get_name: unsafe { load_symbol(&cuda_library, b"cuDeviceGetName\0")? },
            cu_device_get_attribute: unsafe {
                load_symbol(&cuda_library, b"cuDeviceGetAttribute\0")?
            },
            cu_ctx_create: unsafe { load_symbol(&cuda_library, b"cuCtxCreate_v2\0")? },
            cu_ctx_destroy: unsafe { load_symbol(&cuda_library, b"cuCtxDestroy_v2\0")? },
            cu_mem_alloc: unsafe { load_symbol(&cuda_library, b"cuMemAlloc_v2\0")? },
            cu_mem_free: unsafe { load_symbol(&cuda_library, b"cuMemFree_v2\0")? },
            cu_memcpy_htod: unsafe { load_symbol(&cuda_library, b"cuMemcpyHtoD_v2\0")? },
            cu_memcpy_dtoh: unsafe { load_symbol(&cuda_library, b"cuMemcpyDtoH_v2\0")? },
            cu_module_load_data: unsafe { load_symbol(&cuda_library, b"cuModuleLoadData\0")? },
            cu_module_unload: unsafe { load_symbol(&cuda_library, b"cuModuleUnload\0")? },
            cu_module_get_function: unsafe {
                load_symbol(&cuda_library, b"cuModuleGetFunction\0")?
            },
            cu_launch_kernel: unsafe { load_symbol(&cuda_library, b"cuLaunchKernel\0")? },
            cu_ctx_synchronize: unsafe { load_symbol(&cuda_library, b"cuCtxSynchronize\0")? },
            cu_get_error_string: unsafe { load_symbol(&cuda_library, b"cuGetErrorString\0")? },
            nvrtc_create_program: unsafe { load_symbol(&nvrtc_library, b"nvrtcCreateProgram\0")? },
            nvrtc_compile_program: unsafe {
                load_symbol(&nvrtc_library, b"nvrtcCompileProgram\0")?
            },
            nvrtc_destroy_program: unsafe {
                load_symbol(&nvrtc_library, b"nvrtcDestroyProgram\0")?
            },
            nvrtc_get_program_log_size: unsafe {
                load_symbol(&nvrtc_library, b"nvrtcGetProgramLogSize\0")?
            },
            nvrtc_get_program_log: unsafe { load_symbol(&nvrtc_library, b"nvrtcGetProgramLog\0")? },
            nvrtc_get_ptx_size: unsafe { load_symbol(&nvrtc_library, b"nvrtcGetPTXSize\0")? },
            nvrtc_get_ptx: unsafe { load_symbol(&nvrtc_library, b"nvrtcGetPTX\0")? },
            nvrtc_get_error_string: unsafe {
                load_symbol(&nvrtc_library, b"nvrtcGetErrorString\0")?
            },
            _cuda_library: cuda_library,
            _nvrtc_library: nvrtc_library,
        })
    }

    fn check_cuda(&self, status: CuResult, action: &str) -> Result<()> {
        if status == CUDA_SUCCESS {
            return Ok(());
        }

        Err(anyhow!("{action}: {}", self.cuda_error_string(status)))
    }

    fn check_nvrtc(&self, status: NvrtcResult, action: &str) -> Result<()> {
        if status == NVRTC_SUCCESS {
            return Ok(());
        }

        Err(anyhow!("{action}: {}", self.nvrtc_error_string(status)))
    }

    fn cuda_error_string(&self, status: CuResult) -> String {
        unsafe {
            let mut ptr: *const c_char = ptr::null();
            (self.cu_get_error_string)(status, &mut ptr);
            if ptr.is_null() {
                format!("unknown CUDA error {status}")
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        }
    }

    fn nvrtc_error_string(&self, status: NvrtcResult) -> String {
        unsafe {
            let error = (self.nvrtc_get_error_string)(status);
            if error.is_null() {
                format!("unknown NVRTC error {status}")
            } else {
                CStr::from_ptr(error).to_string_lossy().into_owned()
            }
        }
    }
}

struct CudaContext {
    api: SharedCudaApi,
    ctx: CuContext,
}

impl CudaContext {
    fn create(api: SharedCudaApi, device: CuDevice) -> Result<Self> {
        let mut ctx = ptr::null_mut();
        api.check_cuda(
            unsafe { (api.cu_ctx_create)(&mut ctx, 0, device) },
            "failed to create CUDA context",
        )?;
        Ok(Self { api, ctx })
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if self.ctx.is_null() {
            return;
        }

        let _ = unsafe { (self.api.cu_ctx_destroy)(self.ctx) };
    }
}

struct NvrtcProgramHandle {
    api: SharedCudaApi,
    handle: NvrtcProgram,
    _source: CString,
    _name: CString,
}

impl NvrtcProgramHandle {
    fn new(api: SharedCudaApi, source: &str, name: &str) -> Result<Self> {
        let source = CString::new(source).context("CUDA kernel source contained a NUL byte")?;
        let name = CString::new(name).context("CUDA kernel name contained a NUL byte")?;
        let mut handle = ptr::null_mut();

        api.check_nvrtc(
            unsafe {
                (api.nvrtc_create_program)(
                    &mut handle,
                    source.as_ptr(),
                    name.as_ptr(),
                    0,
                    ptr::null(),
                    ptr::null(),
                )
            },
            "failed to create NVRTC program",
        )?;

        Ok(Self {
            api,
            handle,
            _source: source,
            _name: name,
        })
    }

    fn program_log(&self) -> Result<String> {
        let mut size = 0usize;
        self.api.check_nvrtc(
            unsafe { (self.api.nvrtc_get_program_log_size)(self.handle, &mut size) },
            "failed to query NVRTC compile log size",
        )?;

        if size == 0 {
            return Ok(String::new());
        }

        let mut buffer = vec![0u8; size];
        self.api.check_nvrtc(
            unsafe { (self.api.nvrtc_get_program_log)(self.handle, buffer.as_mut_ptr().cast()) },
            "failed to read NVRTC compile log",
        )?;

        Ok(String::from_utf8_lossy(&buffer)
            .trim_end_matches('\0')
            .to_string())
    }

    fn ptx(&self) -> Result<Vec<u8>> {
        let mut size = 0usize;
        self.api.check_nvrtc(
            unsafe { (self.api.nvrtc_get_ptx_size)(self.handle, &mut size) },
            "failed to query NVRTC PTX size",
        )?;
        ensure!(size > 0, "NVRTC produced an empty PTX object");

        let mut ptx = vec![0u8; size];
        self.api.check_nvrtc(
            unsafe { (self.api.nvrtc_get_ptx)(self.handle, ptx.as_mut_ptr().cast()) },
            "failed to read NVRTC PTX",
        )?;
        Ok(ptx)
    }
}

impl Drop for NvrtcProgramHandle {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        let mut handle = self.handle;
        let _ = unsafe { (self.api.nvrtc_destroy_program)(&mut handle) };
    }
}

struct DeviceBuffer {
    api: SharedCudaApi,
    ptr: CuDevicePtr,
}

impl DeviceBuffer {
    fn new(api: SharedCudaApi, size_bytes: usize) -> Result<Self> {
        let mut ptr: CuDevicePtr = 0;
        api.check_cuda(
            unsafe { (api.cu_mem_alloc)(&mut ptr, size_bytes) },
            "failed to allocate CUDA device buffer",
        )?;
        Ok(Self { api, ptr })
    }

    fn ptr(&self) -> CuDevicePtr {
        self.ptr
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }

        let _ = unsafe { (self.api.cu_mem_free)(self.ptr) };
    }
}

struct GpuKernelModule {
    api: SharedCudaApi,
    _ptx: Vec<u8>,
    module: CuModule,
    kernel: CuFunction,
    full_batch_kernel: CuFunction,
}

impl GpuKernelModule {
    fn build(
        api: SharedCudaApi,
        compute_capability: (u32, u32),
        block_size: u32,
        poll_interval: u32,
        maxrregcount: u32,
    ) -> Result<Rc<Self>> {
        let ptx = build_ptx(
            api.clone(),
            compute_capability,
            block_size,
            poll_interval,
            maxrregcount,
        )?;
        let mut module = ptr::null_mut();
        api.check_cuda(
            unsafe { (api.cu_module_load_data)(&mut module, ptx.as_ptr().cast()) },
            "failed to load CUDA kernel module",
        )?;

        let kernel_name = CString::new(KERNEL_NAME).expect("kernel name must not contain a NUL");
        let mut kernel = ptr::null_mut();
        if let Err(error) = api.check_cuda(
            unsafe { (api.cu_module_get_function)(&mut kernel, module, kernel_name.as_ptr()) },
            "failed to find CUDA kernel entry point",
        ) {
            let _ = unsafe { (api.cu_module_unload)(module) };
            return Err(error);
        }

        let full_batch_kernel_name =
            CString::new(FULL_BATCH_KERNEL_NAME).expect("kernel name must not contain a NUL");
        let mut full_batch_kernel = ptr::null_mut();
        if let Err(error) = api.check_cuda(
            unsafe {
                (api.cu_module_get_function)(
                    &mut full_batch_kernel,
                    module,
                    full_batch_kernel_name.as_ptr(),
                )
            },
            "failed to find CUDA full-batch kernel entry point",
        ) {
            let _ = unsafe { (api.cu_module_unload)(module) };
            return Err(error);
        }

        Ok(Rc::new(Self {
            api,
            _ptx: ptx,
            module,
            kernel,
            full_batch_kernel,
        }))
    }
}

impl Drop for GpuKernelModule {
    fn drop(&mut self) {
        if self.module.is_null() {
            return;
        }

        let _ = unsafe { (self.api.cu_module_unload)(self.module) };
    }
}

struct GpuKernelVariant {
    api: SharedCudaApi,
    _module: Rc<GpuKernelModule>,
    kernel: CuFunction,
    full_batch_kernel: CuFunction,
    block_size: u32,
    poll_interval: u32,
    maxrregcount: u32,
    global_work_size: u32,
}

impl GpuKernelVariant {
    fn build(
        api: SharedCudaApi,
        compute_capability: (u32, u32),
        block_size: u32,
        poll_interval: u32,
        maxrregcount: u32,
        global_work_size: u32,
    ) -> Result<Self> {
        let module = GpuKernelModule::build(
            api,
            compute_capability,
            block_size,
            poll_interval,
            maxrregcount,
        )?;
        Ok(Self::from_compiled(
            module,
            block_size,
            poll_interval,
            maxrregcount,
            global_work_size,
        ))
    }

    fn from_compiled(
        module: Rc<GpuKernelModule>,
        block_size: u32,
        poll_interval: u32,
        maxrregcount: u32,
        global_work_size: u32,
    ) -> Self {
        let api = module.api.clone();
        let kernel = module.kernel;
        let full_batch_kernel = module.full_batch_kernel;

        Self {
            api,
            _module: module,
            kernel,
            full_batch_kernel,
            block_size,
            poll_interval,
            maxrregcount,
            global_work_size: round_up_u32(global_work_size.max(block_size), block_size),
        }
    }

    fn batch_size(&self) -> u64 {
        u64::from(self.global_work_size)
            .saturating_mul(BATCH_ITERATIONS)
            .min(u64::from(u32::MAX))
    }
}

struct CachedVariant {
    block_size: u32,
    poll_interval: u32,
    maxrregcount: u32,
    global_work_size: u32,
}

pub(crate) struct CudaSearchEngine {
    _api: SharedCudaApi,
    _context: CudaContext,
    device_name: String,
    base_words_buffer: DeviceBuffer,
    best_timestamp_buffer: DeviceBuffer,
    prefixes_buffer: DeviceBuffer,
    prefix_count: u32,
    max_error: u32,
    variant: GpuKernelVariant,
    batch_size: u64,
}

impl CudaSearchEngine {
    pub(crate) fn new() -> Result<Self> {
        let api = Arc::new(CudaApi::load()?);

        api.check_cuda(
            unsafe { (api.cu_init)(0) },
            "failed to initialize CUDA driver",
        )?;

        let (device, device_name, compute_capability) = select_gpu_device(&api)?;
        let context = CudaContext::create(api.clone(), device)?;

        let base_words_buffer = DeviceBuffer::new(api.clone(), std::mem::size_of::<[u32; 16]>())?;
        let best_timestamp_buffer = DeviceBuffer::new(api.clone(), std::mem::size_of::<u32>())?;
        let prefixes_buffer = DeviceBuffer::new(api.clone(), PREFIXES_BUFFER_BYTES)?;

        let variant = if let Some(cached) = load_cached_variant(&device_name, compute_capability) {
            match GpuKernelVariant::build(
                api.clone(),
                compute_capability,
                cached.block_size,
                cached.poll_interval,
                cached.maxrregcount,
                cached.global_work_size,
            ) {
                Ok(variant) => variant,
                Err(_) => {
                    let variant = tune_variant(
                        api.clone(),
                        compute_capability,
                        &base_words_buffer,
                        &best_timestamp_buffer,
                        &prefixes_buffer,
                    )?;
                    store_cached_variant(&device_name, compute_capability, &variant);
                    variant
                }
            }
        } else {
            let variant = tune_variant(
                api.clone(),
                compute_capability,
                &base_words_buffer,
                &best_timestamp_buffer,
                &prefixes_buffer,
            )?;
            store_cached_variant(&device_name, compute_capability, &variant);
            variant
        };
        let batch_size = variant.batch_size();

        Ok(Self {
            _api: api,
            _context: context,
            device_name,
            base_words_buffer,
            best_timestamp_buffer,
            prefixes_buffer,
            prefix_count: 0,
            max_error: 0,
            variant,
            batch_size,
        })
    }

    pub(crate) fn device_name(&self) -> &str {
        &self.device_name
    }

    pub(crate) fn batch_size(&self) -> u64 {
        self.batch_size
    }

    pub(crate) fn prepare_search(&mut self, base_words: &[u32; 16]) -> Result<()> {
        self.variant.api.check_cuda(
            unsafe {
                (self.variant.api.cu_memcpy_htod)(
                    self.base_words_buffer.ptr(),
                    base_words.as_ptr().cast(),
                    std::mem::size_of_val(base_words),
                )
            },
            "failed to upload fingerprint base words",
        )?;
        Ok(())
    }

    pub(crate) fn prepare_prefixes(&mut self, prefixes: &HexPrefixSet) -> Result<()> {
        let packed = prefixes.pack();
        ensure!(
            packed.len() / PREFIX_WORDS <= MAX_PREFIXES,
            "prefix set exceeds the configured maximum of {MAX_PREFIXES} prefixes"
        );

        self.variant.api.check_cuda(
            unsafe {
                (self.variant.api.cu_memcpy_htod)(
                    self.prefixes_buffer.ptr(),
                    packed.as_ptr().cast(),
                    std::mem::size_of_val(packed.as_slice()),
                )
            },
            "failed to upload prefix array",
        )?;
        self.prefix_count = (packed.len() / PREFIX_WORDS) as u32;
        self.max_error = prefixes.max_error();
        Ok(())
    }

    pub(crate) fn search_batch(&mut self, start_timestamp: u32, count: u32) -> Result<Option<u32>> {
        ensure!(
            self.prefix_count > 0,
            "prepare_prefixes must be called before search_batch"
        );

        let not_found = [NOT_FOUND_TIMESTAMP; 1];
        self.variant.api.check_cuda(
            unsafe {
                (self.variant.api.cu_memcpy_htod)(
                    self.best_timestamp_buffer.ptr(),
                    not_found.as_ptr().cast(),
                    std::mem::size_of_val(&not_found),
                )
            },
            "failed to reset best-timestamp buffer",
        )?;

        let work_items = if count <= self.variant.global_work_size {
            round_up_u32(count.max(1), self.variant.block_size)
        } else {
            self.variant.global_work_size
        };
        launch_kernel(
            &self.variant,
            self.base_words_buffer.ptr(),
            start_timestamp,
            count,
            self.prefixes_buffer.ptr(),
            self.prefix_count,
            self.max_error,
            self.best_timestamp_buffer.ptr(),
            work_items,
        )?;

        let mut best_timestamp = [u32::MAX; 1];
        self.variant.api.check_cuda(
            unsafe {
                (self.variant.api.cu_memcpy_dtoh)(
                    best_timestamp.as_mut_ptr().cast(),
                    self.best_timestamp_buffer.ptr(),
                    std::mem::size_of_val(&best_timestamp),
                )
            },
            "failed to read best-timestamp buffer",
        )?;

        if best_timestamp[0] == NOT_FOUND_TIMESTAMP {
            return Ok(None);
        }

        Ok(Some(best_timestamp[0]))
    }
}

fn build_ptx(
    api: SharedCudaApi,
    compute_capability: (u32, u32),
    block_size: u32,
    poll_interval: u32,
    maxrregcount: u32,
) -> Result<Vec<u8>> {
    let program = NvrtcProgramHandle::new(api.clone(), KERNEL_SOURCE, "pgp_vanity_kernel.cu")?;
    let arch_option = format!(
        "--gpu-architecture=compute_{}{}",
        compute_capability.0, compute_capability.1
    );
    let mut options = vec![
        CString::new(arch_option).context("failed to encode --gpu-architecture option")?,
        CString::new("--std=c++17").expect("static compile option must not contain a NUL"),
        CString::new("-default-device").expect("static compile option must not contain a NUL"),
        CString::new(format!("-DFIXED_BLOCK_SIZE={block_size}"))
            .context("failed to encode block-size compile option")?,
        CString::new(format!("-DPOLL_INTERVAL={poll_interval}"))
            .context("failed to encode poll-interval compile option")?,
        CString::new(format!("-DFULL_BATCH_OUTER_LOOPS={}", BATCH_ITERATIONS / 4))
            .context("failed to encode full-batch loop-count compile option")?,
    ];
    if maxrregcount > 0 {
        options.push(
            CString::new(format!("--maxrregcount={maxrregcount}"))
                .context("failed to encode --maxrregcount option")?,
        );
    }
    let option_ptrs = options
        .iter()
        .map(|option| option.as_ptr())
        .collect::<Vec<_>>();

    let compile_status = unsafe {
        (api.nvrtc_compile_program)(
            program.handle,
            option_ptrs.len() as c_int,
            option_ptrs.as_ptr(),
        )
    };
    let log = program.program_log()?;
    if compile_status != NVRTC_SUCCESS {
        let suffix = if log.is_empty() {
            String::new()
        } else {
            format!("\n{log}")
        };
        bail!(
            "failed to compile CUDA kernel for compute_{}{}, block size {block_size}, poll interval {poll_interval}, maxrregcount {maxrregcount}: {}{suffix}",
            compute_capability.0,
            compute_capability.1,
            api.nvrtc_error_string(compile_status),
        );
    }

    let ptx = program.ptx()?;
    if let Some(path) = env::var_os("PGP_VANITY_DUMP_PTX") {
        let _ = fs::write(path, &ptx);
    }
    Ok(ptx)
}

#[allow(clippy::too_many_arguments)]
fn launch_partial_kernel(
    variant: &GpuKernelVariant,
    base_words_buffer: CuDevicePtr,
    start_timestamp: u32,
    count: u32,
    prefixes_buffer: CuDevicePtr,
    prefix_count: u32,
    max_error: u32,
    best_timestamp_buffer: CuDevicePtr,
    work_items: u32,
) -> Result<()> {
    let grid_dim_x = work_items / variant.block_size;
    ensure!(
        grid_dim_x > 0,
        "CUDA grid dimension must be greater than zero"
    );

    let mut base_words_arg = base_words_buffer;
    let mut start_timestamp_arg = start_timestamp;
    let mut count_arg = count;
    let mut prefixes_arg = prefixes_buffer;
    let mut prefix_count_arg = prefix_count;
    let mut max_error_arg = max_error;
    let mut best_timestamp_arg = best_timestamp_buffer;
    let mut kernel_params = [
        (&mut base_words_arg as *mut CuDevicePtr).cast::<c_void>(),
        (&mut start_timestamp_arg as *mut u32).cast::<c_void>(),
        (&mut count_arg as *mut u32).cast::<c_void>(),
        (&mut prefixes_arg as *mut CuDevicePtr).cast::<c_void>(),
        (&mut prefix_count_arg as *mut u32).cast::<c_void>(),
        (&mut max_error_arg as *mut u32).cast::<c_void>(),
        (&mut best_timestamp_arg as *mut CuDevicePtr).cast::<c_void>(),
    ];

    variant.api.check_cuda(
        unsafe {
            (variant.api.cu_launch_kernel)(
                variant.kernel,
                grid_dim_x,
                1,
                1,
                variant.block_size,
                1,
                1,
                0,
                ptr::null_mut(),
                kernel_params.as_mut_ptr(),
                ptr::null_mut(),
            )
        },
        "failed to launch CUDA kernel",
    )?;
    variant.api.check_cuda(
        unsafe { (variant.api.cu_ctx_synchronize)() },
        "CUDA kernel execution failed",
    )?;
    Ok(())
}

fn launch_full_batch_kernel(
    variant: &GpuKernelVariant,
    base_words_buffer: CuDevicePtr,
    start_timestamp: u32,
    prefixes_buffer: CuDevicePtr,
    prefix_count: u32,
    max_error: u32,
    best_timestamp_buffer: CuDevicePtr,
) -> Result<()> {
    let grid_dim_x = variant.global_work_size / variant.block_size;
    ensure!(
        grid_dim_x > 0,
        "CUDA grid dimension must be greater than zero"
    );

    let mut base_words_arg = base_words_buffer;
    let mut start_timestamp_arg = start_timestamp;
    let mut prefixes_arg = prefixes_buffer;
    let mut prefix_count_arg = prefix_count;
    let mut max_error_arg = max_error;
    let mut best_timestamp_arg = best_timestamp_buffer;
    let mut kernel_params = [
        (&mut base_words_arg as *mut CuDevicePtr).cast::<c_void>(),
        (&mut start_timestamp_arg as *mut u32).cast::<c_void>(),
        (&mut prefixes_arg as *mut CuDevicePtr).cast::<c_void>(),
        (&mut prefix_count_arg as *mut u32).cast::<c_void>(),
        (&mut max_error_arg as *mut u32).cast::<c_void>(),
        (&mut best_timestamp_arg as *mut CuDevicePtr).cast::<c_void>(),
    ];

    variant.api.check_cuda(
        unsafe {
            (variant.api.cu_launch_kernel)(
                variant.full_batch_kernel,
                grid_dim_x,
                1,
                1,
                variant.block_size,
                1,
                1,
                0,
                ptr::null_mut(),
                kernel_params.as_mut_ptr(),
                ptr::null_mut(),
            )
        },
        "failed to launch CUDA full-batch kernel",
    )?;
    variant.api.check_cuda(
        unsafe { (variant.api.cu_ctx_synchronize)() },
        "CUDA full-batch kernel execution failed",
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_kernel(
    variant: &GpuKernelVariant,
    base_words_buffer: CuDevicePtr,
    start_timestamp: u32,
    count: u32,
    prefixes_buffer: CuDevicePtr,
    prefix_count: u32,
    max_error: u32,
    best_timestamp_buffer: CuDevicePtr,
    work_items: u32,
) -> Result<()> {
    if count == variant.batch_size() as u32 && work_items == variant.global_work_size {
        return launch_full_batch_kernel(
            variant,
            base_words_buffer,
            start_timestamp,
            prefixes_buffer,
            prefix_count,
            max_error,
            best_timestamp_buffer,
        );
    }

    launch_partial_kernel(
        variant,
        base_words_buffer,
        start_timestamp,
        count,
        prefixes_buffer,
        prefix_count,
        max_error,
        best_timestamp_buffer,
        work_items,
    )
}

fn tune_variant(
    api: SharedCudaApi,
    compute_capability: (u32, u32),
    base_words_buffer: &DeviceBuffer,
    best_timestamp_buffer: &DeviceBuffer,
    prefixes_buffer: &DeviceBuffer,
) -> Result<GpuKernelVariant> {
    let benchmark_prefix = benchmark_prefix();
    let benchmark_words = FingerprintSearch::new(BENCHMARK_PUBLIC_KEY).base_words();
    api.check_cuda(
        unsafe {
            (api.cu_memcpy_htod)(
                base_words_buffer.ptr(),
                benchmark_words.as_ptr().cast(),
                std::mem::size_of_val(&benchmark_words),
            )
        },
        "failed to upload benchmark base words",
    )?;

    let (mask_high, mask_low) = benchmark_prefix.mask_words();
    let (value_high, value_low) = benchmark_prefix.value_words();
    let packed_prefix = [mask_high, mask_low, value_high, value_low];
    api.check_cuda(
        unsafe {
            (api.cu_memcpy_htod)(
                prefixes_buffer.ptr(),
                packed_prefix.as_ptr().cast(),
                std::mem::size_of_val(&packed_prefix),
            )
        },
        "failed to upload benchmark prefix",
    )?;
    let prefix_count = 1u32;

    let mut finalists = Vec::with_capacity(FINALIST_COUNT);
    let mut last_error: Option<anyhow::Error> = None;

    for &maxrregcount in &MAXRREGCOUNT_CANDIDATES {
        for &block_size in &BLOCK_SIZE_CANDIDATES {
            for &poll_interval in &POLL_INTERVAL_CANDIDATES {
                let compiled = match GpuKernelModule::build(
                    api.clone(),
                    compute_capability,
                    block_size,
                    poll_interval,
                    maxrregcount,
                ) {
                    Ok(compiled) => compiled,
                    Err(error) => {
                        last_error = Some(error);
                        continue;
                    }
                };

                for &global_work_size in &GLOBAL_WORK_SIZE_CANDIDATES {
                    let variant = GpuKernelVariant::from_compiled(
                        compiled.clone(),
                        block_size,
                        poll_interval,
                        maxrregcount,
                        global_work_size,
                    );
                    let count = variant.batch_size() as u32;
                    let hashes_per_second = match benchmark_variant_once(
                        &variant,
                        base_words_buffer.ptr(),
                        best_timestamp_buffer.ptr(),
                        prefixes_buffer.ptr(),
                        prefix_count,
                        BENCHMARK_START_TIMESTAMP,
                        count,
                    )
                    .and_then(|_| {
                        benchmark_variant_once(
                            &variant,
                            base_words_buffer.ptr(),
                            best_timestamp_buffer.ptr(),
                            prefixes_buffer.ptr(),
                            prefix_count,
                            BENCHMARK_START_TIMESTAMP.wrapping_add(0x4000_0000),
                            count,
                        )
                    }) {
                        Ok(hashes_per_second) => hashes_per_second,
                        Err(error) => {
                            last_error = Some(error);
                            continue;
                        }
                    };

                    if env::var_os("PGP_VANITY_TUNE_LOG").is_some() {
                        eprintln!(
                            "tune: block={} poll={} maxreg={} gws={} -> {:.3} GH/s",
                            block_size,
                            poll_interval,
                            maxrregcount,
                            global_work_size,
                            hashes_per_second / 1e9,
                        );
                    }
                    push_finalist(&mut finalists, variant, hashes_per_second);
                }
            }
        }
    }

    let mut best: Option<(GpuKernelVariant, f64)> = None;
    for (index, (variant, _)) in finalists.into_iter().enumerate() {
        let hashes_per_second = benchmark_variant_sustained(
            &variant,
            base_words_buffer.ptr(),
            best_timestamp_buffer.ptr(),
            prefixes_buffer.ptr(),
            prefix_count,
            BENCHMARK_START_TIMESTAMP.wrapping_add((index as u32).wrapping_mul(0x1F12_3BB5)),
            variant.batch_size() as u32,
            SUSTAINED_BENCHMARK_DURATION,
        )?;

        if env::var_os("PGP_VANITY_TUNE_LOG").is_some() {
            eprintln!(
                "finalist {}: block={} poll={} maxreg={} gws={} -> {:.3} GH/s (sustained)",
                index,
                variant.block_size,
                variant.poll_interval,
                variant.maxrregcount,
                variant.global_work_size,
                hashes_per_second / 1e9,
            );
        }
        match &best {
            Some((_, best_hashes_per_second)) if *best_hashes_per_second >= hashes_per_second => {}
            _ => best = Some((variant, hashes_per_second)),
        }
    }

    if let Some((variant, _)) = best {
        return Ok(variant);
    }

    if let Some(error) = last_error {
        return Err(error);
    }

    GpuKernelVariant::build(
        api,
        compute_capability,
        BLOCK_SIZE_CANDIDATES[0],
        POLL_INTERVAL_CANDIDATES[0],
        MAXRREGCOUNT_CANDIDATES[0],
        GLOBAL_WORK_SIZE_CANDIDATES[0],
    )
}

#[allow(clippy::too_many_arguments)]
fn benchmark_variant_elapsed(
    variant: &GpuKernelVariant,
    base_words_buffer: CuDevicePtr,
    best_timestamp_buffer: CuDevicePtr,
    prefixes_buffer: CuDevicePtr,
    prefix_count: u32,
    start_timestamp: u32,
    count: u32,
) -> Result<Duration> {
    let not_found = [NOT_FOUND_TIMESTAMP; 1];
    variant.api.check_cuda(
        unsafe {
            (variant.api.cu_memcpy_htod)(
                best_timestamp_buffer,
                not_found.as_ptr().cast(),
                std::mem::size_of_val(&not_found),
            )
        },
        "failed to reset benchmark best-timestamp buffer",
    )?;

    let start = Instant::now();
    launch_kernel(
        variant,
        base_words_buffer,
        start_timestamp,
        count,
        prefixes_buffer,
        prefix_count,
        0,
        best_timestamp_buffer,
        variant.global_work_size,
    )?;
    let elapsed = start.elapsed();

    ensure!(
        !elapsed.is_zero(),
        "benchmark kernel completed too quickly to measure"
    );
    Ok(elapsed)
}

#[allow(clippy::too_many_arguments)]
fn benchmark_variant_once(
    variant: &GpuKernelVariant,
    base_words_buffer: CuDevicePtr,
    best_timestamp_buffer: CuDevicePtr,
    prefixes_buffer: CuDevicePtr,
    prefix_count: u32,
    start_timestamp: u32,
    count: u32,
) -> Result<f64> {
    let elapsed = benchmark_variant_elapsed(
        variant,
        base_words_buffer,
        best_timestamp_buffer,
        prefixes_buffer,
        prefix_count,
        start_timestamp,
        count,
    )?;
    Ok(count as f64 / elapsed.as_secs_f64())
}

#[allow(clippy::too_many_arguments)]
fn benchmark_variant_sustained(
    variant: &GpuKernelVariant,
    base_words_buffer: CuDevicePtr,
    best_timestamp_buffer: CuDevicePtr,
    prefixes_buffer: CuDevicePtr,
    prefix_count: u32,
    start_timestamp: u32,
    count: u32,
    duration: Duration,
) -> Result<f64> {
    let _ = benchmark_variant_once(
        variant,
        base_words_buffer,
        best_timestamp_buffer,
        prefixes_buffer,
        prefix_count,
        start_timestamp,
        count,
    )?;

    let benchmark_start = Instant::now();
    let mut total_count = 0u64;
    let mut iteration = 0u32;

    while benchmark_start.elapsed() < duration {
        let iteration_timestamp = start_timestamp.wrapping_add(iteration.wrapping_mul(0x9E37_79B9));
        let elapsed = benchmark_variant_elapsed(
            variant,
            base_words_buffer,
            best_timestamp_buffer,
            prefixes_buffer,
            prefix_count,
            iteration_timestamp,
            count,
        )?;
        total_count = total_count.saturating_add(u64::from(count));
        iteration = iteration.wrapping_add(1);

        if benchmark_start.elapsed() >= duration {
            ensure!(
                !elapsed.is_zero(),
                "sustained benchmark iteration completed too quickly to measure"
            );
        }
    }

    let elapsed = benchmark_start.elapsed();
    ensure!(
        !elapsed.is_zero(),
        "sustained benchmark completed too quickly to measure"
    );
    Ok(total_count as f64 / elapsed.as_secs_f64())
}

fn load_cached_variant(device_name: &str, compute_capability: (u32, u32)) -> Option<CachedVariant> {
    let path = tuning_cache_path(TUNING_CACHE_FILE);
    let contents = fs::read_to_string(path).ok()?;
    let mut lines = contents.lines();

    let version = lines.next()?.parse::<u32>().ok()?;
    let cached_device_name = lines.next()?;
    let cached_cc = lines.next()?;
    let block_size = lines.next()?.parse::<u32>().ok()?;
    let poll_interval = lines.next()?.parse::<u32>().ok()?;
    let maxrregcount = lines.next()?.parse::<u32>().ok()?;
    let global_work_size = lines.next()?.parse::<u32>().ok()?;

    let expected_cc = format!("{}.{}", compute_capability.0, compute_capability.1);
    if version != TUNING_CACHE_VERSION
        || cached_device_name != device_name
        || cached_cc != expected_cc
    {
        return None;
    }

    if !BLOCK_SIZE_CANDIDATES.contains(&block_size)
        || !POLL_INTERVAL_CANDIDATES.contains(&poll_interval)
        || !MAXRREGCOUNT_CANDIDATES.contains(&maxrregcount)
        || !GLOBAL_WORK_SIZE_CANDIDATES.contains(&global_work_size)
    {
        return None;
    }

    Some(CachedVariant {
        block_size,
        poll_interval,
        maxrregcount,
        global_work_size,
    })
}

fn store_cached_variant(
    device_name: &str,
    compute_capability: (u32, u32),
    variant: &GpuKernelVariant,
) {
    let path = tuning_cache_path(TUNING_CACHE_FILE);

    if let Some(parent) = path.parent()
        && fs::create_dir_all(parent).is_err()
    {
        return;
    }

    let contents = format!(
        "{TUNING_CACHE_VERSION}\n{device_name}\n{}.{}\n{}\n{}\n{}\n{}\n",
        compute_capability.0,
        compute_capability.1,
        variant.block_size,
        variant.poll_interval,
        variant.maxrregcount,
        variant.global_work_size,
    );
    let _ = fs::write(path, contents);
}

fn select_gpu_device(api: &CudaApi) -> Result<(CuDevice, String, (u32, u32))> {
    let mut count = 0;
    api.check_cuda(
        unsafe { (api.cu_device_get_count)(&mut count) },
        "failed to enumerate CUDA devices",
    )?;
    if count <= 0 {
        bail!(
            "no CUDA GPU device found; install a recent NVIDIA driver and the CUDA toolkit for your NVIDIA GPU"
        );
    }

    let mut device: CuDevice = 0;
    api.check_cuda(
        unsafe { (api.cu_device_get)(&mut device, 0) },
        "failed to select CUDA device",
    )?;

    let mut device_name = vec![0 as c_char; 256];
    api.check_cuda(
        unsafe {
            (api.cu_device_get_name)(device_name.as_mut_ptr(), device_name.len() as c_int, device)
        },
        "failed to query CUDA device name",
    )?;
    let device_name = unsafe { CStr::from_ptr(device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    let mut major: c_int = 0;
    api.check_cuda(
        unsafe {
            (api.cu_device_get_attribute)(
                &mut major,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device,
            )
        },
        "failed to query CUDA compute-capability major",
    )?;
    let mut minor: c_int = 0;
    api.check_cuda(
        unsafe {
            (api.cu_device_get_attribute)(
                &mut minor,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device,
            )
        },
        "failed to query CUDA compute-capability minor",
    )?;

    Ok((device, device_name, (major as u32, minor as u32)))
}

fn load_cuda_driver() -> Result<Library> {
    // nvcuda.dll ships with the NVIDIA driver and is installed in System32, so
    // a bare LoadLibrary lookup is enough on Windows. On Linux the driver
    // installs libcuda.so.1 (with libcuda.so provided by the toolkit stubs).
    let candidates: &[&str] = if cfg!(windows) {
        &["nvcuda.dll"]
    } else {
        &["libcuda.so.1", "libcuda.so"]
    };

    for name in candidates {
        if let Ok(library) = unsafe { Library::new(name) } {
            return Ok(library);
        }
    }

    Err(anyhow!(
        "failed to load the CUDA driver library; ensure an NVIDIA driver is installed"
    ))
}

fn load_nvrtc() -> Result<Library> {
    let mut candidates = Vec::new();

    if let Some(path) = env::var_os("CUDA_PATH") {
        candidates.extend(nvrtc_candidates_in(&PathBuf::from(path)));
    }
    if let Some(path) = env::var_os("CUDA_HOME") {
        candidates.extend(nvrtc_candidates_in(&PathBuf::from(path)));
    }
    if let Some(paths) = env::var_os("PATH") {
        for entry in env::split_paths(&paths) {
            candidates.extend(nvrtc_dlls_in_dir(&entry));
        }
    }

    // Fallback to whichever name the dynamic loader can resolve (matches
    // historical Linux ABI names and the unversioned Windows stub when
    // present).
    if cfg!(windows) {
        candidates.push(PathBuf::from("nvrtc64.dll"));
    } else {
        candidates.push(PathBuf::from("libnvrtc.so"));
        candidates.push(PathBuf::from("libnvrtc.so.1"));
    }

    let mut tried: Vec<PathBuf> = Vec::new();
    for candidate in candidates {
        if tried.iter().any(|existing| existing == &candidate) {
            continue;
        }
        // NVRTC loads `nvrtc-builtins64_*.dll` from its own directory at
        // runtime via LoadLibrary, which on Windows does not search the
        // calling DLL's directory by default. Make sure that directory is on
        // PATH before opening NVRTC so the sidecar resolves.
        if let Some(parent) = candidate.parent() {
            prepend_to_path_env(parent);
        }
        if let Ok(library) = unsafe { Library::new(&candidate) } {
            return Ok(library);
        }
        tried.push(candidate);
    }

    Err(anyhow!(
        "could not locate an NVRTC runtime library; checked: {:?}",
        tried,
    ))
}

fn prepend_to_path_env(directory: &Path) {
    let key = if cfg!(windows) {
        "PATH"
    } else {
        "LD_LIBRARY_PATH"
    };
    let mut entries: Vec<PathBuf> = env::var_os(key)
        .map(|paths| env::split_paths(&paths).collect())
        .unwrap_or_default();
    if entries.iter().any(|existing| existing == directory) {
        return;
    }
    entries.insert(0, directory.to_path_buf());
    if let Ok(joined) = env::join_paths(entries) {
        // SAFETY: set_var is unsafe on unix because of multi-threaded
        // observers of the env block; we mutate PATH before launching any
        // background work and the CUDA loader reads it synchronously.
        unsafe { env::set_var(key, joined) };
    }
}

fn nvrtc_candidates_in(root: &Path) -> Vec<PathBuf> {
    let mut results = Vec::new();
    // CUDA toolkit on Windows places DLLs under `bin\x64\`; the older layout
    // (and Linux) puts them in `bin/` or `lib/`.
    for sub in ["bin/x64", "bin", "lib/x64", "lib", "lib64"] {
        results.extend(nvrtc_dlls_in_dir(&root.join(sub)));
    }
    results
}

fn nvrtc_dlls_in_dir(dir: &Path) -> Vec<PathBuf> {
    let mut results = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return results;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let lower = name.to_ascii_lowercase();
        // Match Windows `nvrtc64_<ver>_0.dll` and Linux `libnvrtc.so[.ver]`.
        if (lower.starts_with("nvrtc64_") && lower.ends_with(".dll"))
            || lower == "nvrtc64.dll"
            || lower.starts_with("libnvrtc.so")
        {
            results.push(path);
        }
    }
    // Prefer the highest-versioned NVRTC build so repeated runs converge on
    // the newest toolkit available in PATH.
    results.sort();
    results.reverse();
    results
}
