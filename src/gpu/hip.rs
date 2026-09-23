use std::env;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fs;
use std::path::PathBuf;
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

// wave64 can produce HSA_STATUS_ERROR_INVALID_ISA on some GPU + ROCm
// combinations (process-fatal, not catchable from Rust). Limit to wave32 to
// keep tuning from killing the binary.
const WAVE64_CANDIDATES: [bool; 1] = [false];
// CU mode adds a 6th tuning dimension with marginal headroom on RDNA2; the
// extra variants more than double tuning time. Skip for now.
const CU_MODE_CANDIDATES: [bool; 1] = [false];
const TUNING_CACHE_VERSION: u32 = 10;
const TUNING_CACHE_FILE: &str = "hip-tuning.txt";

type HipError = c_int;
type HiprtcResult = c_int;
type HipDevice = c_int;
type HipStream = *mut c_void;
type HipModule = *mut c_void;
type HipFunction = *mut c_void;
type HipDevicePtr = *mut c_void;
type HiprtcProgram = *mut c_void;

const HIP_SUCCESS: HipError = 0;
const HIPRTC_SUCCESS: HiprtcResult = 0;

type SharedHipApi = Arc<HipApi>;

struct HipApi {
    _hip_library: Library,
    _hiprtc_library: Library,
    hip_get_device_count: unsafe extern "C" fn(*mut c_int) -> HipError,
    hip_set_device: unsafe extern "C" fn(HipDevice) -> HipError,
    hip_device_get_name: unsafe extern "C" fn(*mut c_char, c_int, HipDevice) -> HipError,
    hip_malloc: unsafe extern "C" fn(*mut HipDevicePtr, usize) -> HipError,
    hip_free: unsafe extern "C" fn(HipDevicePtr) -> HipError,
    hip_memcpy_htod: unsafe extern "C" fn(HipDevicePtr, *const c_void, usize) -> HipError,
    hip_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, HipDevicePtr, usize) -> HipError,
    hip_module_load_data: unsafe extern "C" fn(*mut HipModule, *const c_void) -> HipError,
    hip_module_unload: unsafe extern "C" fn(HipModule) -> HipError,
    hip_module_get_function:
        unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> HipError,
    hip_module_launch_kernel: unsafe extern "C" fn(
        HipFunction,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        HipStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> HipError,
    hip_device_synchronize: unsafe extern "C" fn() -> HipError,
    hip_get_error_string: unsafe extern "C" fn(HipError) -> *const c_char,
    hiprtc_create_program: unsafe extern "C" fn(
        *mut HiprtcProgram,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> HiprtcResult,
    hiprtc_compile_program:
        unsafe extern "C" fn(HiprtcProgram, c_int, *const *const c_char) -> HiprtcResult,
    hiprtc_destroy_program: unsafe extern "C" fn(*mut HiprtcProgram) -> HiprtcResult,
    hiprtc_get_program_log_size: unsafe extern "C" fn(HiprtcProgram, *mut usize) -> HiprtcResult,
    hiprtc_get_program_log: unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> HiprtcResult,
    hiprtc_get_code_size: unsafe extern "C" fn(HiprtcProgram, *mut usize) -> HiprtcResult,
    hiprtc_get_code: unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> HiprtcResult,
    hiprtc_get_error_string: unsafe extern "C" fn(HiprtcResult) -> *const c_char,
}

impl HipApi {
    fn load() -> Result<Self> {
        let hip_library = load_library("libamdhip64.so")
            .context("failed to load ROCm HIP runtime library (libamdhip64.so)")?;
        let hiprtc_library =
            load_library("libhiprtc.so").context("failed to load HIPRTC library (libhiprtc.so)")?;

        Ok(Self {
            hip_get_device_count: unsafe { load_symbol(&hip_library, b"hipGetDeviceCount\0")? },
            hip_set_device: unsafe { load_symbol(&hip_library, b"hipSetDevice\0")? },
            hip_device_get_name: unsafe { load_symbol(&hip_library, b"hipDeviceGetName\0")? },
            hip_malloc: unsafe { load_symbol(&hip_library, b"hipMalloc\0")? },
            hip_free: unsafe { load_symbol(&hip_library, b"hipFree\0")? },
            hip_memcpy_htod: unsafe { load_symbol(&hip_library, b"hipMemcpyHtoD\0")? },
            hip_memcpy_dtoh: unsafe { load_symbol(&hip_library, b"hipMemcpyDtoH\0")? },
            hip_module_load_data: unsafe { load_symbol(&hip_library, b"hipModuleLoadData\0")? },
            hip_module_unload: unsafe { load_symbol(&hip_library, b"hipModuleUnload\0")? },
            hip_module_get_function: unsafe {
                load_symbol(&hip_library, b"hipModuleGetFunction\0")?
            },
            hip_module_launch_kernel: unsafe {
                load_symbol(&hip_library, b"hipModuleLaunchKernel\0")?
            },
            hip_device_synchronize: unsafe {
                load_symbol(&hip_library, b"hipDeviceSynchronize\0")?
            },
            hip_get_error_string: unsafe { load_symbol(&hip_library, b"hipGetErrorString\0")? },
            hiprtc_create_program: unsafe {
                load_symbol(&hiprtc_library, b"hiprtcCreateProgram\0")?
            },
            hiprtc_compile_program: unsafe {
                load_symbol(&hiprtc_library, b"hiprtcCompileProgram\0")?
            },
            hiprtc_destroy_program: unsafe {
                load_symbol(&hiprtc_library, b"hiprtcDestroyProgram\0")?
            },
            hiprtc_get_program_log_size: unsafe {
                load_symbol(&hiprtc_library, b"hiprtcGetProgramLogSize\0")?
            },
            hiprtc_get_program_log: unsafe {
                load_symbol(&hiprtc_library, b"hiprtcGetProgramLog\0")?
            },
            hiprtc_get_code_size: unsafe { load_symbol(&hiprtc_library, b"hiprtcGetCodeSize\0")? },
            hiprtc_get_code: unsafe { load_symbol(&hiprtc_library, b"hiprtcGetCode\0")? },
            hiprtc_get_error_string: unsafe {
                load_symbol(&hiprtc_library, b"hiprtcGetErrorString\0")?
            },
            _hip_library: hip_library,
            _hiprtc_library: hiprtc_library,
        })
    }

    fn check_hip(&self, status: HipError, action: &str) -> Result<()> {
        if status == HIP_SUCCESS {
            return Ok(());
        }

        Err(anyhow!("{action}: {}", self.hip_error_string(status)))
    }

    fn check_hiprtc(&self, status: HiprtcResult, action: &str) -> Result<()> {
        if status == HIPRTC_SUCCESS {
            return Ok(());
        }

        Err(anyhow!("{action}: {}", self.hiprtc_error_string(status)))
    }

    fn hip_error_string(&self, status: HipError) -> String {
        unsafe {
            let error = (self.hip_get_error_string)(status);
            if error.is_null() {
                format!("unknown HIP error {status}")
            } else {
                CStr::from_ptr(error).to_string_lossy().into_owned()
            }
        }
    }

    fn hiprtc_error_string(&self, status: HiprtcResult) -> String {
        unsafe {
            let error = (self.hiprtc_get_error_string)(status);
            if error.is_null() {
                format!("unknown HIPRTC error {status}")
            } else {
                CStr::from_ptr(error).to_string_lossy().into_owned()
            }
        }
    }
}

struct HiprtcProgramHandle {
    api: SharedHipApi,
    handle: HiprtcProgram,
    _source: CString,
    _name: CString,
}

impl HiprtcProgramHandle {
    fn new(api: SharedHipApi, source: &str, name: &str) -> Result<Self> {
        let source = CString::new(source).context("HIP kernel source contained a NUL byte")?;
        let name = CString::new(name).context("HIP kernel name contained a NUL byte")?;
        let mut handle = ptr::null_mut();

        api.check_hiprtc(
            unsafe {
                (api.hiprtc_create_program)(
                    &mut handle,
                    source.as_ptr(),
                    name.as_ptr(),
                    0,
                    ptr::null(),
                    ptr::null(),
                )
            },
            "failed to create HIPRTC program",
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
        self.api.check_hiprtc(
            unsafe { (self.api.hiprtc_get_program_log_size)(self.handle, &mut size) },
            "failed to query HIPRTC compile log size",
        )?;

        if size == 0 {
            return Ok(String::new());
        }

        let mut buffer = vec![0u8; size];
        self.api.check_hiprtc(
            unsafe { (self.api.hiprtc_get_program_log)(self.handle, buffer.as_mut_ptr().cast()) },
            "failed to read HIPRTC compile log",
        )?;

        Ok(String::from_utf8_lossy(&buffer)
            .trim_end_matches('\0')
            .to_string())
    }

    fn code(&self) -> Result<Vec<u8>> {
        let mut size = 0usize;
        self.api.check_hiprtc(
            unsafe { (self.api.hiprtc_get_code_size)(self.handle, &mut size) },
            "failed to query HIPRTC code size",
        )?;
        ensure!(size > 0, "HIPRTC produced an empty code object");

        let mut code = vec![0u8; size];
        self.api.check_hiprtc(
            unsafe { (self.api.hiprtc_get_code)(self.handle, code.as_mut_ptr().cast()) },
            "failed to read HIPRTC code object",
        )?;
        Ok(code)
    }
}

impl Drop for HiprtcProgramHandle {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        let mut handle = self.handle;
        let _ = unsafe { (self.api.hiprtc_destroy_program)(&mut handle) };
    }
}

struct DeviceBuffer {
    api: SharedHipApi,
    ptr: HipDevicePtr,
}

impl DeviceBuffer {
    fn new(api: SharedHipApi, size_bytes: usize) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        api.check_hip(
            unsafe { (api.hip_malloc)(&mut ptr, size_bytes) },
            "failed to allocate HIP device buffer",
        )?;
        Ok(Self { api, ptr })
    }

    fn ptr(&self) -> HipDevicePtr {
        self.ptr
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        let _ = unsafe { (self.api.hip_free)(self.ptr) };
    }
}

struct GpuKernelModule {
    api: SharedHipApi,
    _code: Vec<u8>,
    module: HipModule,
    kernel: HipFunction,
    full_batch_kernel: HipFunction,
}

impl GpuKernelModule {
    fn build(
        api: SharedHipApi,
        block_size: u32,
        poll_interval: u32,
        wave64: bool,
        cu_mode: bool,
    ) -> Result<Rc<Self>> {
        let code = build_code_object(api.clone(), block_size, poll_interval, wave64, cu_mode)?;
        let mut module = ptr::null_mut();
        api.check_hip(
            unsafe { (api.hip_module_load_data)(&mut module, code.as_ptr().cast()) },
            "failed to load HIP kernel module",
        )?;

        let kernel_name = CString::new(KERNEL_NAME).expect("kernel name must not contain a NUL");
        let mut kernel = ptr::null_mut();
        if let Err(error) = api.check_hip(
            unsafe { (api.hip_module_get_function)(&mut kernel, module, kernel_name.as_ptr()) },
            "failed to find HIP kernel entry point",
        ) {
            let _ = unsafe { (api.hip_module_unload)(module) };
            return Err(error);
        }

        let full_batch_kernel_name =
            CString::new(FULL_BATCH_KERNEL_NAME).expect("kernel name must not contain a NUL");
        let mut full_batch_kernel = ptr::null_mut();
        if let Err(error) = api.check_hip(
            unsafe {
                (api.hip_module_get_function)(
                    &mut full_batch_kernel,
                    module,
                    full_batch_kernel_name.as_ptr(),
                )
            },
            "failed to find HIP full-batch kernel entry point",
        ) {
            let _ = unsafe { (api.hip_module_unload)(module) };
            return Err(error);
        }

        Ok(Rc::new(Self {
            api,
            _code: code,
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

        let _ = unsafe { (self.api.hip_module_unload)(self.module) };
    }
}

struct GpuKernelVariant {
    api: SharedHipApi,
    _module: Rc<GpuKernelModule>,
    kernel: HipFunction,
    full_batch_kernel: HipFunction,
    block_size: u32,
    poll_interval: u32,
    wave64: bool,
    cu_mode: bool,
    global_work_size: u32,
}

impl GpuKernelVariant {
    fn build(
        api: SharedHipApi,
        block_size: u32,
        poll_interval: u32,
        wave64: bool,
        cu_mode: bool,
        global_work_size: u32,
    ) -> Result<Self> {
        let module = GpuKernelModule::build(api, block_size, poll_interval, wave64, cu_mode)?;
        Ok(Self::from_compiled(
            module,
            block_size,
            poll_interval,
            wave64,
            cu_mode,
            global_work_size,
        ))
    }

    fn from_compiled(
        module: Rc<GpuKernelModule>,
        block_size: u32,
        poll_interval: u32,
        wave64: bool,
        cu_mode: bool,
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
            wave64,
            cu_mode,
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
    wave64: bool,
    cu_mode: bool,
    global_work_size: u32,
}

pub(crate) struct HipSearchEngine {
    _api: SharedHipApi,
    device_name: String,
    base_words_buffer: DeviceBuffer,
    best_timestamp_buffer: DeviceBuffer,
    prefixes_buffer: DeviceBuffer,
    prefix_count: u32,
    max_error: u32,
    variant: GpuKernelVariant,
    batch_size: u64,
}

impl HipSearchEngine {
    pub(crate) fn new() -> Result<Self> {
        let api = Arc::new(HipApi::load()?);
        let (_device, device_name) = select_gpu_device(&api)?;

        let base_words_buffer = DeviceBuffer::new(api.clone(), std::mem::size_of::<[u32; 16]>())?;
        let best_timestamp_buffer = DeviceBuffer::new(api.clone(), std::mem::size_of::<u32>())?;
        let prefixes_buffer = DeviceBuffer::new(api.clone(), PREFIXES_BUFFER_BYTES)?;

        let variant = if let Some(cached) = load_cached_variant(&device_name) {
            match GpuKernelVariant::build(
                api.clone(),
                cached.block_size,
                cached.poll_interval,
                cached.wave64,
                cached.cu_mode,
                cached.global_work_size,
            ) {
                Ok(variant) => variant,
                Err(_) => {
                    let variant = tune_variant(
                        api.clone(),
                        &base_words_buffer,
                        &best_timestamp_buffer,
                        &prefixes_buffer,
                    )?;
                    store_cached_variant(&device_name, &variant);
                    variant
                }
            }
        } else {
            let variant = tune_variant(
                api.clone(),
                &base_words_buffer,
                &best_timestamp_buffer,
                &prefixes_buffer,
            )?;
            store_cached_variant(&device_name, &variant);
            variant
        };
        let batch_size = variant.batch_size();

        Ok(Self {
            _api: api,
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
        self.variant.api.check_hip(
            unsafe {
                (self.variant.api.hip_memcpy_htod)(
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

        self.variant.api.check_hip(
            unsafe {
                (self.variant.api.hip_memcpy_htod)(
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
        self.variant.api.check_hip(
            unsafe {
                (self.variant.api.hip_memcpy_htod)(
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
        self.variant.api.check_hip(
            unsafe {
                (self.variant.api.hip_memcpy_dtoh)(
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

fn build_code_object(
    api: SharedHipApi,
    block_size: u32,
    poll_interval: u32,
    wave64: bool,
    cu_mode: bool,
) -> Result<Vec<u8>> {
    let program = HiprtcProgramHandle::new(api.clone(), KERNEL_SOURCE, "pgp_vanity_kernel.hip")?;
    let mut options = vec![
        CString::new("--std=c++17").expect("static compile option must not contain a NUL"),
        CString::new("-O3").expect("static compile option must not contain a NUL"),
        CString::new("-DNDEBUG").expect("static compile option must not contain a NUL"),
        CString::new(format!("-D FIXED_BLOCK_SIZE={block_size}"))
            .context("failed to encode block-size compile option")?,
        CString::new(format!("-D POLL_INTERVAL={poll_interval}"))
            .context("failed to encode poll-interval compile option")?,
        CString::new(format!(
            "-D FULL_BATCH_OUTER_LOOPS={}",
            BATCH_ITERATIONS / 4
        ))
        .context("failed to encode full-batch loop-count compile option")?,
    ];
    if wave64 {
        options.push(
            CString::new("-mwavefrontsize64")
                .expect("static compile option must not contain a NUL"),
        );
    } else {
        options.push(
            CString::new("-mno-wavefrontsize64")
                .expect("static compile option must not contain a NUL"),
        );
    }
    if cu_mode {
        options
            .push(CString::new("-mcumode").expect("static compile option must not contain a NUL"));
    }
    let option_ptrs = options
        .iter()
        .map(|option| option.as_ptr())
        .collect::<Vec<_>>();

    let compile_status = unsafe {
        (api.hiprtc_compile_program)(
            program.handle,
            option_ptrs.len() as c_int,
            option_ptrs.as_ptr(),
        )
    };
    let log = program.program_log()?;
    if compile_status != HIPRTC_SUCCESS {
        let mode = if wave64 { "wave64" } else { "wave32" };
        let cu_mode_label = if cu_mode { ", cu-mode" } else { "" };
        let suffix = if log.is_empty() {
            String::new()
        } else {
            format!("\n{log}")
        };
        bail!(
            "failed to compile HIP kernel for block size {block_size}, poll interval {poll_interval}, {mode}{cu_mode_label}: {}{suffix}",
            api.hiprtc_error_string(compile_status),
        );
    }

    program.code()
}

#[allow(clippy::too_many_arguments)]
fn launch_partial_kernel(
    variant: &GpuKernelVariant,
    base_words_buffer: HipDevicePtr,
    start_timestamp: u32,
    count: u32,
    prefixes_buffer: HipDevicePtr,
    prefix_count: u32,
    max_error: u32,
    best_timestamp_buffer: HipDevicePtr,
    work_items: u32,
) -> Result<()> {
    let grid_dim_x = work_items / variant.block_size;
    ensure!(
        grid_dim_x > 0,
        "HIP grid dimension must be greater than zero"
    );

    let mut base_words_arg = base_words_buffer;
    let mut start_timestamp_arg = start_timestamp;
    let mut count_arg = count;
    let mut prefixes_arg = prefixes_buffer;
    let mut prefix_count_arg = prefix_count;
    let mut max_error_arg = max_error;
    let mut best_timestamp_arg = best_timestamp_buffer;
    let mut kernel_params = [
        (&mut base_words_arg as *mut HipDevicePtr).cast::<c_void>(),
        (&mut start_timestamp_arg as *mut u32).cast::<c_void>(),
        (&mut count_arg as *mut u32).cast::<c_void>(),
        (&mut prefixes_arg as *mut HipDevicePtr).cast::<c_void>(),
        (&mut prefix_count_arg as *mut u32).cast::<c_void>(),
        (&mut max_error_arg as *mut u32).cast::<c_void>(),
        (&mut best_timestamp_arg as *mut HipDevicePtr).cast::<c_void>(),
    ];

    variant.api.check_hip(
        unsafe {
            (variant.api.hip_module_launch_kernel)(
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
        "failed to launch HIP kernel",
    )?;
    variant.api.check_hip(
        unsafe { (variant.api.hip_device_synchronize)() },
        "HIP kernel execution failed",
    )?;
    Ok(())
}

fn launch_full_batch_kernel(
    variant: &GpuKernelVariant,
    base_words_buffer: HipDevicePtr,
    start_timestamp: u32,
    prefixes_buffer: HipDevicePtr,
    prefix_count: u32,
    max_error: u32,
    best_timestamp_buffer: HipDevicePtr,
) -> Result<()> {
    let grid_dim_x = variant.global_work_size / variant.block_size;
    ensure!(
        grid_dim_x > 0,
        "HIP grid dimension must be greater than zero"
    );

    let mut base_words_arg = base_words_buffer;
    let mut start_timestamp_arg = start_timestamp;
    let mut prefixes_arg = prefixes_buffer;
    let mut prefix_count_arg = prefix_count;
    let mut max_error_arg = max_error;
    let mut best_timestamp_arg = best_timestamp_buffer;
    let mut kernel_params = [
        (&mut base_words_arg as *mut HipDevicePtr).cast::<c_void>(),
        (&mut start_timestamp_arg as *mut u32).cast::<c_void>(),
        (&mut prefixes_arg as *mut HipDevicePtr).cast::<c_void>(),
        (&mut prefix_count_arg as *mut u32).cast::<c_void>(),
        (&mut max_error_arg as *mut u32).cast::<c_void>(),
        (&mut best_timestamp_arg as *mut HipDevicePtr).cast::<c_void>(),
    ];

    variant.api.check_hip(
        unsafe {
            (variant.api.hip_module_launch_kernel)(
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
        "failed to launch HIP full-batch kernel",
    )?;
    variant.api.check_hip(
        unsafe { (variant.api.hip_device_synchronize)() },
        "HIP full-batch kernel execution failed",
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_kernel(
    variant: &GpuKernelVariant,
    base_words_buffer: HipDevicePtr,
    start_timestamp: u32,
    count: u32,
    prefixes_buffer: HipDevicePtr,
    prefix_count: u32,
    max_error: u32,
    best_timestamp_buffer: HipDevicePtr,
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
    api: SharedHipApi,
    base_words_buffer: &DeviceBuffer,
    best_timestamp_buffer: &DeviceBuffer,
    prefixes_buffer: &DeviceBuffer,
) -> Result<GpuKernelVariant> {
    let benchmark_prefix = benchmark_prefix();
    let benchmark_words = FingerprintSearch::new(BENCHMARK_PUBLIC_KEY).base_words();
    api.check_hip(
        unsafe {
            (api.hip_memcpy_htod)(
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
    api.check_hip(
        unsafe {
            (api.hip_memcpy_htod)(
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

    for &cu_mode in &CU_MODE_CANDIDATES {
        for &wave64 in &WAVE64_CANDIDATES {
            for &block_size in &BLOCK_SIZE_CANDIDATES {
                for &poll_interval in &POLL_INTERVAL_CANDIDATES {
                    let compiled = match GpuKernelModule::build(
                        api.clone(),
                        block_size,
                        poll_interval,
                        wave64,
                        cu_mode,
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
                            wave64,
                            cu_mode,
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

                        push_finalist(&mut finalists, variant, hashes_per_second);
                    }
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
        BLOCK_SIZE_CANDIDATES[0],
        POLL_INTERVAL_CANDIDATES[0],
        false,
        false,
        GLOBAL_WORK_SIZE_CANDIDATES[0],
    )
}

#[allow(clippy::too_many_arguments)]
fn benchmark_variant_elapsed(
    variant: &GpuKernelVariant,
    base_words_buffer: HipDevicePtr,
    best_timestamp_buffer: HipDevicePtr,
    prefixes_buffer: HipDevicePtr,
    prefix_count: u32,
    start_timestamp: u32,
    count: u32,
) -> Result<Duration> {
    let not_found = [NOT_FOUND_TIMESTAMP; 1];
    variant.api.check_hip(
        unsafe {
            (variant.api.hip_memcpy_htod)(
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
    base_words_buffer: HipDevicePtr,
    best_timestamp_buffer: HipDevicePtr,
    prefixes_buffer: HipDevicePtr,
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
    base_words_buffer: HipDevicePtr,
    best_timestamp_buffer: HipDevicePtr,
    prefixes_buffer: HipDevicePtr,
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

fn load_cached_variant(device_name: &str) -> Option<CachedVariant> {
    let path = tuning_cache_path(TUNING_CACHE_FILE);
    let contents = fs::read_to_string(path).ok()?;
    let mut lines = contents.lines();

    let version = lines.next()?.parse::<u32>().ok()?;
    let cached_device_name = lines.next()?;
    let block_size = lines.next()?.parse::<u32>().ok()?;
    let poll_interval = lines.next()?.parse::<u32>().ok()?;
    let wave64 = lines.next()?.parse::<u8>().ok()? != 0;
    let cu_mode = lines.next()?.parse::<u8>().ok()? != 0;
    let global_work_size = lines.next()?.parse::<u32>().ok()?;

    if version != TUNING_CACHE_VERSION || cached_device_name != device_name {
        return None;
    }

    if !BLOCK_SIZE_CANDIDATES.contains(&block_size)
        || !POLL_INTERVAL_CANDIDATES.contains(&poll_interval)
        || !WAVE64_CANDIDATES.contains(&wave64)
        || !CU_MODE_CANDIDATES.contains(&cu_mode)
        || !GLOBAL_WORK_SIZE_CANDIDATES.contains(&global_work_size)
    {
        return None;
    }

    Some(CachedVariant {
        block_size,
        poll_interval,
        wave64,
        cu_mode,
        global_work_size,
    })
}

fn store_cached_variant(device_name: &str, variant: &GpuKernelVariant) {
    let path = tuning_cache_path(TUNING_CACHE_FILE);

    if let Some(parent) = path.parent()
        && fs::create_dir_all(parent).is_err()
    {
        return;
    }

    let contents = format!(
        "{TUNING_CACHE_VERSION}\n{device_name}\n{}\n{}\n{}\n{}\n{}\n",
        variant.block_size,
        variant.poll_interval,
        u8::from(variant.wave64),
        u8::from(variant.cu_mode),
        variant.global_work_size,
    );
    let _ = fs::write(path, contents);
}

fn select_gpu_device(api: &HipApi) -> Result<(HipDevice, String)> {
    let mut count = 0;
    api.check_hip(
        unsafe { (api.hip_get_device_count)(&mut count) },
        "failed to enumerate HIP devices",
    )?;
    if count <= 0 {
        bail!("no HIP GPU device found; install a ROCm runtime with HIP support for your AMD GPU");
    }

    let device = 0;
    api.check_hip(
        unsafe { (api.hip_set_device)(device) },
        "failed to select HIP device",
    )?;

    let mut device_name = vec![0 as c_char; 256];
    api.check_hip(
        unsafe {
            (api.hip_device_get_name)(device_name.as_mut_ptr(), device_name.len() as c_int, device)
        },
        "failed to query HIP device name",
    )?;
    let device_name = unsafe { CStr::from_ptr(device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    Ok((device, device_name))
}

fn load_library(name: &str) -> Result<Library> {
    let mut candidates = Vec::new();

    if let Some(path) = env::var_os("ROCM_LIB_DIR") {
        candidates.push(PathBuf::from(path).join(name));
    }
    if let Some(path) = env::var_os("ROCM_PATH") {
        candidates.push(PathBuf::from(path).join("lib").join(name));
    }
    if let Some(paths) = env::var_os("LD_LIBRARY_PATH") {
        candidates.extend(env::split_paths(&paths).map(|path| path.join(name)));
    }
    candidates.push(PathBuf::from("/opt/rocm/lib").join(name));

    for candidate in &candidates {
        if let Ok(library) = unsafe { Library::new(candidate) } {
            return Ok(library);
        }
    }

    unsafe { Library::new(name) }.with_context(|| {
        format!(
            "failed to find {name} via ROCM_PATH, ROCM_LIB_DIR, LD_LIBRARY_PATH, or /opt/rocm/lib"
        )
    })
}
