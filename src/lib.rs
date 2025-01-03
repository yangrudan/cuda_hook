use ctor::ctor;
use ilhook::x64::{Hooker, HookType, Registers, CallbackOption, HookFlags};
use std::os::raw::{c_void};
use libloading::{Library, Symbol};

#[repr(C)]
#[derive(Copy, Clone)]
pub struct dim3 {
    x: u32,
    y: u32,
    z: u32,
}

type CudaError = u32;

// 定义 cudaLaunchKernel 的函数签名
type CudaLaunchKernelFn = unsafe extern "win64" fn(
    func: *const c_void,
    grid_dim: dim3,
    block_dim: dim3,
    args: *mut *mut c_void,
    shared_mem: usize,
    stream: *mut c_void,
) -> CudaError;

static mut ORIGINAL_CUDA_LAUNCH_KERNEL: Option<CudaLaunchKernelFn> = None;

// Hook回调函数
unsafe extern "C" fn hook_callback(
    func: *const c_void,
    grid_dim: dim3,
    block_dim: dim3,
    args: *mut *mut c_void,
    shared_mem: usize,
    stream: *mut c_void,
) -> CudaError {
    // 打印调试信息
    println!(
        "Hooked cudaLaunchKernel called with function pointer: {:p}, grid: ({}, {}, {}), block: ({}, {}, {}), shared memory: {}, stream: {:p}",
        func,
        grid_dim.x, grid_dim.y, grid_dim.z,
        block_dim.x, block_dim.y, block_dim.z,
        shared_mem, stream
    );

    // 调用原始的 cudaLaunchKernel 函数
    let original_fn = unsafe { ORIGINAL_CUDA_LAUNCH_KERNEL.as_ref().unwrap() };
    original_fn(func, grid_dim, block_dim, args, shared_mem, stream)
}

#[ctor]
fn initialize() {
    // 动态加载 CUDA 库
    let lib = unsafe { Library::new("libcudart.so").expect("Failed to load libcudart.so") };

    // 获取原始的 cudaLaunchKernel 函数地址
    let original_fn: Symbol<CudaLaunchKernelFn> = unsafe {
        lib.get(b"cudaLaunchKernel\0").expect("Failed to find cudaLaunchKernel")
    };

    // 保存原始函数指针
    unsafe {
        ORIGINAL_CUDA_LAUNCH_KERNEL = Some(unsafe { std::mem::transmute(original_fn) });
    }

    // 创建 Hooker
    let hooker = Hooker::new(
        original_fn as usize,
        HookType::Retn(hook_callback),
        CallbackOption::None,
        0,
        HookFlags::empty(),
    );

    // 安装 hook
    let _hook_point = unsafe { hooker.hook().expect("Failed to install hook") };
}
