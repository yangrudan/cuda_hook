use ctor::ctor;
use ilhook::x64::{Hooker, HookType, Registers, CallbackOption, HookFlags};
use std::os::raw::{c_void};
use libloading::{Library};

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
unsafe extern "win64" fn hook_callback(
    reg: *mut Registers,
    ori_func: usize,
    _user_data: usize,
) -> usize {
    // 获取原始参数
    // 获取原始参数
    let func = (*reg).rdi as *const c_void; // 第一个参数通过 rdi 传递
    let grid_dim_ptr = (*reg).rsi as *const dim3; // 第二个参数通过 rsi 传递
    let block_dim_ptr = (*reg).rdx as *const dim3; // 第三个参数通过 rdx 传递
    let args = (*reg).rcx as *mut *mut c_void; // 第四个参数通过 rcx 传递
    let shared_mem = (*reg).r8 as usize; // 第五个参数通过 r8 传递
    let stream = (*reg).r9 as *mut c_void; // 第六个参数通过 r9 传递

    // 打印调试信息
    println!(
        "Hooked cudaLaunchKernel called with function pointer: {:p}, grid: ({}, {}, {}), block: ({}, {}, {}), shared memory: {}, stream: {:p}",
        func, 
        (*grid_dim_ptr).x, (*grid_dim_ptr).y, (*grid_dim_ptr).z,
        (*block_dim_ptr).x, (*block_dim_ptr).y, (*block_dim_ptr).z,
        shared_mem, stream
    );

    // 调用原始的 cudaLaunchKernel 函数
    let original_fn = std::mem::transmute::<usize, CudaLaunchKernelFn>(ori_func);
    let result = original_fn(func, *grid_dim_ptr, *block_dim_ptr, args, shared_mem, stream);

    // 返回原始结果
    result as usize
}

#[ctor]
fn initialize() {
    // 动态加载 CUDA 库
    let lib = unsafe { Library::new("libcudart.so").expect("Failed to load libcudart.so") };

    // 获取原始的 cudaLaunchKernel 函数地址
    let original_fn: CudaLaunchKernelFn = unsafe {
        *lib.get(b"cudaLaunchKernel\0").expect("Failed to find cudaLaunchKernel")
    };

    // 保存原始函数指针
    unsafe {
        ORIGINAL_CUDA_LAUNCH_KERNEL = Some(std::mem::transmute(original_fn));
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
