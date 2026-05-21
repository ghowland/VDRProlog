## Zig 0.16.0 GPU Status Report

### PTX (NVIDIA CUDA) — WORKS

**Build configuration that succeeds:**
- Target: `nvptx64-freestanding` (not `nvptx64-cuda` — cuda pulls in start.zig)
- CPU model: `sm_75` (must be explicit, `baseline` fails)
- Optimize: `Debug` (ReleaseFast strips dead code)
- Entry point: `pub fn main` with `callconv(.kernel)`
- No `export` keyword (causes LLVM alias error on kernel functions)
- Retain function: `comptime { _ = &main; }`

**What works in PTX kernels:**
- `callconv(.kernel)` → emits `.entry` in PTX
- Kernel parameters as function args (pointers + scalars)
- Inline asm for thread/block ID (`%tid.x`, `%ctaid.x`, `%ntid.x`)
- Integer arithmetic (i32 add, multiply)
- Array indexing via pointer + computed offset
- Bounds checking (Zig emits overflow/bounds checks in Debug)
- Function calls within kernel (blockIdX, threadIdX as separate `.func`)

**What the output shows:**
- `.entry ptx_test_kernel_$_main` — valid CUDA kernel entry point
- Parameters: two 64-bit pointers + one 32-bit int
- Thread ID via inline asm works correctly
- Multiplication, addition, comparison all present
- Debug mode includes integer overflow checks + panic stubs

**What to do for production:**
- Build with `ReleaseFast` but keep `comptime { _ = &main; }` to prevent stripping
- Or build with `Debug` and accept the overflow checks (they're safety nets)
- The `$` in symbol names (`ptx_test_kernel_$_main`) is Zig's name mangling — CUDA driver API loads by name, so use the mangled name or find a way to control it

### SPIR-V (Vulkan) — DOES NOT WORK on 0.16.0

**Failures:**
- `@setExecProperty` — does not exist
- `gpu.executionMode()` — cannot emit `OpExecutionMode` from inline asm ("cannot set execution mode in assembly")
- `gpu.binding()` — does not exist in 0.16.0 std.gpu
- `@fence(.workgroup)` — does not exist
- `NotOpenForWriting` — Windows linker cannot write .spv files (ziglang/zig#23883)
- `AccessDenied` — Linux zig on WSL2 Windows-mounted filesystem fails on cache rename

### Recommendation

**Use PTX path.** Port `vlp_kernel.zig` to PTX calling conventions:
- `extern var addrspace(.storage_buffer)` → kernel function parameters
- `gpu.global_invocation_id` → inline asm `%tid.x` + `%ctaid.x * %ntid.x`
- `@atomicRmw` → test if it works, else inline asm `atom.global.add`
- Workgroup barrier → inline asm `bar.sync 0;`
- Shared memory → `addrspace(.shared)` (same as SPIR-V, test if it works)
- Host bridge → CUDA Driver API (`cuModuleLoad`, `cuLaunchKernel`) instead of Vulkan
- NVIDIA only — but GCP T4/L4/H100 are all NVIDIA

### Files that change for PTX vs SPIR-V

| File | Change |
|---|---|
| `vlp_kernel.zig` | Rewrite: params as fn args, inline asm for IDs/barriers |
| `vlp_gpu_shared.zig` | Minor: remove descriptor set constants, keep everything else |
| `vlp_bridge.zig` | Rewrite: CUDA Driver API instead of Vulkan |
| `vlp_gpu_params.zig` | Simplify: params passed as kernel args, not uniform buffer |
| `build.zig` | Change target to nvptx64-freestanding |
| 20 host modules | Find-replace `bridge.dispatch()` call signature |
