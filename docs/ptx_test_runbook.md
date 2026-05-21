**Testing VDR-Prolog PTX on your GTX 1070 — Windows 10**

**What you have:** Your NVIDIA gaming drivers already include `nvcuda.dll` (the CUDA Driver API). You don't need the full CUDA Toolkit just to load and launch PTX. The Driver API ships with every NVIDIA display driver since ~2008.

**Verify you have it:**

Open PowerShell and run:
```
where.exe nvcuda.dll
```
Should show `C:\Windows\System32\nvcuda.dll`. If it does, you're ready.

If not, update your NVIDIA drivers from nvidia.com — any Game Ready or Studio driver includes it.

**The test plan:**

We need a minimal C or Zig host program that does five things: initialize CUDA, load the PTX, allocate device buffers, launch the kernel with a simple op, read back results. This validates the entire pipeline from Zig source → PTX → GPU execution → integer results.

**Write this file as `src/test_ptx.zig`:**

```zig
const std = @import("std");

// CUDA Driver API types
const CUdevice = i32;
const CUcontext = ?*anyopaque;
const CUmodule = ?*anyopaque;
const CUfunction = ?*anyopaque;
const CUdeviceptr = u64;
const CUresult = c_int;
const CUDA_SUCCESS: CUresult = 0;

// CUDA Driver API imports — linked from nvcuda
extern "nvcuda" fn cuInit(flags: c_uint) CUresult;
extern "nvcuda" fn cuDeviceGet(device: *CUdevice, ordinal: c_int) CUresult;
extern "nvcuda" fn cuDeviceGetName(name: [*]u8, len: c_int, dev: CUdevice) CUresult;
extern "nvcuda" fn cuDeviceGetCount(count: *c_int) CUresult;
extern "nvcuda" fn cuCtxCreate_v2(pctx: *CUcontext, flags: c_uint, dev: CUdevice) CUresult;
extern "nvcuda" fn cuCtxDestroy_v2(ctx: CUcontext) CUresult;
extern "nvcuda" fn cuModuleLoadData(module: *CUmodule, image: [*]const u8) CUresult;
extern "nvcuda" fn cuModuleUnload(module: CUmodule) CUresult;
extern "nvcuda" fn cuModuleGetFunction(hfunc: *CUfunction, hmod: CUmodule, name: [*:0]const u8) CUresult;
extern "nvcuda" fn cuMemAlloc_v2(dptr: *CUdeviceptr, bytesize: usize) CUresult;
extern "nvcuda" fn cuMemFree_v2(dptr: CUdeviceptr) CUresult;
extern "nvcuda" fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: [*]const u8, byteCount: usize) CUresult;
extern "nvcuda" fn cuMemcpyDtoH_v2(dst: [*]u8, src: CUdeviceptr, byteCount: usize) CUresult;
extern "nvcuda" fn cuMemsetD32_v2(dptr: CUdeviceptr, ui: c_uint, n: usize) CUresult;
extern "nvcuda" fn cuLaunchKernel(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: ?*anyopaque,
    kernelParams: [*]?*anyopaque,
    extra: ?*anyopaque,
) CUresult;
extern "nvcuda" fn cuCtxSynchronize() CUresult;

// PTX source — built by build.zig
const ptx_source = @embedFile("vlp_kernel_ptx");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();

    try stdout.print("=== VDR-Prolog PTX Test ===\n\n", .{});

    // 1. Init CUDA
    var r = cuInit(0);
    if (r != CUDA_SUCCESS) {
        try stdout.print("FAIL: cuInit returned {}\n", .{r});
        return;
    }
    try stdout.print("[OK] cuInit\n", .{});

    // 2. Get device
    var dev_count: c_int = 0;
    _ = cuDeviceGetCount(&dev_count);
    try stdout.print("[OK] {} CUDA device(s) found\n", .{dev_count});

    var device: CUdevice = 0;
    r = cuDeviceGet(&device, 0);
    if (r != CUDA_SUCCESS) {
        try stdout.print("FAIL: cuDeviceGet returned {}\n", .{r});
        return;
    }

    var name_buf: [256]u8 = undefined;
    _ = cuDeviceGetName(&name_buf, 256, device);
    const name_len = std.mem.indexOfScalar(u8, &name_buf, 0) orelse 256;
    try stdout.print("[OK] Device: {s}\n", .{name_buf[0..name_len]});

    // 3. Create context
    var ctx: CUcontext = null;
    r = cuCtxCreate_v2(&ctx, 0, device);
    if (r != CUDA_SUCCESS) {
        try stdout.print("FAIL: cuCtxCreate returned {}\n", .{r});
        return;
    }
    try stdout.print("[OK] CUDA context created\n", .{});

    // 4. Load PTX module
    var module: CUmodule = null;
    r = cuModuleLoadData(&module, ptx_source.ptr);
    if (r != CUDA_SUCCESS) {
        try stdout.print("FAIL: cuModuleLoadData returned {} (PTX size: {} bytes)\n", .{ r, ptx_source.len });
        _ = cuCtxDestroy_v2(ctx);
        return;
    }
    try stdout.print("[OK] PTX module loaded ({} bytes)\n", .{ptx_source.len});

    // 5. Get kernel function
    var func: CUfunction = null;
    r = cuModuleGetFunction(&func, module, "vlp_kernel_$_main");
    if (r != CUDA_SUCCESS) {
        try stdout.print("FAIL: cuModuleGetFunction returned {}\n", .{r});
        _ = cuModuleUnload(module);
        _ = cuCtxDestroy_v2(ctx);
        return;
    }
    try stdout.print("[OK] Kernel function found: vlp_kernel_$_main\n", .{});

    // 6. Allocate 15 device buffers (matching kernel signature)
    const BUFFER_COUNT = 15;
    const BUF_SIZE: usize = 4096; // 1024 ints × 4 bytes — small for test
    var device_ptrs: [BUFFER_COUNT]CUdeviceptr = .{0} ** BUFFER_COUNT;

    for (0..BUFFER_COUNT) |i| {
        r = cuMemAlloc_v2(&device_ptrs[i], BUF_SIZE);
        if (r != CUDA_SUCCESS) {
            try stdout.print("FAIL: cuMemAlloc buffer {} returned {}\n", .{ i, r });
            cleanup(ctx, module, &device_ptrs);
            return;
        }
        _ = cuMemsetD32_v2(device_ptrs[i], 0, BUF_SIZE / 4);
    }
    try stdout.print("[OK] {} device buffers allocated ({} bytes each)\n", .{ BUFFER_COUNT, BUF_SIZE });

    // Buffer indices (matching vlp_bridge.zig BufferTarget):
    // 9 = scratch_a, 10 = scratch_b, 12 = params

    // ============================================================
    // TEST 1: residual_add (op 10) — simplest op
    // scratch_a[i] += scratch_b[i]
    // ============================================================
    try stdout.print("\n--- Test 1: residual_add ---\n", .{});
    {
        const N = 8;

        // scratch_a = [10, 20, 30, 40, 50, 60, 70, 80]
        var a_data = [_]i32{ 10, 20, 30, 40, 50, 60, 70, 80 };
        r = cuMemcpyHtoD_v2(device_ptrs[9], @ptrCast(&a_data), N * 4);
        if (r != CUDA_SUCCESS) {
            try stdout.print("FAIL: upload scratch_a returned {}\n", .{r});
            cleanup(ctx, module, &device_ptrs);
            return;
        }

        // scratch_b = [1, 2, 3, 4, 5, 6, 7, 8]
        var b_data = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        r = cuMemcpyHtoD_v2(device_ptrs[10], @ptrCast(&b_data), N * 4);
        if (r != CUDA_SUCCESS) {
            try stdout.print("FAIL: upload scratch_b returned {}\n", .{r});
            cleanup(ctx, module, &device_ptrs);
            return;
        }

        // params: op_code=10 (residual_add), F0=8 (n_elements)
        var params_data = [_]i32{0} ** 64;
        params_data[0] = 10; // P_OP_CODE = residual_add
        params_data[1] = N; // P_FIELD_0 = n_elements
        r = cuMemcpyHtoD_v2(device_ptrs[12], @ptrCast(&params_data), 64 * 4);
        if (r != CUDA_SUCCESS) {
            try stdout.print("FAIL: upload params returned {}\n", .{r});
            cleanup(ctx, module, &device_ptrs);
            return;
        }

        // Build kernel args — 15 device pointers
        var arg_ptrs: [BUFFER_COUNT]CUdeviceptr = device_ptrs;
        var kernel_args: [BUFFER_COUNT]?*anyopaque = undefined;
        for (0..BUFFER_COUNT) |i| {
            kernel_args[i] = @ptrCast(&arg_ptrs[i]);
        }

        // Launch: 1 block × 256 threads
        r = cuLaunchKernel(
            func,
            1, 1, 1, // grid
            256, 1, 1, // block
            256 * (4 + 8 + 4), // shared memory
            null, // default stream
            &kernel_args,
            null,
        );
        if (r != CUDA_SUCCESS) {
            try stdout.print("FAIL: cuLaunchKernel returned {}\n", .{r});
            cleanup(ctx, module, &device_ptrs);
            return;
        }

        r = cuCtxSynchronize();
        if (r != CUDA_SUCCESS) {
            try stdout.print("FAIL: cuCtxSynchronize returned {}\n", .{r});
            cleanup(ctx, module, &device_ptrs);
            return;
        }

        // Read back scratch_a (residual_add writes back to scratch_a)
        var result: [N]i32 = undefined;
        r = cuMemcpyDtoH_v2(@ptrCast(&result), device_ptrs[9], N * 4);
        if (r != CUDA_SUCCESS) {
            try stdout.print("FAIL: download returned {}\n", .{r});
            cleanup(ctx, module, &device_ptrs);
            return;
        }

        // Expected: [11, 22, 33, 44, 55, 66, 77, 88]
        const expected = [_]i32{ 11, 22, 33, 44, 55, 66, 77, 88 };
        var pass = true;
        for (0..N) |i| {
            if (result[i] != expected[i]) {
                try stdout.print("  FAIL: result[{}] = {} (expected {})\n", .{ i, result[i], expected[i] });
                pass = false;
            }
        }
        if (pass) {
            try stdout.print("[PASS] residual_add: {any}\n", .{result});
        }
    }

    // ============================================================
    // TEST 2: builtin_unary abs (op 19, sub_op 0)
    // scratch_b[i] = abs(scratch_a[i])
    // ============================================================
    try stdout.print("\n--- Test 2: builtin_unary abs ---\n", .{});
    {
        const N = 4;

        var a_data = [_]i32{ -100, 200, -300, 0 };
        _ = cuMemcpyHtoD_v2(device_ptrs[9], @ptrCast(&a_data), N * 4);

        var params_data = [_]i32{0} ** 64;
        params_data[0] = 19; // builtin_unary
        params_data[1] = N; // n_elements
        params_data[2] = 0; // sub_op = abs
        params_data[3] = 0; // input_offset
        params_data[4] = 0; // output_offset
        _ = cuMemcpyHtoD_v2(device_ptrs[12], @ptrCast(&params_data), 64 * 4);

        var arg_ptrs: [BUFFER_COUNT]CUdeviceptr = device_ptrs;
        var kernel_args: [BUFFER_COUNT]?*anyopaque = undefined;
        for (0..BUFFER_COUNT) |i| kernel_args[i] = @ptrCast(&arg_ptrs[i]);

        _ = cuLaunchKernel(func, 1, 1, 1, 256, 1, 1, 256 * (4 + 8 + 4), null, &kernel_args, null);
        _ = cuCtxSynchronize();

        var result: [N]i32 = undefined;
        _ = cuMemcpyDtoH_v2(@ptrCast(&result), device_ptrs[10], N * 4);

        const expected = [_]i32{ 100, 200, 300, 0 };
        var pass = true;
        for (0..N) |i| {
            if (result[i] != expected[i]) {
                try stdout.print("  FAIL: result[{}] = {} (expected {})\n", .{ i, result[i], expected[i] });
                pass = false;
            }
        }
        if (pass) {
            try stdout.print("[PASS] abs: {any}\n", .{result});
        }
    }

    // ============================================================
    // TEST 3: builtin_binary add (op 20, sub_op 0)
    // scratch_b[i] = scratch_a[in_a + i] + scratch_a[in_b + i]
    // ============================================================
    try stdout.print("\n--- Test 3: builtin_binary add ---\n", .{});
    {
        const N = 4;

        // Put both inputs in scratch_a: a at offset 0, b at offset 4
        var ab_data = [_]i32{ 100, 200, 300, 400, 1, 2, 3, 4 };
        _ = cuMemcpyHtoD_v2(device_ptrs[9], @ptrCast(&ab_data), 8 * 4);

        var params_data = [_]i32{0} ** 64;
        params_data[0] = 20; // builtin_binary
        params_data[1] = N; // n_elements
        params_data[2] = 0; // sub_op = add
        params_data[3] = 0; // in_a_offset
        params_data[4] = N; // in_b_offset
        params_data[5] = 0; // output_offset
        _ = cuMemcpyHtoD_v2(device_ptrs[12], @ptrCast(&params_data), 64 * 4);

        var arg_ptrs: [BUFFER_COUNT]CUdeviceptr = device_ptrs;
        var kernel_args: [BUFFER_COUNT]?*anyopaque = undefined;
        for (0..BUFFER_COUNT) |i| kernel_args[i] = @ptrCast(&arg_ptrs[i]);

        _ = cuLaunchKernel(func, 1, 1, 1, 256, 1, 1, 256 * (4 + 8 + 4), null, &kernel_args, null);
        _ = cuCtxSynchronize();

        var result: [N]i32 = undefined;
        _ = cuMemcpyDtoH_v2(@ptrCast(&result), device_ptrs[10], N * 4);

        const expected = [_]i32{ 101, 202, 303, 404 };
        var pass = true;
        for (0..N) |i| {
            if (result[i] != expected[i]) {
                try stdout.print("  FAIL: result[{}] = {} (expected {})\n", .{ i, result[i], expected[i] });
                pass = false;
            }
        }
        if (pass) {
            try stdout.print("[PASS] binary add: {any}\n", .{result});
        }
    }

    // ============================================================
    // TEST 4: softmax_exact (op 4) — the critical test
    // Verify output sums to D=65536 exactly
    // ============================================================
    try stdout.print("\n--- Test 4: softmax_exact (unity test) ---\n", .{});
    {
        const ROW_LEN = 5;

        // Input logits in scratch_a
        var logits = [_]i32{ 10000, 20000, 5000, 30000, 15000 };
        _ = cuMemcpyHtoD_v2(device_ptrs[9], @ptrCast(&logits), ROW_LEN * 4);

        var params_data = [_]i32{0} ** 64;
        params_data[0] = 4; // softmax_exact
        params_data[1] = ROW_LEN; // row_length
        params_data[2] = 1; // n_rows
        params_data[3] = 65536; // denominator = D
        _ = cuMemcpyHtoD_v2(device_ptrs[12], @ptrCast(&params_data), 64 * 4);

        var arg_ptrs: [BUFFER_COUNT]CUdeviceptr = device_ptrs;
        var kernel_args: [BUFFER_COUNT]?*anyopaque = undefined;
        for (0..BUFFER_COUNT) |i| kernel_args[i] = @ptrCast(&arg_ptrs[i]);

        _ = cuLaunchKernel(func, 1, 1, 1, 256, 1, 1, 256 * (4 + 8 + 4), null, &kernel_args, null);
        _ = cuCtxSynchronize();

        // Read back from scratch_a (softmax writes in-place)
        var result: [ROW_LEN]i32 = undefined;
        _ = cuMemcpyDtoH_v2(@ptrCast(&result), device_ptrs[9], ROW_LEN * 4);

        var sum: i64 = 0;
        for (0..ROW_LEN) |i| {
            sum += result[i];
        }

        try stdout.print("  probs: {any}\n", .{result});
        try stdout.print("  sum:   {} (expected 65536)\n", .{sum});

        if (sum == 65536) {
            try stdout.print("[PASS] softmax sums to D exactly\n", .{});
        } else {
            try stdout.print("[FAIL] softmax sum = {} != 65536\n", .{sum});
        }

        // Verify all probs are non-negative
        var all_positive = true;
        for (result) |v| {
            if (v < 0) all_positive = false;
        }
        if (all_positive) {
            try stdout.print("[PASS] all probabilities non-negative\n", .{});
        } else {
            try stdout.print("[FAIL] negative probability found\n", .{});
        }
    }

    // ============================================================
    // TEST 5: determinism — run softmax twice, compare bit-for-bit
    // ============================================================
    try stdout.print("\n--- Test 5: determinism ---\n", .{});
    {
        const ROW_LEN = 5;
        var logits = [_]i32{ 10000, 20000, 5000, 30000, 15000 };
        var params_data = [_]i32{0} ** 64;
        params_data[0] = 4;
        params_data[1] = ROW_LEN;
        params_data[2] = 1;
        params_data[3] = 65536;

        var results: [2][ROW_LEN]i32 = undefined;

        for (0..2) |run| {
            _ = cuMemcpyHtoD_v2(device_ptrs[9], @ptrCast(&logits), ROW_LEN * 4);
            _ = cuMemcpyHtoD_v2(device_ptrs[12], @ptrCast(&params_data), 64 * 4);

            var arg_ptrs: [BUFFER_COUNT]CUdeviceptr = device_ptrs;
            var kernel_args: [BUFFER_COUNT]?*anyopaque = undefined;
            for (0..BUFFER_COUNT) |i| kernel_args[i] = @ptrCast(&arg_ptrs[i]);

            _ = cuLaunchKernel(func, 1, 1, 1, 256, 1, 1, 256 * (4 + 8 + 4), null, &kernel_args, null);
            _ = cuCtxSynchronize();

            _ = cuMemcpyDtoH_v2(@ptrCast(&results[run]), device_ptrs[9], ROW_LEN * 4);
        }

        var identical = true;
        for (0..ROW_LEN) |i| {
            if (results[0][i] != results[1][i]) {
                try stdout.print("  FAIL: position {} differs: {} vs {}\n", .{ i, results[0][i], results[1][i] });
                identical = false;
            }
        }
        if (identical) {
            try stdout.print("[PASS] bit-identical across 2 runs\n", .{});
        }
    }

    // ============================================================
    // Summary
    // ============================================================
    try stdout.print("\n=== All tests complete ===\n", .{});

    // Cleanup
    cleanup(ctx, module, &device_ptrs);
}

fn cleanup(ctx: CUcontext, module: CUmodule, ptrs: *[15]CUdeviceptr) void {
    for (ptrs) |p| {
        if (p != 0) _ = cuMemFree_v2(p);
    }
    if (module != null) _ = cuModuleUnload(module);
    if (ctx != null) _ = cuCtxDestroy_v2(ctx);
}
```

**Update `build.zig`** to build both the kernel and the test binary:

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ============================================================
    // Step 1: Build PTX kernel
    // ============================================================

    const ptx_target = b.resolveTargetQuery(.{
        .cpu_arch = .nvptx64,
        .os_tag = .freestanding,
        .cpu_model = .{ .explicit = &std.Target.nvptx.cpu.sm_61 },
    });

    const gpu_shared_kernel = b.createModule(.{
        .root_source_file = b.path("src/vlp_gpu_shared.zig"),
        .target = ptx_target,
        .optimize = .Debug,
    });

    const kernel_module = b.createModule(.{
        .root_source_file = b.path("src/vlp_kernel.zig"),
        .target = ptx_target,
        .optimize = .Debug,
    });
    kernel_module.addImport("vlp_gpu_shared", gpu_shared_kernel);

    const ptx_kernel = b.addObject(.{
        .name = "vlp_kernel",
        .root_module = kernel_module,
    });

    const install_ptx = b.addInstallFile(ptx_kernel.getEmittedAsm(), "vlp_kernel.ptx");
    b.getInstallStep().dependOn(&install_ptx.step);

    // ============================================================
    // Step 2: Build test binary
    // ============================================================

    const test_module = b.createModule(.{
        .root_source_file = b.path("src/test_ptx.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Embed PTX into test binary
    test_module.addAnonymousImport("vlp_kernel_ptx", .{
        .root_source_file = ptx_kernel.getEmittedAsm(),
    });

    const test_exe = b.addExecutable(.{
        .name = "test_ptx",
        .root_module = test_module,
    });

    test_exe.linkLibC();

    b.installArtifact(test_exe);

    // Run step
    const run_cmd = b.addRunArtifact(test_exe);
    const run_step = b.step("test", "Run PTX GPU tests");
    run_step.dependOn(&run_cmd.step);
}
```

**To run:**

```bash
# Build everything (kernel + test binary)
zig build

# Run the test
zig build test
```

Or manually:
```bash
zig build
./zig-out/bin/test_ptx.exe
```

**Expected output on your 1070:**
```
=== VDR-Prolog PTX Test ===

[OK] cuInit
[OK] 1 CUDA device(s) found
[OK] Device: GeForce GTX 1070
[OK] CUDA context created
[OK] PTX module loaded (455000 bytes)
[OK] Kernel function found: vlp_kernel_$_main
[OK] 15 device buffers allocated (4096 bytes each)

--- Test 1: residual_add ---
[PASS] residual_add: { 11, 22, 33, 44, 55, 66, 77, 88 }

--- Test 2: builtin_unary abs ---
[PASS] abs: { 100, 200, 300, 0 }

--- Test 3: builtin_binary add ---
[PASS] binary add: { 101, 202, 303, 404 }

--- Test 4: softmax_exact (unity test) ---
  probs: { ... }
  sum:   65536 (expected 65536)
[PASS] softmax sums to D exactly
[PASS] all probabilities non-negative

--- Test 5: determinism ---
[PASS] bit-identical across 2 runs

=== All tests complete ===
```

Test 4 is the one that matters most — exact unity softmax on actual GPU hardware, in integer, no float, deterministic.
