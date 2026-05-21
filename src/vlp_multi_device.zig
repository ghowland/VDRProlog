// ============================================================
// vlp_multi_device.zig
// Multi-device support — pipeline parallelism for large models.
// KB replication across devices.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const bridge_mod = @import("vlp_bridge.zig");
const llm_mod = @import("vlp_llm.zig");
const kb_mod = @import("vlp_kb_store.zig");

// ============================================================
// Configuration
// ============================================================

pub const PartitionStrategy = enum(i32) {
    pipeline = 0, // contiguous layer ranges per device
    tensor_parallel = 1, // each device gets slice of each layer
};

pub const MultiDeviceConfig = struct {
    n_devices: i32,
    device_ids: []const i32,
    model: llm_mod.ModelConfig,
    strategy: PartitionStrategy = .pipeline,
};

// ============================================================
// Device assignment — which layers on which device
// ============================================================

pub const DeviceAssignment = struct {
    device_id: i32,
    layer_start: i32, // inclusive
    layer_end: i32, // exclusive
    bridge: ?*bridge_mod.Bridge,
};

// ============================================================
// Multi-device manager
// ============================================================

pub const MultiDeviceManager = struct {
    allocator: std.mem.Allocator,
    assignments: []DeviceAssignment,
    bridges: []bridge_mod.Bridge,
    n_devices: i32,
    strategy: PartitionStrategy,
    model_config: llm_mod.ModelConfig,
    initialized: bool,
};

// ============================================================
// Lifecycle
// ============================================================

pub fn init(allocator: std.mem.Allocator, config: *const MultiDeviceConfig) ?*MultiDeviceManager {
    var mgr = allocator.create(MultiDeviceManager) catch return null;
    mgr.allocator = allocator;
    mgr.n_devices = config.n_devices;
    mgr.strategy = config.strategy;
    mgr.model_config = config.model;
    mgr.initialized = false;

    const n: usize = @intCast(config.n_devices);

    mgr.assignments = allocator.alloc(DeviceAssignment, n) catch {
        allocator.destroy(mgr);
        return null;
    };

    mgr.bridges = allocator.alloc(bridge_mod.Bridge, n) catch {
        allocator.free(mgr.assignments);
        allocator.destroy(mgr);
        return null;
    };

    // Compute layer assignments
    const layers_per_device = @divTrunc(config.model.n_layers, config.n_devices);
    var layer_cursor: i32 = 0;

    for (0..n) |i| {
        const start = layer_cursor;
        const end = if (i == n - 1) config.model.n_layers else start + layers_per_device;

        mgr.assignments[i] = .{
            .device_id = config.device_ids[i],
            .layer_start = start,
            .layer_end = end,
            .bridge = null,
        };

        layer_cursor = end;
    }

    // Initialize a bridge per device
    for (0..n) |i| {
        // Each device gets the same sizing but different device_id
        var sizing = @import("vlp_device_memory.zig").defaultSizingConfig();
        // Adjust model params for this device's shard
        const shard_layers = mgr.assignments[i].layer_end - mgr.assignments[i].layer_start;
        const params_per_layer = @divTrunc(config.model.totalParams(), @as(i64, config.model.n_layers));
        sizing.model_params = params_per_layer * @as(i64, shard_layers);

        const bridge_config = bridge_mod.BridgeConfig{
            .sizing = sizing,
            .shader_dir = &.{},
            .enable_validation = false,
            .preferred_device_index = config.device_ids[i],
            .force_host_visible_memory = false,
        };

        mgr.bridges[i] = bridge_mod.init(allocator, &bridge_config);
        mgr.assignments[i].bridge = &mgr.bridges[i];
    }

    mgr.initialized = true;
    return mgr;
}

pub fn deinit(mgr: *MultiDeviceManager) void {
    const n: usize = @intCast(mgr.n_devices);
    for (0..n) |i| {
        mgr.bridges[i].deinit();
    }
    mgr.allocator.free(mgr.assignments);
    mgr.allocator.free(mgr.bridges);
    mgr.initialized = false;
    mgr.allocator.destroy(mgr);
}

// ============================================================
// Multi-device forward pass — pipeline parallelism
// ============================================================

pub fn forward(mgr: *MultiDeviceManager, input_ids: []const i32, logits: []i32) types.Status {
    if (!mgr.initialized) return types.Status.err(.device, .device_not_found, 0);

    // Pipeline forward:
    // 1. Device 0: embedding + layers[0..k]
    // 2. Transfer hidden state to device 1
    // 3. Device 1: layers[k..2k]
    // 4. Transfer → device 2, etc.
    // 5. Last device: final norm + lm_head → logits

    const n: usize = @intCast(mgr.n_devices);

    // Upload input to device 0
    const input_bytes: []const u8 = @as([*]const u8, @ptrCast(input_ids.ptr))[0 .. input_ids.len * 4];
    var status = mgr.bridges[0].uploadToBuffer(.scratch_a, 0, input_bytes);
    if (status.isErr()) return status;

    // Each device processes its layers
    for (0..n) |i| {
        const assignment = &mgr.assignments[i];
        const bridge = &mgr.bridges[i];

        // If not device 0, receive hidden state from previous device
        if (i > 0) {
            status = transferHiddenState(&mgr.bridges[i - 1], bridge, &mgr.model_config, @intCast(input_ids.len));
            if (status.isErr()) return status;
        }

        // Process layers on this device
        // Simplified: dispatch layer-by-layer using this device's bridge
        var layer = assignment.layer_start;
        while (layer < assignment.layer_end) : (layer += 1) {
            // Each layer dispatch uses the same kernel sequence as LlmEngine.forwardLayer
            // but dispatched to this device's bridge
            // _ = bridge;
            // _ = layer;
            // Real implementation: create an LlmEngine per device,
            // call forwardLayer for the assigned range
        }
    }

    // Download logits from last device
    const last_bridge = &mgr.bridges[n - 1];
    const logit_bytes: []u8 = @as([*]u8, @ptrCast(logits.ptr))[0 .. logits.len * 4];
    status = last_bridge.downloadFromBuffer(.scratch_b, 0, logit_bytes);

    return status;
}

// ============================================================
// KB replication
// ============================================================

pub fn replicateKb(mgr: *MultiDeviceManager, source_device: i32, target_device: i32, kb_id: i32) types.Status {
    if (source_device < 0 or source_device >= mgr.n_devices) return types.Status.err(.device, .device_not_found, source_device);
    if (target_device < 0 or target_device >= mgr.n_devices) return types.Status.err(.device, .device_not_found, target_device);

    const src_bridge = &mgr.bridges[@intCast(source_device)];
    const dst_bridge = &mgr.bridges[@intCast(target_device)];

    // Download KB struct from source
    var kb: types.Kb = undefined;
    const kb_offset = @as(i64, kb_id) * types.KB_STRUCT_SIZE;
    const kb_bytes: []u8 = @as([*]u8, @ptrCast(&kb))[0..@sizeOf(types.Kb)];
    var status = src_bridge.downloadFromBuffer(.kb_store, kb_offset, kb_bytes);
    if (status.isErr()) return status;

    // Upload KB struct to target
    const kb_src: []const u8 = @as([*]const u8, @ptrCast(&kb))[0..@sizeOf(types.Kb)];
    status = dst_bridge.uploadToBuffer(.kb_store, kb_offset, kb_src);
    if (status.isErr()) return status;

    // Download and upload facts
    if (kb.facts_count > 0) {
        const fact_size = @as(i64, kb.facts_count) * @sizeOf(types.Fact);
        const fact_offset = @as(i64, kb.facts_offset) * @sizeOf(types.Fact);

        // Allocate staging buffer
        const staging = mgr.allocator.alloc(u8, @intCast(fact_size)) catch
            return types.Status.err(.device, .device_out_of_memory, 0);
        defer mgr.allocator.free(staging);

        status = src_bridge.downloadFromBuffer(.fact_store, fact_offset, staging);
        if (status.isErr()) return status;

        status = dst_bridge.uploadToBuffer(.fact_store, fact_offset, staging);
        if (status.isErr()) return status;
    }

    return types.Status.ok();
}

pub fn syncKb(mgr: *MultiDeviceManager, kb_id: i32) types.Status {
    // Broadcast from device 0 to all other devices
    const n: usize = @intCast(mgr.n_devices);
    var i: usize = 1;
    while (i < n) : (i += 1) {
        const status = replicateKb(mgr, 0, @intCast(i), kb_id);
        if (status.isErr()) return status;
    }
    return types.Status.ok();
}

// ============================================================
// Helpers
// ============================================================

fn transferHiddenState(src: *bridge_mod.Bridge, dst: *bridge_mod.Bridge, model: *const llm_mod.ModelConfig, n_tokens: i32) types.Status {
    // Transfer hidden state: [n_tokens × d_model] Q16 values
    // In real implementation: NVLink peer-to-peer copy
    // Fallback: download from src → upload to dst through host staging
    const size = @as(i64, n_tokens) * @as(i64, model.d_model) * 8;

    // This is a placeholder — real multi-GPU uses peer access
    _ = src;
    _ = dst;
    _ = size;
    return types.Status.ok();
}
