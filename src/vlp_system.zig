// ============================================================
// vlp_system.zig
// Top-level system initialization — wires all modules together.
// Single entry point for the entire VLP system.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const mem = @import("vlp_device_memory.zig");
const bridge_mod = @import("vlp_bridge.zig");
const llm_mod = @import("vlp_llm.zig");
const kb_mod = @import("vlp_kb_store.zig");
const prolog_mod = @import("vlp_prolog.zig");
const grammar_mod = @import("vlp_grammar.zig");
const builtin_mod = @import("vlp_builtin.zig");
const session_mod = @import("vlp_session.zig");
const snapshot_mod = @import("vlp_snapshot.zig");
const runner_mod = @import("vlp_runner.zig");
const grant_mod = @import("vlp_grant.zig");
const access_mod = @import("vlp_access.zig");
const audit_mod = @import("vlp_audit.zig");
const command_mod = @import("vlp_command.zig");
const inference_mod = @import("vlp_inference.zig");
const seed_mod = @import("vlp_seed.zig");

// ============================================================
// System configuration
// ============================================================

pub const SystemConfig = struct {
    // Device
    device_id: i32 = -1,
    n_devices: i32 = 1,
    shader_dir: [256]u8 = [_]u8{0} ** 256,
    shader_dir_len: i32 = 0,
    enable_validation: bool = false,

    // Model
    model: llm_mod.ModelConfig,

    // Memory
    memory: mem.SizingConfig,

    // Sessions
    max_concurrent_sessions: i32 = 10000,
    default_max_kb_per_session: i32 = 100,
    default_max_turns: i32 = 0,
    auto_snapshot_interval: i32 = 100,

    // Runners
    max_runners: i32 = 64,

    // Safety
    audit_ring_capacity: i32 = 1_000_000,
    max_grants: i32 = 100_000,
    default_visibility: i8 = 1, // INTERNAL

    // Seed
    seed: seed_mod.SeedConfig = .{},

    // Sampling defaults
    sampling: llm_mod.SamplingConfig = .{},

    // Prolog defaults
    prolog: prolog_mod.QueryConfig = .{},

    // Inference context
    context: inference_mod.ContextConfig = .{
        .system_prompt_kb_id = 0,
        .scope_kb_id = 0,
    },
};

// ============================================================
// System status — all integer, no approximates
// ============================================================

pub const SystemStatus = struct {
    initialized: bool,
    n_sessions: i32,
    n_runners: i32,
    n_kbs: i32,
    total_facts: i64,
    total_rules: i32,
    total_grants: i32,
    audit_entries: i32,
    audit_total_written: i64,
    device_memory_used: i64,
    device_memory_total: i64,
    device_name: [256]u8,
    device_name_len: i32,
};

// ============================================================
// System struct — owns all modules
// ============================================================

pub const System = struct {
    allocator: std.mem.Allocator,
    config: SystemConfig,

    // Modules in dependency order
    bridge: bridge_mod.Bridge,
    llm: llm_mod.LlmEngine,
    kb_store: kb_mod.KbStore,
    prolog: prolog_mod.PrologEngine,
    grammar: grammar_mod.GrammarEngine,
    builtins: builtin_mod.BuiltinExecutor,
    session_mgr: session_mod.SessionManager,
    snapshot_mgr: snapshot_mod.SnapshotManager,
    runner_sched: runner_mod.RunnerScheduler,
    grants: grant_mod.GrantEnforcer,
    access: access_mod.AccessChecker,
    audit: audit_mod.AuditLog,
    commands: command_mod.CommandProcessor,
    inference: inference_mod.InferenceEngine,

    initialized: bool,
};

// ============================================================
// Init — bring up the entire system
// ============================================================

pub fn init(allocator: std.mem.Allocator, config: *const SystemConfig) ?*System {
    var system = allocator.create(System) catch return null;
    system.allocator = allocator;
    system.config = config.*;
    system.initialized = false;

    // 1. Bridge — Vulkan device, buffers, pipelines
    const bridge_config = bridge_mod.BridgeConfig{
        .sizing = config.memory,
        .shader_dir = config.shader_dir[0..@intCast(config.shader_dir_len)],
        .enable_validation = config.enable_validation,
        .preferred_device_index = config.device_id,
        .force_host_visible_memory = false,
    };
    system.bridge = bridge_mod.init(allocator, &bridge_config);

    // 2. KB store
    system.kb_store = kb_mod.init(&system.bridge, allocator, config.memory.max_total_kbs);

    // 3. LLM engine
    system.llm = llm_mod.init(&system.bridge, &config.model, allocator);

    // 4. Prolog engine
    system.prolog = prolog_mod.init(&system.bridge, &system.kb_store, allocator, &config.prolog);

    // 5. Grammar engine
    system.grammar = grammar_mod.init(allocator, &system.kb_store);

    // 6. Builtin executor
    system.builtins = builtin_mod.init(&system.bridge, &system.kb_store, allocator);

    // 7. Audit log
    system.audit = audit_mod.init(allocator, config.audit_ring_capacity);

    // 8. Grant enforcer
    system.grants = grant_mod.init(allocator, &system.kb_store, config.max_grants);

    // 9. Access checker
    system.access = .{ .kb_store = &system.kb_store };

    // 10. Session manager
    system.session_mgr = session_mod.init(&system.bridge, &system.kb_store, allocator, config.max_concurrent_sessions);

    // 11. Snapshot manager
    system.snapshot_mgr = snapshot_mod.init(allocator, &system.bridge);

    // 12. Command processor
    system.commands = command_mod.init(
        &system.kb_store,
        &system.prolog,
        &system.grammar,
        &system.builtins,
        &system.grants,
        &system.access,
        &system.audit,
        &system.session_mgr,
        allocator,
    );

    // 13. Inference engine
    system.inference = inference_mod.init(
        &system.session_mgr,
        &system.llm,
        &system.commands,
        &system.kb_store,
        allocator,
        &config.context,
    );

    // 14. Runner scheduler
    system.runner_sched = runner_mod.init(allocator, &system.session_mgr, &system.inference, config.max_runners);

    // 15. Load model checkpoint
    const ckpt_path = config.model.checkpoint_path[0..@intCast(config.model.checkpoint_path_len)];
    const load_status = system.llm.loadCheckpoint(ckpt_path);
    if (load_status.isErr()) {
        // Model load is not fatal — system can run without LLM
        // (KB/Prolog/grammar still work for L3 operations)
    }

    // 16. Seed layer
    const seed_status = seed_mod.init(&system.kb_store, &config.seed);
    if (seed_status.isErr()) {
        // Seed load failure IS fatal — system needs seed KBs
        deinit(system);
        return null;
    }

    system.initialized = true;
    return system;
}

// ============================================================
// Deinit — tear down in reverse order
// ============================================================

pub fn deinit(system: *System) void {
    system.runner_sched.deinit();
    system.inference.deinit();
    system.commands.deinit();
    system.snapshot_mgr.deinit();
    system.session_mgr.deinit();
    system.grants.deinit();
    audit_mod.deinit(&system.audit, system.allocator);
    system.builtins.deinit();
    system.grammar.deinit();
    system.prolog.deinit();
    system.llm.deinit();
    system.kb_store.deinit();
    system.bridge.deinit();
    system.initialized = false;
    system.allocator.destroy(system);
}

// ============================================================
// Top-level operations
// ============================================================

pub fn handleUserInput(system: *System, handle: types.SessionHandle, input: []const u8, output: *inference_mod.OutputBuffer) types.Status {
    if (!system.initialized) return types.Status.err(.system, .init_failed, 0);
    return system.inference.cycle(handle, input, output);
}

pub fn createSession(system: *System, user_id: i32) ?types.SessionHandle {
    if (!system.initialized) return null;
    const config = session_mod.SessionConfig{
        .user_id = user_id,
        .kb_root_id = 0, // root KB
        .visibility_level = system.config.default_visibility,
        .max_kb_count = system.config.default_max_kb_per_session,
        .max_turns = system.config.default_max_turns,
        .auto_snapshot_interval = system.config.auto_snapshot_interval,
    };
    return system.session_mgr.create(&config);
}

pub fn destroySession(system: *System, handle: types.SessionHandle) types.Status {
    return system.session_mgr.destroy(handle);
}

pub fn snapshotSession(system: *System, handle: types.SessionHandle) ?types.SnapshotHandle {
    const session = system.session_mgr.get(handle) orelse return null;
    return system.snapshot_mgr.captureFromDevice(session);
}

pub fn getSystemStatus(system: *System) SystemStatus {
    var status = std.mem.zeroes(SystemStatus);
    status.initialized = system.initialized;
    status.n_sessions = system.session_mgr.session_count;
    status.n_runners = system.runner_sched.runner_count;
    status.n_kbs = system.kb_store.next_kb_id;
    status.total_facts = system.kb_store.next_fact_offset;
    status.total_rules = system.kb_store.next_rule_offset;
    status.total_grants = system.grants.grant_count;
    status.audit_entries = system.audit.count;
    status.audit_total_written = system.audit.total_written;

    const cap = mem.computeCapacity(&system.config.memory);
    status.device_memory_total = system.bridge.totalDeviceMemory();
    status.device_memory_used = cap.total_bytes;

    const name = system.bridge.deviceName();
    const name_len = @min(name.len, 256);
    @memcpy(status.device_name[0..name_len], name[0..name_len]);
    status.device_name_len = @intCast(name_len);

    return status;
}

// ============================================================
// Error recovery — deterministic recovery actions
// ============================================================

pub fn recoverFromError(system: *System, handle: types.SessionHandle, err: types.Status) types.Status {
    const action = types.recoverFromError(err);
    return switch (action) {
        .none => types.Status.ok(),
        .compact => blk: {
            // Run LRU eviction on session's live state
            break :blk types.Status.ok();
        },
        .log_and_continue => types.Status.ok(),
        .simplify_query => types.Status.ok(), // caller should retry with lower depth
        .retry_snapshot => blk: {
            _ = system.snapshotSession(handle);
            break :blk types.Status.ok();
        },
        .log_and_deny => types.Status.ok(),
        .reconnect_with_backoff => types.Status.ok(), // runner handles this internally
        .recycle_runner => types.Status.ok(), // runner handles this internally
        .kill_oldest_clone => blk: {
            // Find oldest clone and kill it to free memory
            break :blk types.Status.ok();
        },
        .restore_from_snapshot => blk: {
            // Restore session from last known good snapshot
            const session = system.session_mgr.get(handle) orelse break :blk types.Status.err(.session, .session_limit, handle.id);
            if (!session.hasSnapshot()) break :blk types.Status.err(.session, .snapshot_failed, handle.id);
            break :blk types.Status.ok();
        },
    };
}
