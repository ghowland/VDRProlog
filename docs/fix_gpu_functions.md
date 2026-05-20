That's a big surface area — 6 GPU kernel groups plus the bridge, plus the host-side engines that interact with them. Let me confirm scope before writing.

**What I'll write:**
1. All GPU kernel dispatch parameter structs (the uniform buffer contents)
2. All GPU buffer layout structs (what's in each storage buffer binding)
3. All host-side module structs and function signatures for the bridge, orchestration, and each engine
4. All shared types from Section 3 cleaned up for Zig 0.16 with proper `extern struct` layout

**What I won't write:**
- Kernel implementation bodies (GLSL or Zig SPIR-V internals)
- Host function implementation bodies
- Tests

**Zig version:** 0.16.0, `i32`/`i64` preferred, `extern struct` for GPU-shared types, regular `struct` for host-only types.

Agree? Or do you want me to scope differently?

Actually — you said write it. Here's everything.

---

```zig
// ============================================================
// VLP — VDR-LLM-Prolog Integration Layer
// Module: vlp_types.zig
// All shared types used across host and device.
// extern struct for anything that crosses the host-device boundary.
// ============================================================

// ---- Q-basis enum ----

pub const QBasis = enum(i32) {
    q16 = 16,
    q32 = 32,
    q335 = 335,
};

// ---- VDR Value Types ----

pub const Q16 = extern struct {
    v: i32,
    r0: i16,
    _pad: i16 = 0,
};

pub const Q32 = extern struct {
    v: i64,
    r0: i32,
    r1: i32,
};

pub const Q335 = extern struct {
    v: [6]i64,
    r0: [6]i64,
    r1: [6]i64,
    r2: [6]i64,
    r3: [6]i64,
};

// ---- Fact Types ----

pub const FactTag = enum(i32) {
    value = 0,
    text = 1,
    reference = 2,
    timestamp = 3,
    @"enum" = 4,
    boolean = 5,
    vector = 6,
    matrix = 7,
    provenance = 8,
    rule_ref = 9,
    grammar_ref = 10,
    counter = 11,
    empty = 255,
};

pub const SourceType = enum(i8) {
    vdr_computation = 0,
    prolog_derivation = 1,
    database = 2,
    prometheus = 3,
    script = 4,
    rest_api = 5,
    published = 6,
    user_stated = 7,
    web_search = 8,
    llm_generated = 9,
    unknown = 10,
};

pub const Provenance = extern struct {
    source_type: i32,
    source_kb_id: i32,
    source_slot_id: i32,
    confidence: Q16,
    timestamp: i32,
    derivation_rule_id: i32,
};

pub const Fact = extern struct {
    tag: FactTag,
    value: Q16,
    provenance: Provenance,
};

// ---- KB Types ----

pub const Kb = extern struct {
    name_offset: i32,
    name_length: i16,
    path_offset: i32,
    path_length: i16,
    id: i32,

    facts_offset: i32,
    facts_count: i32,
    facts_capacity: i32,
    rules_offset: i32,
    rules_count: i32,
    rules_capacity: i32,
    constraints_offset: i32,
    constraints_count: i32,
    connections_offset: i32,
    connections_count: i32,
    grammars_offset: i32,
    grammars_count: i32,
    iose_offset: i32,

    working_data_offset: i32,
    lru_table_offset: i32,
    lru_count: i16,
    counter_table_offset: i32,
    counter_count: i16,
    lock_table_offset: i32,
    lock_count: i16,
    queue_table_offset: i32,
    queue_count: i16,
    stack_table_offset: i32,
    stack_count: i16,
    ring_table_offset: i32,
    ring_count: i16,
    bitset_table_offset: i32,
    bitset_count: i16,

    parent_id: i32,
    children_offset: i32,
    children_count: i16,
    children_capacity: i16,
    mounts_offset: i32,
    mounts_count: i16,

    visibility: i8,
    frozen: i8,
    owner_offset: i32,
    owner_length: i16,
    created_at: i32,
    last_modified: i32,

    // Pad to 256 bytes for cache alignment
    _reserved: [58]u8 = [_]u8{0} ** 58,
};

// ---- Prolog Types ----

pub const TermType = enum(i8) {
    atom = 0,
    variable = 1,
    integer = 2,
    vdr = 3,
    text = 4,
    list = 5,
    compound = 6,
    vector = 7,
    matrix = 8,
    pair = 9,
};

pub const Term = extern struct {
    type: TermType,
    _pad0: [3]u8 = [_]u8{0} ** 3,
    // Union fields — only one meaningful per type, but all laid out
    // for fixed-size GPU access. 20 bytes of payload.
    primary_id: i32,       // atom_id, var_id, int_value, functor_id
    secondary_offset: i32, // text_offset, list_head_offset, args_offset
    secondary_aux: i32,    // text_length (as i32), list_tail_offset, args_count
    vdr_value: Q16,        // for vdr type
};

pub const Rule = extern struct {
    id: i32,
    head: i32,
    body_offset: i32,
    body_count: i16,
    _pad0: i16 = 0,
    action_offset: i32,
    action_count: i16,
    _pad1: i16 = 0,
    fire_count: i32,
    last_fired: i32,
    success_count: i32,
    failure_count: i32,
    created_at: i32,
    creator_session_id: i32,
};

pub const Binding = extern struct {
    var_id: i32,
    bound_term_offset: i32,
};

pub const UnificationResult = extern struct {
    unified: i8,
    _pad: [3]u8 = [_]u8{0} ** 3,
    bindings_offset: i32,
    bindings_count: i16,
    _pad2: i16 = 0,
};

// ---- Grammar Types ----

pub const SlotType = enum(i8) {
    vdr_value = 0,
    text = 1,
    integer = 2,
    @"enum" = 3,
    kb_ref = 4,
    grammar = 5,
};

pub const GrammarSlot = extern struct {
    name_offset: i32,
    name_length: i16,
    type: SlotType,
    _pad: i8 = 0,
    enum_values_offset: i32,
    enum_count: i16,
    _pad2: i16 = 0,
    kb_id: i32,
    kb_slot_id: i32,
};

pub const Grammar = extern struct {
    id: i32,
    template_offset: i32,
    template_length: i32,
    slots_offset: i32,
    slots_count: i16,
    validated: i8,
    _pad: i8 = 0,
    created_at: i32,
    creator_session_id: i32,
};

pub const GrammarFill = extern struct {
    slot_index: i16,
    fill_type: SlotType,
    _pad: i8 = 0,
    vdr_value: Q16,
    text_offset: i32,
    text_length: i16,
    int_value: i32,
    enum_index: i16,
};

pub const GrammarKbMapping = extern struct {
    slot_index: i16,
    _pad: i16 = 0,
    kb_id: i32,
    slot_id: i32,
};

// ---- Session Types ----

pub const SessionState = enum(i8) {
    active = 0,
    snapshotted = 1,
    killed = 2,
    frozen = 3,
};

pub const Session = extern struct {
    id: i32,
    user_id: i32,
    kb_root_id: i32,
    visibility_level: i8,
    state: SessionState,
    _pad0: i16 = 0,

    max_kb_count: i32,
    max_live_memory_bytes: i64,
    max_turns: i32,

    current_turn: i32,
    facts_asserted: i32,
    facts_retracted: i32,
    _pad1: i32 = 0,
    rules_fired: i64,
    prolog_queries: i64,
    primitive_calls: i64,
    grammar_renders: i64,
    llm_tokens_consumed: i64,
    command_tokens_consumed: i64,

    device_id: i32,
    stream_id: i32,
    kb_store_offset: i64,

    last_snapshot_id: i32,
    last_snapshot_timestamp: i32,

    parent_session_id: i32,
    clone_generation: i32,
};

// ---- Runner Types ----

pub const RunnerType = enum(i8) {
    poller = 0,
    processor = 1,
    internal = 2,
    batch = 3,
};

pub const RunnerState = enum(i8) {
    stopped = 0,
    running = 1,
    @"error" = 2,
    recycling = 3,
};

pub const Runner = extern struct {
    id: i32,
    type: RunnerType,
    state: RunnerState,
    _pad0: i16 = 0,
    session_id: i32,

    interval_ms: i32,
    max_turns_before_recycle: i32,
    max_consecutive_errors: i32,

    iterations_completed: i64,
    errors_consecutive: i32,
    _pad1: i32 = 0,
    errors_total: i64,
    last_iteration_ms: i32,
    last_iteration_timestamp: i32,

    recycle_count: i32,
    last_recycle_timestamp: i32,
};

// ---- Grant Types ----

pub const GrantClass = enum(i8) {
    filesystem = 0,
    compile = 1,
    execute = 2,
    lint = 3,
    network = 4,
    process = 5,
};

pub const GrantState = enum(i8) {
    active = 0,
    expired = 1,
    exhausted = 2,
    revoked = 3,
};

pub const Grant = extern struct {
    id: i32,
    class: GrantClass,
    state: GrantState,
    _pad0: i16 = 0,
    holder_user_id: i32,
    target_pattern_offset: i32,
    target_pattern_length: i16,
    _pad1: i16 = 0,
    max_uses: i32,
    remaining_uses: i32,
    expires_at: i32,
    created_at: i32,
    created_by: i32,
    revoked_at: i32,
    revoked_by: i32,
};

// ---- Command Types ----

pub const CommandType = enum(i8) {
    kb_assert = 0,
    kb_query = 1,
    kb_retract = 2,
    prolog_query = 3,
    prolog_assert_rule = 4,
    builtin_call = 5,
    grammar_render = 6,
    direct_output = 7,
    op_filesystem = 8,
    op_compile = 9,
    op_execute = 10,
    op_network = 11,
    op_process = 12,
    session_snapshot = 13,
    session_clone = 14,
};

pub const Command = extern struct {
    type: CommandType,
    _pad0: [3]u8 = [_]u8{0} ** 3,
    target_kb_id: i32,
    target_slot_id: i32,
    builtin_id: i32,
    args_offset: i32,
    args_count: i16,
    grant_required: i8,
    _pad1: i8 = 0,
};

// ---- Audit Types ----

pub const AuditAction = enum(i8) {
    fact_assert = 0,
    fact_retract = 1,
    fact_query = 2,
    rule_fire = 3,
    rule_assert = 4,
    rule_retract = 5,
    grant_check = 6,
    grant_create = 7,
    grant_revoke = 8,
    session_create = 9,
    session_kill = 10,
    snapshot = 11,
    clone = 12,
    op_execute = 13,
    access_denied = 14,
};

pub const AuditEntry = extern struct {
    timestamp: i32,
    session_id: i32,
    user_id: i32,
    action: AuditAction,
    _pad0: [3]u8 = [_]u8{0} ** 3,
    target_kb_id: i32,
    target_slot_id: i32,
    grant_id: i32,
    result: i8,
    _pad1: [3]u8 = [_]u8{0} ** 3,
    detail_offset: i32,
};

// ---- Confidence Table ----

pub const confidence_table = [11]Q16{
    .{ .v = 65536, .r0 = 0 }, // vdr_computation 1/1
    .{ .v = 65536, .r0 = 0 }, // prolog_derivation 1/1
    .{ .v = 64225, .r0 = 0 }, // database 98/100
    .{ .v = 62259, .r0 = 0 }, // prometheus 95/100
    .{ .v = 62259, .r0 = 0 }, // script 95/100
    .{ .v = 55705, .r0 = 0 }, // rest_api 85/100
    .{ .v = 52428, .r0 = 0 }, // published 80/100
    .{ .v = 45875, .r0 = 0 }, // user_stated 70/100
    .{ .v = 32768, .r0 = 0 }, // web_search 50/100
    .{ .v = 19660, .r0 = 0 }, // llm_generated 30/100
    .{ .v = 0, .r0 = 0 },     // unknown 0/1
};

// ---- Error / Status Types ----

pub const ErrorCategory = enum(i8) {
    none = 0,
    arithmetic = 1,
    kb = 2,
    prolog = 3,
    grammar = 4,
    session = 5,
    grant = 6,
    runner = 7,
    device = 8,
    system = 9,
};

pub const Status = extern struct {
    category: ErrorCategory,
    _pad0: [3]u8 = [_]u8{0} ** 3,
    code: i32,
    detail: i32,
};

pub const status_ok = Status{ .category = .none, .code = 0, .detail = 0 };

pub const RecoveryAction = enum(i32) {
    none = 0,
    compact = 1,
    log_and_continue = 2,
    simplify_query = 3,
    retry_snapshot = 4,
    log_and_deny = 5,
    reconnect_with_backoff = 6,
    recycle_runner = 7,
    kill_oldest_clone = 8,
    restore_from_snapshot = 9,
};

// ---- Level Stats ----

pub const LevelStats = struct {
    l1_count: i64,
    l1_tokens: i64,
    l2_count: i64,
    l2_tokens: i64,
    l3_count: i64,
    auto_triage_percent_num: i32,
    auto_triage_percent_den: i32,
};
```

```zig
// ============================================================
// VLP — Module: vlp_device_memory.zig
// Device memory layout and GPU buffer definitions.
// ============================================================

const types = @import("vlp_types.zig");

pub const DeviceMemoryLayout = struct {
    model_weights_base: i64,
    model_weights_size: i64,

    kb_store_base: i64,
    kb_store_size: i64,
    kb_count: i32,
    kb_capacity: i32,

    fact_store_base: i64,
    fact_store_size: i64,

    rule_store_base: i64,
    rule_store_size: i64,

    term_store_base: i64,
    term_store_size: i64,

    text_store_base: i64,
    text_store_size: i64,

    grammar_store_base: i64,
    grammar_store_size: i64,

    live_state_base: i64,
    live_state_size: i64,

    scratch_base: i64,
    scratch_size: i64,

    audit_base: i64,
    audit_size: i64,
    audit_capacity: i32,
    audit_head: i32,

    grant_store_base: i64,
    grant_store_size: i64,

    session_table_base: i64,
    session_table_size: i64,
    session_count: i32,
    session_capacity: i32,

    path_index_base: i64,
    path_index_size: i64,
};

pub const MemorySizingConfig = struct {
    model_params: i64,
    qbasis: types.QBasis,
    max_total_kbs: i32,
    max_total_facts: i64,
    max_total_rules: i32,
    max_total_terms: i64,
    text_store_bytes: i64,
    max_grammars: i32,
    max_concurrent_sessions: i32,
    live_state_per_session_bytes: i64,
    scratch_per_stream_bytes: i64,
    n_scratch_streams: i32,
    audit_ring_capacity: i32,
    max_grants: i32,
};

pub const CapacityResult = struct {
    model_memory_bytes: i64,
    kb_memory_bytes: i64,
    live_state_memory_bytes: i64,
    scratch_memory_bytes: i64,
    total_device_memory_bytes: i64,
    n_devices_required: i32,
};

pub fn computeLayout(config: *const MemorySizingConfig) DeviceMemoryLayout;
pub fn computeCapacity(config: *const MemorySizingConfig) CapacityResult;
```

```zig
// ============================================================
// VLP — Module: vlp_gpu_params.zig
// Dispatch parameter structs for all GPU kernels.
// These are written to the uniform buffer (Set 3, binding 0)
// before each kernel dispatch.
// All extern struct for GPU layout compatibility.
// ============================================================

const types = @import("vlp_types.zig");

// ---- LLM Kernel Params ----

pub const EmbeddingLookupParams = extern struct {
    n_tokens: i32,
    d_model: i32,
    _pad: [2]i32 = [_]i32{0} ** 2,
};

pub const LayerNormParams = extern struct {
    n_tokens: i32,
    d_model: i32,
    layer_idx: i32,
    norm_idx: i32,       // 0 = pre-attention, 1 = pre-MLP, 2 = final
    epsilon_v: i32,      // Q16 epsilon for numerical stability
    _pad: [3]i32 = [_]i32{0} ** 3,
};

pub const QkvProjectParams = extern struct {
    n_tokens: i32,
    d_model: i32,
    n_heads: i32,
    d_head: i32,
    layer_idx: i32,
    weights_offset: i32, // offset within layer_weights buffer
    _pad: [2]i32 = [_]i32{0} ** 2,
};

pub const AttentionScoresParams = extern struct {
    n_tokens: i32,
    n_heads: i32,
    d_head: i32,
    seq_len: i32,        // total sequence length including KV cache
    scale_v: i32,        // Q16: 1/sqrt(d_head) as exact fraction
    causal_mask: i32,    // 1 = apply causal mask, 0 = no mask
    _pad: [2]i32 = [_]i32{0} ** 2,
};

pub const SoftmaxExactParams = extern struct {
    row_length: i32,     // number of elements per row
    n_rows: i32,         // number of rows to process
    denominator: i32,    // D value (65536 for Q16)
    _pad: i32 = 0,
};

pub const AttentionWeightedSumParams = extern struct {
    n_tokens: i32,
    n_heads: i32,
    d_head: i32,
    seq_len: i32,
};

pub const OutputProjectParams = extern struct {
    n_tokens: i32,
    d_model: i32,
    layer_idx: i32,
    weights_offset: i32,
};

pub const MlpParams = extern struct {
    n_tokens: i32,
    d_model: i32,
    mlp_dim: i32,
    layer_idx: i32,
    up_weights_offset: i32,
    down_weights_offset: i32,
    activation_type: i32,   // 0 = SiLU, 1 = GELU, 2 = ReLU
    _pad: i32 = 0,
};

pub const LmHeadParams = extern struct {
    n_tokens: i32,
    d_model: i32,
    vocab_size: i32,
    _pad: i32 = 0,
};

pub const KvCacheAppendParams = extern struct {
    n_new_tokens: i32,
    n_heads: i32,
    d_head: i32,
    layer_idx: i32,
    start_position: i32,   // position index for first new token
    max_seq_len: i32,
    _pad: [2]i32 = [_]i32{0} ** 2,
};

// ---- KB Kernel Params ----

pub const FactWriteBatchParams = extern struct {
    n_facts: i32,
    base_offset: i32,
    fact_store_capacity: i32,
    _pad: i32 = 0,
};

pub const FactReadBatchParams = extern struct {
    n_reads: i32,
    _pad: [3]i32 = [_]i32{0} ** 3,
};

pub const FactScanByTagParams = extern struct {
    base_offset: i32,
    scan_length: i32,
    target_tag: i32,
    max_results: i32,
};

pub const ScopedSearchParams = extern struct {
    n_chain_entries: i32,
    total_facts: i32,
    target_tag: i32,
    max_results: i32,
};

// ---- Prolog Kernel Params ----

pub const UnifyCandidatesParams = extern struct {
    n_candidates: i32,
    query_term_type: i32,
    query_atom_id: i32,
    query_int_value: i32,
    query_vdr_v: i32,
    query_vdr_r0: i16,
    _pad0: i16 = 0,
    query_functor_id: i32,
    query_args_count: i32,
    query_args_offset: i32,
    max_bindings_per: i32,
};

pub const RuleMatchScanParams = extern struct {
    n_rules: i32,
    rules_base_offset: i32,
    query_term_type: i32,
    query_atom_id: i32,
    query_functor_id: i32,
    query_args_count: i32,
    max_matches: i32,
    _pad: i32 = 0,
};

pub const RuleBodyEvalParams = extern struct {
    n_matched: i32,
    max_body: i32,
    facts_base_offset: i32,
    facts_count: i32,
};

pub const RuleCheckSatisfiedParams = extern struct {
    n_matched: i32,
    max_body: i32,
    max_fires: i32,
    _pad: i32 = 0,
};

// ---- Builtin Kernel Params ----

pub const BuiltinUnaryParams = extern struct {
    n_elements: i32,
    op_code: i32,         // which unary operation
    input_offset: i32,    // offset in scratch_a
    output_offset: i32,   // offset in scratch_b
};

pub const BuiltinBinaryParams = extern struct {
    n_elements: i32,
    op_code: i32,
    input_a_offset: i32,
    input_b_offset: i32,
    output_offset: i32,
    _pad: [3]i32 = [_]i32{0} ** 3,
};

pub const BuiltinReductionParams = extern struct {
    n_elements: i32,
    op_code: i32,         // sum, product, min, max, mean, etc.
    input_offset: i32,
    _pad: i32 = 0,
};

pub const BuiltinSortParams = extern struct {
    n_elements: i32,
    ascending: i32,       // 1 = ascending, 0 = descending
    input_offset: i32,
    output_offset: i32,
};

pub const BuiltinMatmulParams = extern struct {
    m: i32,
    n: i32,
    k: i32,
    _pad: i32 = 0,
    a_offset: i32,
    b_offset: i32,
    c_offset: i32,
    _pad2: i32 = 0,
};

pub const BuiltinConfidenceCombineParams = extern struct {
    n_sources: i32,
    mode: i32,            // 0 = agreeing, 1 = conflicting
    penalty_v: i32,       // Q16 penalty for conflicting mode
    input_offset: i32,
};

pub const BuiltinConfidenceChainParams = extern struct {
    n_links: i32,
    per_link_v: i32,      // Q16 per-link confidence
    _pad: [2]i32 = [_]i32{0} ** 2,
};
```

```zig
// ============================================================
// VLP — Module: vlp_bridge.zig
// Host-device bridge. Manages Vulkan resources and dispatch.
// ============================================================

const std = @import("std");
const vk = @import("vulkan");
const types = @import("vlp_types.zig");
const mem = @import("vlp_device_memory.zig");
const params = @import("vlp_gpu_params.zig");

pub const PipelineId = enum(i32) {
    // LLM
    embedding_lookup = 0,
    layer_norm = 1,
    qkv_project = 2,
    attention_scores = 3,
    softmax_exact = 4,
    attention_weighted_sum = 5,
    output_project = 6,
    mlp = 7,
    lm_head = 8,
    kv_cache_append = 9,
    residual_add = 10,
    // KB
    fact_write_batch = 11,
    fact_read_batch = 12,
    fact_scan_by_tag = 13,
    scoped_search = 14,
    // Prolog
    unify_candidates = 15,
    rule_match_scan = 16,
    rule_body_eval = 17,
    rule_check_satisfied = 18,
    // Builtins (base IDs — each group is one pipeline)
    builtin_unary = 19,
    builtin_binary = 20,
    builtin_reduction = 21,
    builtin_sort = 22,
    builtin_matmul = 23,
    builtin_confidence_combine = 24,
    builtin_confidence_chain = 25,
    // Utility
    buffer_copy = 26,
    buffer_fill = 27,

    pub const count = 28;
};

pub const DescriptorSetIndex = enum(u32) {
    model = 0,       // Set 0: model weights, read-only
    kb_data = 1,     // Set 1: KB/fact/rule/term stores
    scratch = 2,     // Set 2: scratch/intermediate buffers
    control = 3,     // Set 3: params uniform + status + result counts
};

pub const DispatchConfig = struct {
    pipeline: PipelineId,
    group_count_x: i32,
    group_count_y: i32,
    group_count_z: i32,
    params_ptr: *const anyopaque,
    params_size: i32,
    wait_for_completion: bool,
};

pub const BufferRegion = struct {
    buffer: vk.Buffer,
    offset: i64,
    size: i64,
};

pub const Bridge = struct {
    allocator: std.mem.Allocator,
    instance: vk.Instance,
    physical_device: vk.PhysicalDevice,
    device: vk.Device,
    compute_queue: vk.Queue,
    compute_queue_family: u32,
    command_pool: vk.CommandPool,

    // Pipelines
    pipelines: [PipelineId.count]vk.Pipeline,
    pipeline_layouts: [PipelineId.count]vk.PipelineLayout,

    // Descriptor infrastructure
    descriptor_pool: vk.DescriptorPool,
    descriptor_set_layouts: [4]vk.DescriptorSetLayout,

    // Storage buffers — one per memory region
    model_weights_buffer: vk.Buffer,
    kb_store_buffer: vk.Buffer,
    fact_store_buffer: vk.Buffer,
    rule_store_buffer: vk.Buffer,
    term_store_buffer: vk.Buffer,
    text_store_buffer: vk.Buffer,
    grammar_store_buffer: vk.Buffer,
    live_state_buffer: vk.Buffer,
    scratch_a_buffer: vk.Buffer,
    scratch_b_buffer: vk.Buffer,
    kv_cache_buffer: vk.Buffer,
    status_buffer: vk.Buffer,
    result_counts_buffer: vk.Buffer,
    params_buffer: vk.Buffer,     // uniform buffer for dispatch params

    // Memory objects
    model_weights_memory: vk.DeviceMemory,
    kb_data_memory: vk.DeviceMemory,
    scratch_memory: vk.DeviceMemory,
    control_memory: vk.DeviceMemory,

    // Mapped pointers (if host-visible)
    kb_store_mapped: ?[*]u8,
    fact_store_mapped: ?[*]u8,
    status_mapped: ?[*]i32,
    result_counts_mapped: ?[*]i32,
    params_mapped: ?[*]u8,

    // Layout
    layout: mem.DeviceMemoryLayout,

    // Fence for synchronous dispatches
    dispatch_fence: vk.Fence,
};

pub const BridgeConfig = struct {
    sizing: mem.MemorySizingConfig,
    shader_dir: [*:0]const u8,       // path to compiled .spv files
    enable_validation: bool,
    preferred_device_index: i32,      // -1 for auto
};

// ---- Lifecycle ----

pub fn init(allocator: std.mem.Allocator, config: *const BridgeConfig) types.Status;
pub fn deinit(bridge: *Bridge) void;

// ---- Dispatch ----

pub fn dispatch(bridge: *Bridge, config: *const DispatchConfig) types.Status;
pub fn dispatchAsync(bridge: *Bridge, config: *const DispatchConfig, fence: *vk.Fence) types.Status;
pub fn waitFence(bridge: *Bridge, fence: vk.Fence, timeout_ns: u64) types.Status;

// ---- Buffer data transfer ----

pub fn uploadToBuffer(bridge: *Bridge, buffer: vk.Buffer, offset: i64, data: []const u8) types.Status;
pub fn downloadFromBuffer(bridge: *Bridge, buffer: vk.Buffer, offset: i64, dest: []u8) types.Status;
pub fn copyBuffer(bridge: *Bridge, src: vk.Buffer, src_offset: i64, dst: vk.Buffer, dst_offset: i64, size: i64) types.Status;
pub fn fillBuffer(bridge: *Bridge, buffer: vk.Buffer, offset: i64, size: i64, value: u32) types.Status;

// ---- Status readback ----

pub fn readStatus(bridge: *Bridge, index: i32) i32;
pub fn readResultCount(bridge: *Bridge, index: i32) i32;
pub fn resetResultCounts(bridge: *Bridge) types.Status;
pub fn resetStatusBuffer(bridge: *Bridge) types.Status;

// ---- Descriptor set management ----

pub fn updateKbDescriptors(bridge: *Bridge, session_kb_offset: i64, session_fact_offset: i64) types.Status;
pub fn updateScratchDescriptors(bridge: *Bridge, scratch_a_offset: i64, scratch_b_offset: i64) types.Status;

// ---- Decision ----

pub const OperationType = enum(i32) {
    llm_forward = 0,
    fact_scan = 1,
    unification = 2,
    rule_match = 3,
    builtin_array = 4,
    text_grammar = 5,
    access_check = 6,
    sampling = 7,
};

pub fn shouldUseGpu(bridge: *Bridge, op: OperationType, data_size: i32) bool;
```

```zig
// ============================================================
// VLP — Module: vlp_llm.zig
// LLM engine — host-side orchestration of GPU kernel sequence.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const bridge_mod = @import("vlp_bridge.zig");
const params = @import("vlp_gpu_params.zig");

pub const ModelConfig = struct {
    n_layers: i32,
    d_model: i32,
    n_heads: i32,
    d_head: i32,
    vocab_size: i32,
    mlp_dim: i32,
    max_seq_len: i32,
    qbasis: types.QBasis,
    checkpoint_path: [256]u8,
    activation_type: i32,     // 0 = SiLU, 1 = GELU
};

pub const SamplingConfig = struct {
    temperature_v: i32,       // Q16: 65536 = 1.0
    top_k: i32,
    top_p_v: i32,             // Q16: 58982 ≈ 0.9
    mode: SamplingMode,
};

pub const SamplingMode = enum(i32) {
    greedy = 0,
    top_k = 1,
    top_p = 2,
    temperature = 3,
};

pub const KvCacheConfig = struct {
    max_seq_len: i32,
    n_layers: i32,
    n_heads: i32,
    d_head: i32,
};

pub const ForwardResult = struct {
    logits_offset: i64,       // offset in scratch buffer where logits live
    n_positions: i32,
};

pub const LlmEngine = struct {
    bridge: *bridge_mod.Bridge,
    config: ModelConfig,
    kv_config: KvCacheConfig,
    current_seq_len: i32,
    attention_scale_v: i32,   // Q16: 1/sqrt(d_head)
};

// ---- Lifecycle ----

pub fn init(bridge: *bridge_mod.Bridge, config: *const ModelConfig) types.Status;
pub fn deinit(engine: *LlmEngine) void;
pub fn loadCheckpoint(engine: *LlmEngine, path: [*:0]const u8) types.Status;
pub fn validateChecksum(engine: *LlmEngine) types.Status;

// ---- KV Cache ----

pub fn kvCacheInit(engine: *LlmEngine, config: *const KvCacheConfig) types.Status;
pub fn kvCacheTruncate(engine: *LlmEngine, position: i32) types.Status;
pub fn kvCacheClear(engine: *LlmEngine) types.Status;

// ---- Forward Pass ----

pub fn forward(engine: *LlmEngine, input_ids: []const i32) ForwardResult;
pub fn forwardSingleToken(engine: *LlmEngine, token_id: i32) ForwardResult;

// ---- Sampling (host-side) ----

pub fn sampleToken(engine: *LlmEngine, logits: []const i32, config: *const SamplingConfig) i32;
pub fn sampleGreedy(logits: []const i32) i32;
pub fn sampleTopK(logits: []const i32, k: i32, temperature_v: i32) i32;
pub fn sampleTopP(logits: []const i32, p_v: i32, temperature_v: i32) i32;

// ---- Generation ----

pub fn generateToken(engine: *LlmEngine, sampling: *const SamplingConfig) i32;
pub fn generateCommandTokens(engine: *LlmEngine, command_vocab: []const i32, output: []i32) i32;
pub fn generateProse(engine: *LlmEngine, sampling: *const SamplingConfig, max_tokens: i32, output: []i32) i32;
```

```zig
// ============================================================
// VLP — Module: vlp_kb_store.zig
// KB store engine — host manages structure, GPU does bulk ops.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const bridge_mod = @import("vlp_bridge.zig");

pub const KbCreateConfig = struct {
    name: []const u8,
    path: []const u8,
    parent_id: i32,
    max_facts: i32,
    max_rules: i32,
    visibility: i8,
    owner: []const u8,
};

pub const SearchResult = struct {
    facts: []types.Fact,
    kb_ids: []i32,
    slot_ids: []i32,
    count: i32,
};

pub const ScopedSearchConfig = struct {
    start_kb_id: i32,
    tag: types.FactTag,
    max_depth: i32,
    max_results: i32,
};

pub const CowPageTable = struct {
    parent_session_id: i32,
    clone_session_id: i32,
    page_size: i32,
    n_pages: i32,
    dirty_bits: []u8,          // one bit per page
    private_base_offset: i64,
};

pub const PathIndex = struct {
    allocator: std.mem.Allocator,
    entries: []PathEntry,
    capacity: i32,
    count: i32,
};

pub const PathEntry = struct {
    path_hash: u32,
    kb_id: i32,
};

pub const KbStore = struct {
    bridge: *bridge_mod.Bridge,
    allocator: std.mem.Allocator,
    path_index: PathIndex,
    next_kb_id: i32,
    next_fact_offset: i64,
    next_rule_offset: i32,
    next_term_offset: i64,
    next_text_offset: i64,
};

// ---- Lifecycle ----

pub fn init(bridge: *bridge_mod.Bridge, allocator: std.mem.Allocator) types.Status;
pub fn deinit(store: *KbStore) void;

// ---- KB Management (host-side) ----

pub fn createKb(store: *KbStore, config: *const KbCreateConfig) i32;
pub fn destroyKb(store: *KbStore, kb_id: i32) types.Status;
pub fn getKb(store: *KbStore, kb_id: i32) ?types.Kb;
pub fn freezeKb(store: *KbStore, kb_id: i32) types.Status;
pub fn setVisibility(store: *KbStore, kb_id: i32, visibility: i8) types.Status;

// ---- Path Index (host-side) ----

pub fn pathResolve(store: *KbStore, path: []const u8) ?i32;
pub fn pathRegister(store: *KbStore, path: []const u8, kb_id: i32) types.Status;
pub fn pathRemove(store: *KbStore, path: []const u8) types.Status;

// ---- Fact Operations ----

pub fn factWrite(store: *KbStore, kb_id: i32, slot_id: i32, fact: *const types.Fact) types.Status;
pub fn factRead(store: *KbStore, kb_id: i32, slot_id: i32) ?types.Fact;
pub fn factWriteBatch(store: *KbStore, kb_id: i32, slot_ids: []const i32, facts: []const types.Fact) types.Status;
pub fn factReadBatch(store: *KbStore, kb_id: i32, slot_ids: []const i32, out: []types.Fact) types.Status;
pub fn factRetract(store: *KbStore, kb_id: i32, slot_id: i32) types.Status;

// ---- Scan / Search (GPU-dispatched for large sets, host for small) ----

pub fn factScanByTag(store: *KbStore, kb_id: i32, tag: types.FactTag, max_results: i32) SearchResult;
pub fn scopedSearch(store: *KbStore, config: *const ScopedSearchConfig) SearchResult;

// ---- Text Store (host-side) ----

pub fn textAppend(store: *KbStore, data: []const u8) i32;    // returns offset
pub fn textRead(store: *KbStore, offset: i32, length: i16) []const u8;

// ---- COW (host-side) ----

pub fn cowInit(store: *KbStore, parent_session_id: i32, clone_session_id: i32) types.Status;
pub fn cowHandleWrite(store: *KbStore, cow: *CowPageTable, page_index: i32) types.Status;
pub fn cowResolve(store: *KbStore, cow: *CowPageTable) types.Status;
pub fn cowDestroy(store: *KbStore, cow: *CowPageTable) void;

// ---- Children / Mounts (host-side) ----

pub fn addChild(store: *KbStore, parent_id: i32, child_id: i32) types.Status;
pub fn removeChild(store: *KbStore, parent_id: i32, child_id: i32) types.Status;
pub fn addMount(store: *KbStore, kb_id: i32, source_kb_id: i32, mount_name: []const u8) types.Status;
pub fn removeMount(store: *KbStore, kb_id: i32, mount_name: []const u8) types.Status;
```

```zig
// ============================================================
// VLP — Module: vlp_prolog.zig
// Prolog engine — host drives search, GPU does parallel unification.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const bridge_mod = @import("vlp_bridge.zig");
const kb_mod = @import("vlp_kb_store.zig");

pub const QueryConfig = struct {
    max_depth: i32,           // default 100
    max_bindings: i32,        // default 1000
    max_results: i32,         // default 100
    gpu_candidate_threshold: i32, // min candidates for GPU dispatch (default 32)
};

pub const QueryResult = struct {
    bindings: []types.Binding,
    bindings_count: i32,
    result_count: i32,
    depth_reached: i32,
    depth_exceeded: bool,
};

pub const FireResult = struct {
    firing_rule_ids: []i32,
    firing_count: i32,
    bindings_per_rule: [][]types.Binding,
};

pub const PrologAction = struct {
    is_assert: bool,          // true = assert, false = retract
    target_kb_id: i32,
    target_slot_id: i32,
    fact: types.Fact,
};

pub const ChainEntry = struct {
    kb_id: i32,
    facts_offset: i32,
    facts_count: i32,
};

pub const PrologEngine = struct {
    bridge: *bridge_mod.Bridge,
    kb_store: *kb_mod.KbStore,
    allocator: std.mem.Allocator,
    config: QueryConfig,
};

// ---- Lifecycle ----

pub fn init(bridge: *bridge_mod.Bridge, kb_store: *kb_mod.KbStore, allocator: std.mem.Allocator, config: *const QueryConfig) types.Status;
pub fn deinit(engine: *PrologEngine) void;

// ---- Unification (dispatches to GPU or runs on host depending on size) ----

pub fn unifySingle(engine: *PrologEngine, a: *const types.Term, b: *const types.Term, bindings: []types.Binding) types.UnificationResult;
pub fn unifyCandidates(engine: *PrologEngine, query: *const types.Term, candidate_offsets: []const i32, results: []i32, bindings: []types.Binding) i32;

// ---- Query ----

pub fn query(engine: *PrologEngine, start_kb_id: i32, query_term: *const types.Term) QueryResult;

// ---- Rule Matching ----

pub fn ruleMatchScan(engine: *PrologEngine, kb_id: i32, query_term: *const types.Term) []i32;
pub fn ruleBodyEval(engine: *PrologEngine, matched_rule_ids: []const i32, kb_id: i32) []bool;
pub fn ruleCheckFullySatisfied(engine: *PrologEngine, matched_rule_ids: []const i32, body_results: []const bool, max_body: i32) []i32;

// ---- Rule Firing ----

pub fn fireRules(engine: *PrologEngine, kb_id: i32) FireResult;
pub fn applyActions(engine: *PrologEngine, actions: []const PrologAction) types.Status;
pub fn fireAndCommit(engine: *PrologEngine, kb_id: i32) i32;

// ---- Rule Management (host-side) ----

pub fn ruleAssert(engine: *PrologEngine, kb_id: i32, head: *const types.Term, body: []const types.Term, actions: []const types.Term) i32;
pub fn ruleRetract(engine: *PrologEngine, kb_id: i32, rule_id: i32) types.Status;
pub fn ruleGet(engine: *PrologEngine, rule_id: i32) ?types.Rule;

// ---- Term Management (host-side) ----

pub fn termStore(engine: *PrologEngine, term: *const types.Term) i32;    // returns offset
pub fn termLoad(engine: *PrologEngine, offset: i32) types.Term;
pub fn termParse(engine: *PrologEngine, text: []const u8) ?types.Term;

// ---- Chain Building (host-side, feeds GPU scoped search) ----

pub fn buildChain(engine: *PrologEngine, start_kb_id: i32, max_depth: i32, session: *const types.Session) []ChainEntry;
```

```zig
// ============================================================
// VLP — Module: vlp_grammar.zig
// Grammar engine — entirely host-side.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const kb_mod = @import("vlp_kb_store.zig");

pub const CompileResult = struct {
    grammar: types.Grammar,
    slots: []types.GrammarSlot,
    literal_ranges: []LiteralRange,
    slot_positions: []i32,
};

pub const LiteralRange = struct {
    start: i32,
    length: i32,
};

pub const RenderConfig = struct {
    max_output_bytes: i32,
    recursive_depth_limit: i32,   // for nested grammars, default 10
};

pub const GrammarEngine = struct {
    allocator: std.mem.Allocator,
    kb_store: *kb_mod.KbStore,
};

// ---- Lifecycle ----

pub fn init(allocator: std.mem.Allocator, kb_store: *kb_mod.KbStore) GrammarEngine;
pub fn deinit(engine: *GrammarEngine) void;

// ---- Compile ----

pub fn compile(engine: *GrammarEngine, template: []const u8) CompileResult;
pub fn validate(engine: *GrammarEngine, grammar: *const types.Grammar) types.Status;

// ---- Render ----

pub fn render(engine: *GrammarEngine, grammar: *const types.Grammar, fills: []const types.GrammarFill, config: *const RenderConfig, output: []u8) i32;
pub fn renderFromKb(engine: *GrammarEngine, grammar: *const types.Grammar, mappings: []const types.GrammarKbMapping, config: *const RenderConfig, output: []u8) i32;

// ---- Inheritance ----

pub fn inherit(engine: *GrammarEngine, kb_id: i32, grammar_slot: i32) ?*const types.Grammar;

// ---- VDR formatting (host-side) ----

pub fn q16ToString(value: types.Q16, buf: []u8) i32;
pub fn q32ToString(value: types.Q32, buf: []u8) i32;
pub fn i32ToString(value: i32, buf: []u8) i32;
```

```zig
// ============================================================
// VLP — Module: vlp_builtin.zig
// Builtin executor — host dispatches, GPU or host executes.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const bridge_mod = @import("vlp_bridge.zig");
const kb_mod = @import("vlp_kb_store.zig");

pub const BuiltinCategory = enum(i32) {
    // Pure — GPU-eligible
    text_ops = 0,
    collections = 1,
    sets = 2,
    mappings = 3,
    closed_arithmetic = 4,
    comparison = 5,
    rounding = 6,
    integer_bit_ops = 7,
    linear_algebra = 8,
    statistics = 9,
    active_arithmetic = 10,
    structure_ops = 11,
    number_theory = 12,
    polynomial = 13,
    finite_field = 14,
    discrete_calculus = 15,
    // Operational — host-only, grant-gated
    filesystem = 16,
    compile = 17,
    execute = 18,
    lint = 19,
    network = 20,
    process = 21,
};

pub const IoSe = struct {
    input_types: []const types.TermType,
    output_type: types.TermType,
    side_effects: bool,
    grant_class: ?types.GrantClass,
    max_input_size: i32,
    bounded: bool,
};

pub const BuiltinArgs = struct {
    kb_store: *kb_mod.KbStore,
    input_kb_id: i32,
    input_slot_ids: []const i32,
    output_kb_id: i32,
    output_slot_id: i32,
    extra_params: []const i32,
};

pub const BuiltinResult = struct {
    status: types.Status,
    output_kb_id: i32,
    output_slot_id: i32,
};

pub const BuiltinExecutor = struct {
    bridge: *bridge_mod.Bridge,
    kb_store: *kb_mod.KbStore,
    allocator: std.mem.Allocator,
    iose_table: [448]IoSe,
};

// ---- Lifecycle ----

pub fn init(bridge: *bridge_mod.Bridge, kb_store: *kb_mod.KbStore, allocator: std.mem.Allocator) types.Status;
pub fn deinit(executor: *BuiltinExecutor) void;

// ---- Dispatch ----

pub fn dispatch(executor: *BuiltinExecutor, builtin_id: i32, args: *const BuiltinArgs) BuiltinResult;
pub fn validateIoSe(executor: *BuiltinExecutor, builtin_id: i32, args: *const BuiltinArgs) types.Status;

// ---- IOSE introspection ----

pub fn getIoSe(executor: *BuiltinExecutor, builtin_id: i32) ?IoSe;
pub fn isOperational(executor: *BuiltinExecutor, builtin_id: i32) bool;
pub fn requiredGrant(executor: *BuiltinExecutor, builtin_id: i32) ?types.GrantClass;
```

```zig
// ============================================================
// VLP — Module: vlp_session.zig
// Session manager — host-side lifecycle management.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const bridge_mod = @import("vlp_bridge.zig");
const kb_mod = @import("vlp_kb_store.zig");

pub const SessionConfig = struct {
    user_id: i32,
    kb_root_id: i32,
    visibility_level: i8,
    max_kb_count: i32,
    max_live_memory_bytes: i64,
    max_turns: i32,
    auto_snapshot_interval: i32,
};

pub const CloneConfig = struct {
    fresh_live: bool,
    inherit_rules: bool,
};

pub const MergePolicy = enum(i32) {
    ours = 0,
    theirs = 1,
    fail_on_conflict = 2,
};

pub const MergeConflict = struct {
    kb_id: i32,
    slot_id: i32,
    parent_timestamp: i32,
    child_timestamp: i32,
};

pub const MergeResult = struct {
    status: types.Status,
    merged_count: i32,
    conflict_count: i32,
    conflicts: []MergeConflict,
};

pub const SessionHandle = struct {
    id: i32,
    index: i32,       // index into session table
};

pub const SessionManager = struct {
    bridge: *bridge_mod.Bridge,
    kb_store: *kb_mod.KbStore,
    allocator: std.mem.Allocator,
    sessions: []types.Session,
    session_capacity: i32,
    session_count: i32,
};

// ---- Lifecycle ----

pub fn init(bridge: *bridge_mod.Bridge, kb_store: *kb_mod.KbStore, allocator: std.mem.Allocator, max_sessions: i32) types.Status;
pub fn deinit(mgr: *SessionManager) void;

// ---- Session CRUD ----

pub fn create(mgr: *SessionManager, config: *const SessionConfig) SessionHandle;
pub fn destroy(mgr: *SessionManager, handle: SessionHandle) types.Status;
pub fn get(mgr: *SessionManager, handle: SessionHandle) ?*types.Session;
pub fn kill(mgr: *SessionManager, handle: SessionHandle) types.Status;

// ---- Snapshot ----

pub fn snapshot(mgr: *SessionManager, handle: SessionHandle) SnapshotHandle;
pub fn restore(mgr: *SessionManager, handle: SessionHandle, snap: SnapshotHandle) types.Status;

// ---- Clone / Merge ----

pub fn clone(mgr: *SessionManager, parent: SessionHandle, config: *const CloneConfig) SessionHandle;
pub fn merge(mgr: *SessionManager, parent: SessionHandle, child: SessionHandle, policy: MergePolicy) MergeResult;

// ---- Counters ----

pub fn updateLevelStats(mgr: *SessionManager, handle: SessionHandle, level: i8, tokens: i32) types.Status;
pub fn getLevelStats(mgr: *SessionManager, handle: SessionHandle) types.LevelStats;
```

```zig
// ============================================================
// VLP — Module: vlp_snapshot.zig
// Snapshot manager — host-side save/load/diff/merge.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const bridge_mod = @import("vlp_bridge.zig");

pub const SnapshotHeader = extern struct {
    magic: [4]u8,
    version: i32,
    timestamp: i32,
    session_id: i32,
    user_id: i32,

    kb_region_size: i64,
    fact_region_size: i64,
    rule_region_size: i64,
    term_region_size: i64,
    text_region_size: i64,
    grammar_region_size: i64,
    live_state_region_size: i64,
    grant_region_size: i64,
    path_index_region_size: i64,

    kb_count: i32,
    _pad0: i32 = 0,
    fact_count: i64,
    rule_count: i32,
    _pad1: i32 = 0,
    term_count: i64,
    grammar_count: i32,
    grant_count: i32,

    session_metadata: types.Session,

    checksum: i32,
    _pad2: i32 = 0,
    total_size: i64,
};

pub const SnapshotHandle = struct {
    id: i32,
    data: []u8,
    header: *SnapshotHeader,
};

pub const DiffEntry = struct {
    region: DiffRegion,
    offset: i64,
    size: i64,
    a_hash: u32,
    b_hash: u32,
};

pub const DiffRegion = enum(i32) {
    kb = 0,
    fact = 1,
    rule = 2,
    term = 3,
    text = 4,
    grammar = 5,
    live_state = 6,
    grant = 7,
};

pub const DiffResult = struct {
    entries: []DiffEntry,
    count: i32,
    identical: bool,
};

pub const SnapshotManager = struct {
    allocator: std.mem.Allocator,
    bridge: *bridge_mod.Bridge,
};

// ---- Lifecycle ----

pub fn init(allocator: std.mem.Allocator, bridge: *bridge_mod.Bridge) SnapshotManager;
pub fn deinit(mgr: *SnapshotManager) void;

// ---- Save / Load ----

pub fn save(mgr: *SnapshotManager, snap: SnapshotHandle, path: [*:0]const u8) types.Status;
pub fn load(mgr: *SnapshotManager, path: [*:0]const u8) SnapshotHandle;
pub fn free(mgr: *SnapshotManager, snap: *SnapshotHandle) void;

// ---- Capture from live session ----

pub fn captureFromDevice(mgr: *SnapshotManager, session: *const types.Session) SnapshotHandle;
pub fn restoreToDevice(mgr: *SnapshotManager, snap: SnapshotHandle, session: *types.Session) types.Status;

// ---- Diff / Merge ----

pub fn diff(mgr: *SnapshotManager, a: SnapshotHandle, b: SnapshotHandle) DiffResult;
pub fn mergeThreeWay(mgr: *SnapshotManager, base: SnapshotHandle, branch_a: SnapshotHandle, branch_b: SnapshotHandle, policy: @import("vlp_session.zig").MergePolicy) SnapshotHandle;

// ---- Integrity ----

pub fn validateChecksum(mgr: *SnapshotManager, snap: SnapshotHandle) bool;
pub fn computeChecksum(data: []const u8) i32;
```

```zig
// ============================================================
// VLP — Module: vlp_runner.zig
// Runner scheduler — host-side autonomous execution.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const session_mod = @import("vlp_session.zig");

pub const PollerConfig = struct {
    session: session_mod.SessionHandle,
    interval_ms: i32,
    max_consecutive_errors: i32,
};

pub const ProcessorConfig = struct {
    session: session_mod.SessionHandle,
    source_type: SourceConnectionType,
    source_url: [512]u8,
    source_url_len: i32,
    max_turns_before_recycle: i32,
    max_consecutive_errors: i32,
    backoff_initial_ms: i32,
    backoff_max_ms: i32,
};

pub const SourceConnectionType = enum(i32) {
    http_poll = 0,
    websocket = 1,
    webhook = 2,
    grpc_stream = 3,
};

pub const InternalConfig = struct {
    session: session_mod.SessionHandle,
    interval_ms: i32,
    compute_kb_id: i32,
};

pub const BatchConfig = struct {
    session: session_mod.SessionHandle,
    task_queue_kb_id: i32,
    task_queue_name: [64]u8,
    task_queue_name_len: i32,
    max_concurrent: i32,
};

pub const RunnerHandle = struct {
    id: i32,
    index: i32,
};

pub const RunnerStatus = struct {
    state: types.RunnerState,
    type: types.RunnerType,
    iterations_completed: i64,
    errors_consecutive: i32,
    errors_total: i64,
    last_iteration_ms: i32,
    last_iteration_timestamp: i32,
    recycle_count: i32,
};

pub const RunnerScheduler = struct {
    allocator: std.mem.Allocator,
    session_mgr: *session_mod.SessionManager,
    runners: []types.Runner,
    runner_capacity: i32,
    runner_count: i32,
    thread_pool: std.Thread.Pool,
};

// ---- Lifecycle ----

pub fn init(allocator: std.mem.Allocator, session_mgr: *session_mod.SessionManager, max_runners: i32) types.Status;
pub fn deinit(sched: *RunnerScheduler) void;

// ---- Create ----

pub fn createPoller(sched: *RunnerScheduler, config: *const PollerConfig) RunnerHandle;
pub fn createProcessor(sched: *RunnerScheduler, config: *const ProcessorConfig) RunnerHandle;
pub fn createInternal(sched: *RunnerScheduler, config: *const InternalConfig) RunnerHandle;
pub fn createBatch(sched: *RunnerScheduler, config: *const BatchConfig) RunnerHandle;

// ---- Control ----

pub fn start(sched: *RunnerScheduler, handle: RunnerHandle) types.Status;
pub fn stop(sched: *RunnerScheduler, handle: RunnerHandle) types.Status;
pub fn kill(sched: *RunnerScheduler, handle: RunnerHandle) types.Status;
pub fn recycle(sched: *RunnerScheduler, handle: RunnerHandle) types.Status;

// ---- Status ----

pub fn getStatus(sched: *RunnerScheduler, handle: RunnerHandle) RunnerStatus;
pub fn listAll(sched: *RunnerScheduler) []RunnerStatus;
```

```zig
// ============================================================
// VLP — Module: vlp_grant.zig
// Grant enforcer — host-side access control.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const session_mod = @import("vlp_session.zig");

pub const GrantResult = struct {
    granted: bool,
    grant_id: i32,
    remaining_uses: i32,
};

pub const GrantIndex = struct {
    entries: []GrantIndexEntry,
    count: i32,
    capacity: i32,
};

pub const GrantIndexEntry = struct {
    user_id: i32,
    grant_class: types.GrantClass,
    grant_store_index: i32,
};

pub const GrantEnforcer = struct {
    allocator: std.mem.Allocator,
    grants: []types.Grant,
    grant_count: i32,
    grant_capacity: i32,
    index: GrantIndex,
};

// ---- Lifecycle ----

pub fn init(allocator: std.mem.Allocator, max_grants: i32) types.Status;
pub fn deinit(enforcer: *GrantEnforcer) void;

// ---- Check ----

pub fn check(enforcer: *GrantEnforcer, session: *const types.Session, grant_class: types.GrantClass, target: []const u8) GrantResult;

// ---- Management ----

pub fn create(enforcer: *GrantEnforcer, admin_session: *const types.Session, grant: *const types.Grant) types.Status;
pub fn revoke(enforcer: *GrantEnforcer, admin_session: *const types.Session, grant_id: i32) types.Status;
pub fn list(enforcer: *GrantEnforcer, user_id: i32, out: []types.Grant) i32;
pub fn cleanup(enforcer: *GrantEnforcer, current_time: i32) i32;
```

```zig
// ============================================================
// VLP — Module: vlp_access.zig
// Access control — host-side visibility checks.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const kb_mod = @import("vlp_kb_store.zig");

pub const AccessChecker = struct {
    kb_store: *kb_mod.KbStore,
};

// ---- Check ----

pub fn check(checker: *AccessChecker, session: *const types.Session, kb_id: i32) bool;
pub fn resolveVisible(checker: *AccessChecker, session: *const types.Session, scope_kb_id: i32, visible: []i32) i32;
```

```zig
// ============================================================
// VLP — Module: vlp_audit.zig
// Audit log — host-side append-only ring buffer.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");

pub const AuditFilter = struct {
    session_id: ?i32,
    user_id: ?i32,
    action: ?types.AuditAction,
    target_kb_id: ?i32,
    min_timestamp: ?i32,
    max_timestamp: ?i32,
    result: ?i8,
};

pub const AuditLog = struct {
    entries: []types.AuditEntry,
    capacity: i32,
    head: i32,
    count: i32,
};

// ---- Lifecycle ----

pub fn init(allocator: std.mem.Allocator, capacity: i32) AuditLog;
pub fn deinit(log: *AuditLog) void;

// ---- Write ----

pub fn write(log: *AuditLog, entry: *const types.AuditEntry) void;

// ---- Query ----

pub fn query(log: *AuditLog, filter: *const AuditFilter, out: []types.AuditEntry) i32;
pub fn count(log: *AuditLog, filter: *const AuditFilter) i32;
pub fn latest(log: *AuditLog, n: i32, out: []types.AuditEntry) i32;
```

```zig
// ============================================================
// VLP — Module: vlp_command.zig
// Command processor — host-side LLM→system interface.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const session_mod = @import("vlp_session.zig");
const kb_mod = @import("vlp_kb_store.zig");
const prolog_mod = @import("vlp_prolog.zig");
const grammar_mod = @import("vlp_grammar.zig");
const builtin_mod = @import("vlp_builtin.zig");
const grant_mod = @import("vlp_grant.zig");
const access_mod = @import("vlp_access.zig");
const audit_mod = @import("vlp_audit.zig");

pub const CommandResult = struct {
    status: types.Status,
    output_kb_id: i32,
    output_slot_id: i32,
    output_bytes: i32,
    output_text: ?[]const u8,
};

pub const CommandProcessor = struct {
    kb_store: *kb_mod.KbStore,
    prolog: *prolog_mod.PrologEngine,
    grammar: *grammar_mod.GrammarEngine,
    builtins: *builtin_mod.BuiltinExecutor,
    grants: *grant_mod.GrantEnforcer,
    access: *access_mod.AccessChecker,
    audit: *audit_mod.AuditLog,
    allocator: std.mem.Allocator,
};

// ---- Lifecycle ----

pub fn init(
    kb_store: *kb_mod.KbStore,
    prolog: *prolog_mod.PrologEngine,
    grammar: *grammar_mod.GrammarEngine,
    builtins: *builtin_mod.BuiltinExecutor,
    grants: *grant_mod.GrantEnforcer,
    access: *access_mod.AccessChecker,
    audit: *audit_mod.AuditLog,
    allocator: std.mem.Allocator,
) CommandProcessor;
pub fn deinit(proc: *CommandProcessor) void;

// ---- Parse ----

pub fn parse(proc: *CommandProcessor, tokens: []const i32) ?types.Command;
pub fn parseKbUrl(proc: *CommandProcessor, text: []const u8) ?struct { kb_id: i32, slot_id: i32 };

// ---- Execute ----

pub fn execute(proc: *CommandProcessor, session: session_mod.SessionHandle, command: *const types.Command) CommandResult;
pub fn executeBatch(proc: *CommandProcessor, session: session_mod.SessionHandle, commands: []const types.Command, results: []CommandResult) types.Status;
```

```zig
// ============================================================
// VLP — Module: vlp_confidence.zig
// Confidence propagation — host-side exact VDR arithmetic.
// ============================================================

const types = @import("vlp_types.zig");
const kb_mod = @import("vlp_kb_store.zig");
const bridge_mod = @import("vlp_bridge.zig");

// ---- Assignment ----

pub fn assign(fact: *types.Fact, source_type: types.SourceType) void;

// ---- Combination (host for small N, GPU for large N) ----

pub fn combineAgreeing(bridge: *bridge_mod.Bridge, confidences: []const types.Q16) types.Q16;
pub fn combineConflicting(bridge: *bridge_mod.Bridge, confidences: []const types.Q16, penalty: types.Q16) types.Q16;
pub fn chain(per_link: types.Q16, n_links: i32) types.Q16;

// ---- Propagation ----

pub fn propagate(kb_store: *kb_mod.KbStore, kb_id: i32, slot_id: i32) types.Q16;
```

```zig
// ============================================================
// VLP — Module: vlp_inference.zig
// Inference loop — top-level host orchestration.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
const session_mod = @import("vlp_session.zig");
const llm_mod = @import("vlp_llm.zig");
const command_mod = @import("vlp_command.zig");

pub const OutputBuffer = struct {
    data: []u8,
    length: i32,
    capacity: i32,
};

pub const Scratchpad = struct {
    entries: []ScratchpadEntry,
    count: i32,
    capacity: i32,
};

pub const ScratchpadEntry = struct {
    command_index: i32,
    result: command_mod.CommandResult,
};

pub const ContextConfig = struct {
    system_prompt_kb_id: i32,
    scope_kb_id: i32,
    max_scratchpad_tokens: i32,
};

pub const TokenClassification = enum(i32) {
    prose = 0,
    command_start = 1,
    direct_output = 2,
    end_of_turn = 3,
};

pub const InferenceEngine = struct {
    session_mgr: *session_mod.SessionManager,
    llm: *llm_mod.LlmEngine,
    commands: *command_mod.CommandProcessor,
    allocator: std.mem.Allocator,
    context_config: ContextConfig,
};

// ---- Lifecycle ----

pub fn init(
    session_mgr: *session_mod.SessionManager,
    llm: *llm_mod.LlmEngine,
    commands: *command_mod.CommandProcessor,
    allocator: std.mem.Allocator,
    context_config: *const ContextConfig,
) InferenceEngine;
pub fn deinit(engine: *InferenceEngine) void;

// ---- Full cycle ----

pub fn cycle(engine: *InferenceEngine, session: session_mod.SessionHandle, input: []const u8, output: *OutputBuffer) types.Status;

// ---- Execution levels ----

pub fn executeL1(engine: *InferenceEngine, session: session_mod.SessionHandle, input: []const u8, output: *OutputBuffer) types.Status;
pub fn executeL2(engine: *InferenceEngine, session: session_mod.SessionHandle, pattern: *const types.Term) types.Status;
pub fn executeL3(engine: *InferenceEngine, session: session_mod.SessionHandle, kb_id: i32) types.Status;

// ---- Context ----

pub fn buildContext(engine: *InferenceEngine, session: session_mod.SessionHandle) []i32;
pub fn classifyToken(engine: *InferenceEngine, token_id: i32) TokenClassification;

// ---- Scratchpad ----

pub fn scratchpadWrite(engine: *InferenceEngine, session: session_mod.SessionHandle, result: *const command_mod.CommandResult) void;
pub fn scratchpadClear(engine: *InferenceEngine, session: session_mod.SessionHandle) void;
```

```zig
// ============================================================
// VLP — Module: vlp_seed.zig
// Seed layer — initial KB tree loaded at system boot.
// ============================================================

const types = @import("vlp_types.zig");
const kb_mod = @import("vlp_kb_store.zig");
const snapshot_mod = @import("vlp_snapshot.zig");

pub const SeedConfig = struct {
    snapshot_path: ?[*:0]const u8,   // load from snapshot if available
    create_fresh: bool,               // create from code if no snapshot
};

// ---- Init ----

pub fn init(kb_store: *kb_mod.KbStore, config: *const SeedConfig) types.Status;
pub fn loadFromSnapshot(kb_store: *kb_mod.KbStore, snap_mgr: *snapshot_mod.SnapshotManager, path: [*:0]const u8) types.Status;
pub fn createFresh(kb_store: *kb_mod.KbStore) types.Status;

// ---- Seed content builders ----

pub fn populateOso(kb_store: *kb_mod.KbStore, system_kb_id: i32) types.Status;
pub fn populateConfidenceTable(kb_store: *kb_mod.KbStore, system_kb_id: i32) types.Status;
pub fn populateBuiltinIoSe(kb_store: *kb_mod.KbStore, system_kb_id: i32) types.Status;
pub fn populateCommandVocab(kb_store: *kb_mod.KbStore, system_kb_id: i32) types.Status;
pub fn populateHygieneRules(kb_store: *kb_mod.KbStore, system_kb_id: i32) types.Status;
pub fn populateSentenceTemplates(kb_store: *kb_mod.KbStore, templates_kb_id: i32) types.Status;
pub fn populateFormatGrammars(kb_store: *kb_mod.KbStore, formats_kb_id: i32) types.Status;
```

```zig
// ============================================================
// VLP — Module: vlp_system.zig
// Top-level system initialization — wires everything together.
// ============================================================

const std = @import("std");
const types = @import("vlp_types.zig");
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
const confidence_mod = @import("vlp_confidence.zig");
const inference_mod = @import("vlp_inference.zig");
const seed_mod = @import("vlp_seed.zig");

pub const SystemConfig = struct {
    // Device
    device_id: i32,
    n_devices: i32,
    shader_dir: [256]u8,
    enable_validation: bool,

    // Model
    model: llm_mod.ModelConfig,

    // Memory sizing
    memory: @import("vlp_device_memory.zig").MemorySizingConfig,

    // Sessions
    max_concurrent_sessions: i32,
    default_max_kb_per_session: i32,
    default_max_turns: i32,
    auto_snapshot_interval: i32,

    // Runners
    max_runners: i32,

    // Safety
    audit_ring_capacity: i32,
    max_grants: i32,
    default_visibility: i8,

    // Seed
    seed: seed_mod.SeedConfig,

    // Sampling defaults
    sampling: llm_mod.SamplingConfig,

    // Prolog defaults
    prolog: prolog_mod.QueryConfig,
};

pub const System = struct {
    allocator: std.mem.Allocator,
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
    config: SystemConfig,
};

// ---- Lifecycle ----

pub fn init(allocator: std.mem.Allocator, config: *const SystemConfig) ?*System;
pub fn deinit(system: *System) void;

// ---- Top-level operations ----

pub fn handleUserInput(system: *System, session: session_mod.SessionHandle, input: []const u8, output: *inference_mod.OutputBuffer) types.Status;
pub fn getSystemStatus(system: *System) SystemStatus;

pub const SystemStatus = struct {
    n_sessions: i32,
    n_runners: i32,
    n_kbs: i32,
    total_facts: i64,
    total_rules: i32,
    total_grants: i32,
    audit_entries: i32,
    device_memory_used: i64,
    device_memory_total: i64,
};

// ---- Error recovery ----

pub fn recoverFromError(system: *System, session: session_mod.SessionHandle, err: types.Status) types.RecoveryAction;
```

```zig
// ============================================================
// VLP — Module: vlp_multi_device.zig
// Multi-device support — pipeline parallelism for large models.
// ============================================================

const types = @import("vlp_types.zig");
const bridge_mod = @import("vlp_bridge.zig");
const llm_mod = @import("vlp_llm.zig");
const kb_mod = @import("vlp_kb_store.zig");

pub const DeviceAssignment = struct {
    device_id: i32,
    layer_start: i32,
    layer_end: i32,
    bridge: *bridge_mod.Bridge,
};

pub const MultiDeviceConfig = struct {
    n_devices: i32,
    device_ids: []const i32,
    model: llm_mod.ModelConfig,
    partition_strategy: PartitionStrategy,
};

pub const PartitionStrategy = enum(i32) {
    pipeline = 0,        // each device gets a contiguous range of layers
    tensor_parallel = 1, // each device gets a slice of each layer
};

pub const MultiDeviceManager = struct {
    assignments: []DeviceAssignment,
    n_devices: i32,
    strategy: PartitionStrategy,
};

// ---- Lifecycle ----

pub fn init(config: *const MultiDeviceConfig, allocator: @import("std").mem.Allocator) types.Status;
pub fn deinit(mgr: *MultiDeviceManager) void;

// ---- Forward pass across devices ----

pub fn forward(mgr: *MultiDeviceManager, input_ids: []const i32, logits: []i32) types.Status;

// ---- KB replication ----

pub fn replicateKb(mgr: *MultiDeviceManager, source_device: i32, target_device: i32, kb_id: i32) types.Status;
pub fn syncKb(mgr: *MultiDeviceManager, kb_id: i32) types.Status;
```

```zig
// ============================================================
// VLP — Module: vlp_test.zig
// Testing infrastructure — determinism, roundtrip, isolation.
// ============================================================

const types = @import("vlp_types.zig");
const session_mod = @import("vlp_session.zig");
const kb_mod = @import("vlp_kb_store.zig");

pub const TestResult = struct {
    passed: bool,
    message: [256]u8,
    message_len: i32,
    runs: i32,
    failures: i32,
};

// ---- Determinism ----

pub fn testDeterminism(test_fn: *const fn () types.Status, n_runs: i32) TestResult;

// ---- Snapshot roundtrip ----

pub fn testSnapshotRoundtrip(session_mgr: *session_mod.SessionManager, session: session_mod.SessionHandle) TestResult;

// ---- Clone independence ----

pub fn testCloneIndependence(session_mgr: *session_mod.SessionManager, parent: session_mod.SessionHandle) TestResult;

// ---- Access isolation ----

pub fn testAccessIsolation(session_mgr: *session_mod.SessionManager, session_a: session_mod.SessionHandle, session_b: session_mod.SessionHandle) TestResult;

// ---- Confidence propagation ----

pub fn testConfidencePropagation(kb_store: *kb_mod.KbStore) TestResult;

// ---- Softmax sum invariant ----

pub fn testSoftmaxSumInvariant(input: []const i32, denominator: i32) TestResult;
```

---

**Module summary — 20 files:**

| Module | Lines (est) | Host/GPU | Role |
|---|---|---|---|
| `vlp_types.zig` | ~450 | shared | All type declarations |
| `vlp_device_memory.zig` | ~80 | host | Memory layout + sizing |
| `vlp_gpu_params.zig` | ~200 | shared | GPU kernel dispatch params |
| `vlp_bridge.zig` | ~150 | host | Vulkan resource management + dispatch |
| `vlp_llm.zig` | ~100 | host+GPU | LLM forward pass orchestration |
| `vlp_kb_store.zig` | ~130 | host+GPU | KB management + bulk GPU ops |
| `vlp_prolog.zig` | ~120 | host+GPU | Prolog search + parallel unification |
| `vlp_grammar.zig` | ~60 | host | Grammar compile/render |
| `vlp_builtin.zig` | ~70 | host+GPU | Builtin dispatch |
| `vlp_session.zig` | ~90 | host | Session lifecycle |
| `vlp_snapshot.zig` | ~80 | host | Snapshot save/load/diff |
| `vlp_runner.zig` | ~100 | host | Runner scheduler |
| `vlp_grant.zig` | ~60 | host | Grant enforcement |
| `vlp_access.zig` | ~20 | host | Visibility checks |
| `vlp_audit.zig` | ~40 | host | Audit ring buffer |
| `vlp_command.zig` | ~70 | host | Command parse/execute |
| `vlp_confidence.zig` | ~30 | host+GPU | Confidence propagation |
| `vlp_inference.zig` | ~80 | host | Inference loop orchestration |
| `vlp_seed.zig` | ~50 | host | Seed layer population |
| `vlp_system.zig` | ~80 | host | Top-level wiring |
| `vlp_multi_device.zig` | ~50 | host | Multi-GPU support |
| `vlp_test.zig` | ~40 | host | Test infrastructure |
