```
================================================================================
VLP GPU Integration Test Runbook
================================================================================

PURPOSE
───────
Verify that VLP Zig code compiles, the SPIR-V kernel loads on real GPU
hardware, dispatches execute, and data round-trips correctly between
host and device. This is NOT functional testing of the VLP system.
It is hardware integration testing.

ENVIRONMENT
───────────
GCP VM: n1-standard-4 + nvidia-tesla-t4 (cheapest GPU with Vulkan 1.2)
OS: Ubuntu 24.04 LTS
Zig: 0.16.0 (pinned)
Estimated cost: $0.40/hr on-demand, $0.08/hr spot. Budget 2 hours.

================================================================================
STAGE 0: VM SETUP
================================================================================

Create VM:
  gcloud compute instances create vlp-gpu-test \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --boot-disk-size=50GB \
    --image-family=ubuntu-2404-lts-amd64 \
    --image-project=ubuntu-os-cloud \
    --maintenance-policy=TERMINATE

SSH in:
  gcloud compute ssh vlp-gpu-test --zone=us-central1-a

Install drivers + tools:
  sudo apt-get update -qq
  sudo apt-get install -y -qq \
    nvidia-driver-550 nvidia-utils-550 \
    vulkan-tools libvulkan1 libvulkan-dev \
    mesa-vulkan-drivers wget xz-utils

Install Zig 0.16.0:
  cd /tmp
  wget -q https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz
  sudo tar -xf zig-x86_64-linux-0.16.0.tar.xz -C /opt/
  sudo ln -sf /opt/zig-x86_64-linux-0.16.0/zig /usr/local/bin/zig

Reboot (NVIDIA driver needs it):
  sudo reboot
  # Wait 30s, SSH back in

Verify:
  nvidia-smi                    # Should show Tesla T4
  vulkaninfo --summary          # Should show Vulkan 1.2+ device
  zig version                   # Should show 0.16.0

If nvidia-smi fails: driver didn't load. Check dmesg | grep nvidia.
If vulkaninfo fails: Vulkan ICD not found. Check /usr/share/vulkan/icd.d/

Upload project:
  # From local machine:
  gcloud compute scp --recurse ./vlp vlp-gpu-test:~ --zone=us-central1-a

GATE: Do not proceed until nvidia-smi and vulkaninfo both succeed.

================================================================================
STAGE 1: SPIR-V COMPILATION
================================================================================

Goal: Verify vlp_kernel.zig compiles to valid SPIR-V.

Test 1.1 — Kernel compiles:
  cd ~/vlp
  zig build-obj \
    -target spirv32-vulkan \
    -mcpu vulkan_v1_2 \
    -ofmt spirv \
    -fno-llvm \
    src/vlp_kernel.zig

  Expected: produces vlp_kernel.spv (or similar output file)
  If error: note exact error message. Likely causes:
    - std.gpu builtins not recognized → check Zig version is 0.16.0
    - addrspace(.storage_buffer) rejected → backend limitation
    - @atomicRmw not supported → backend limitation
    - @fence(.workgroup) not supported → try barrier() or memoryBarrier()
    - shared memory declaration rejected → syntax issue

  If compilation fails on specific op: comment out ops in the switch
  until you find the minimal failing case. The kernel is designed so
  each op is independent — you can test subsets.

Test 1.2 — SPIR-V validation (if spirv-val available):
  # Install SPIRV-Tools if not present:
  sudo apt-get install -y spirv-tools

  spirv-val vlp_kernel.spv

  Expected: no errors
  Warnings about array bounds are acceptable (fixed-size arrays larger
  than actual buffer — Vulkan runtime handles this).
  Errors about missing decorations: backend bug. Note which decoration.
  Errors about invalid control flow: simplify the offending op.

Test 1.3 — Full build system:
  zig build

  Expected: compiles SPIR-V kernel, embeds in host binary, links Vulkan.
  If error in build.zig: check spirv target query syntax for 0.16.0.
  If linker error for libvulkan: verify libvulkan-dev is installed.
  If embed fails: check @alignOf(u32) on the embedded .spv.

GATE: Do not proceed until zig build succeeds and produces ./zig-out/bin/vlp

================================================================================
STAGE 2: VULKAN DEVICE INIT
================================================================================

Goal: Verify bridge can create Vulkan instance, find GPU, create device.

Write src/test_stage2.zig:

  const std = @import("std");
  const bridge_mod = @import("vlp_bridge");
  const mem = @import("vlp_device_memory");

  pub fn main() !void {
      const w = std.io.getStdOut().writer();
      const sizing = mem.defaultSizingConfig();
      // Override to tiny sizes for testing
      var test_sizing = sizing;
      test_sizing.model_params = 0;
      test_sizing.max_total_kbs = 100;
      test_sizing.max_total_facts = 50000; // 100 KBs × 500 facts
      test_sizing.max_total_rules = 100;
      test_sizing.max_total_terms = 1000;
      test_sizing.text_store_bytes = 1024 * 1024; // 1 MB
      test_sizing.max_grammars = 10;
      test_sizing.max_concurrent_sessions = 10;
      test_sizing.audit_ring_capacity = 1000;
      test_sizing.max_grants = 100;
      test_sizing.max_dispatch_invocations = 1024;

      const config = bridge_mod.BridgeConfig{ .sizing = test_sizing, .enable_validation = true };
      var bridge = bridge_mod.init(std.heap.page_allocator, &config);

      if (bridge.initialized) {
          try w.print("PASS: Vulkan device initialized\n", .{});
          try w.print("  Device: {s}\n", .{bridge.deviceName()});
          try w.print("  Memory: {} MB\n", .{@divTrunc(bridge.totalDeviceMemory(), 1024 * 1024)});
          try w.print("  Int64:  {}\n", .{bridge.supportsInt64()});
          try w.print("  Shared: {} KB\n", .{@divTrunc(bridge.sharedMemorySize(), 1024)});
          bridge.deinit();
      } else {
          try w.print("FAIL: Vulkan device init failed\n", .{});
      }
  }

Run: zig build-exe src/test_stage2.zig [with appropriate imports] && ./test_stage2

Expected output:
  PASS: Vulkan device initialized
  Device: Tesla T4
  Memory: 15360 MB
  Int64:  true
  Shared: 48 KB

If FAIL: check vulkaninfo output. Common causes:
  - No Vulkan ICD for NVIDIA → install nvidia-driver-550
  - Queue family not found → driver issue
  - Memory allocation failed → reduce test_sizing further

GATE: Bridge initializes and reports device name.

================================================================================
STAGE 3: BUFFER ROUND-TRIP
================================================================================

Goal: Verify host can write data to GPU buffer and read it back identically.

Test 3.1 — Write i32 array, read back, compare:
  1. bridge.uploadInts(.scratch_a, 0, &[_]i32{1, 2, 3, 4, 5, 6, 7, 8})
  2. bridge.downloadInts(.scratch_a, 0, &result)
  3. Compare: result must equal input exactly. Every element.

  PASS: all 8 values match.
  FAIL: data corruption in transfer. Check staging buffer path vs mapped path.

Test 3.2 — Write Fact, read back:
  1. Create Fact with known values: tag=0, v=42, r0=7, provenance fields
  2. bridge.uploadFact(0, &fact)
  3. fact2 = bridge.downloadFact(0)
  4. Compare all 10 i32 fields via toInts/fromInts

  PASS: all fields match.
  FAIL: struct layout mismatch between host and GPU buffer.

Test 3.3 — Large buffer write/read:
  1. Allocate 10000 i32s with known pattern: data[i] = i * 7 + 3
  2. Upload to scratch_a
  3. Download from scratch_a
  4. Compare every element

  PASS: all 10000 match.
  FAIL: partial transfer, staging buffer too small, or offset calculation wrong.

GATE: All three round-trip tests pass.

================================================================================
STAGE 4: SIMPLEST KERNEL DISPATCH
================================================================================

Goal: Dispatch the kernel with op_code=27 (buffer_fill), verify output.

Test 4.1 — Buffer fill:
  1. Reset result_counts
  2. Build params: gpu_params.bufferFill(0, 100, 42, 1)
     (fill scratch_b[0..99] with value 42, element size 1 int)
  3. bridge.dispatch1D(params, 100)
  4. Download scratch_b[0..99]
  5. Verify every element == 42

  PASS: all 100 elements are 42.
  FAIL at dispatch: kernel failed to load, pipeline creation failed,
    or shader module invalid. Check Vulkan validation layer output.
  FAIL at verify: kernel ran but produced wrong output.
    Check op_code routing — is the switch hitting case 27?

Test 4.2 — Buffer copy:
  1. Upload [1,2,3,4,5] to scratch_a
  2. Build params: gpu_params.bufferCopy(0, 0, 5, 1)
  3. Dispatch
  4. Download scratch_b[0..4]
  5. Verify matches [1,2,3,4,5]

  PASS: copy is bit-identical.
  FAIL: buffer binding wrong (scratch_a vs scratch_b swapped).

Test 4.3 — Residual add:
  1. Upload [10, 20, 30, 40] to scratch_a
  2. Upload [1, 2, 3, 4] to scratch_b
  3. Build params: gpu_params.residualAdd(4)
  4. Dispatch
  5. Download scratch_a[0..3]
  6. Verify: [11, 22, 33, 44]

  PASS: integer addition on GPU works.
  FAIL: kernel reads wrong buffer, or addition overflows (shouldn't at these values).

GATE: All three dispatch tests pass. GPU kernel executes and produces correct output.

================================================================================
STAGE 5: ARITHMETIC KERNELS
================================================================================

Goal: Verify Q16 integer arithmetic on GPU matches host.

Test 5.1 — Builtin unary (abs, negate, square):
  1. Upload [-100, 200, -300, 400] to scratch_a
  2. Dispatch builtin_unary with sub_op=0 (abs)
  3. Verify scratch_b: [100, 200, 300, 400]
  4. Dispatch builtin_unary with sub_op=1 (negate)
  5. Verify scratch_b: [100, -200, 300, -400]
  6. Upload [256] to scratch_a (Q16: 256/65536 ≈ 0.0039)
  7. Dispatch builtin_unary with sub_op=10 (square)
  8. Verify scratch_b: [1] (256*256/65536 = 1)

  PASS: unary ops match host Q16 arithmetic.

Test 5.2 — Builtin binary (add, mul, div):
  1. Upload [32768, 32768] to scratch_a at offsets 0 and 4
     (Two Q16 values: 0.5, 0.5)
  2. Dispatch builtin_binary sub_op=0 (add), n=1, in_a=0, in_b=1, out=0
  3. Verify scratch_b[0] == 65536 (0.5 + 0.5 = 1.0)
  4. Dispatch builtin_binary sub_op=2 (mul)
  5. Verify scratch_b[0] == 16384 (0.5 * 0.5 = 0.25)
  6. Upload [65536, 32768] (1.0 and 0.5)
  7. Dispatch builtin_binary sub_op=3 (div)
  8. Verify scratch_b[0] == 131072 (1.0 / 0.5 = 2.0)

  PASS: Q16 multiply and divide on GPU produce exact expected values.

Test 5.3 — Builtin reduction (sum):
  1. Upload [10000, 20000, 30000, 5536] to scratch_a (sum = 65536 = 1.0)
  2. Dispatch builtin_reduction sub_op=0 (sum), n=4
  3. Verify scratch_b[0] == 65536

  PASS: parallel reduction produces exact sum.

Test 5.4 — Confidence combine:
  1. Upload two confidence values: [62259, 62259] (95% and 95%)
  2. Dispatch confidence_combine mode=0 (agreeing), n=2
  3. Expected: 1 - (1-0.95)^2 = 0.9975
     In Q16: 65536 - (3277 * 3277 / 65536) = 65536 - 163 = 65373 (approximately)
  4. Verify scratch_b[0] is in range [65370, 65376]
     (Exact value depends on integer division rounding)

  PASS: confidence combination produces expected range.

Test 5.5 — Confidence chain:
  1. Dispatch confidence_chain per_link_v=55705 (85%), n_links=3
  2. Expected: (55705/65536)^3 ≈ 0.614 → ~40263 in Q16
  3. Verify scratch_b[0] is in range [40000, 40500]
  4. Verify scratch_b[0] < 55705 (must be less than single link)

  PASS: chained confidence decreases correctly.

Test 5.6 — Cross-check host vs GPU:
  For each arithmetic test above, compute the same operation on host
  using types.Q16.mul, types.Q16.add, etc. Compare host result to
  GPU result. They MUST be identical. Not close. Identical.

  PASS: every host result == corresponding GPU result, all i32 bits.
  FAIL: host and GPU use different rounding. Find which operation diverges.

GATE: All arithmetic tests pass. Host and GPU produce identical results.

================================================================================
STAGE 6: FACT STORE KERNELS
================================================================================

Goal: Verify fact read/write/scan kernels work.

Test 6.1 — Fact write batch + read batch:
  1. Create 10 facts on host with known values
  2. Upload facts to scratch_a, slot_ids to scratch_b
  3. Dispatch fact_write_batch, base_offset=0, n=10
  4. Check status_buffer: all zeros (no errors)
  5. Upload read offsets [0,1,2,...,9] to scratch_a
  6. Dispatch fact_read_batch, n=10
  7. Download 10 facts from scratch_b
  8. Compare each fact field-by-field with originals

  PASS: 10 facts written and read back identically.

Test 6.2 — Fact scan by tag:
  1. Write 100 facts: 30 with tag=value(0), 50 with tag=text(1), 20 with tag=empty(255)
  2. Reset result_counts
  3. Dispatch fact_scan_by_tag, target_tag=1, max_results=100
  4. Read result_counts[0]: should be 50
  5. Read 50 slot indices from scratch_a
  6. Verify each index points to a fact with tag=1

  PASS: scan found exactly 50 text facts, zero false positives.

Test 6.3 — Fact scan empty store:
  1. Fill 100 fact slots with tag=empty
  2. Dispatch fact_scan_by_tag, target_tag=0
  3. Verify result_counts[0] == 0

  PASS: no false positives on empty store.

Test 6.4 — Fact write bounds check:
  1. Dispatch fact_write_batch with slot_id > capacity
  2. Check status_buffer: should contain error code 201 (KB_FULL)

  PASS: kernel correctly reports out-of-bounds write.

GATE: Fact store round-trip and scan work correctly.

================================================================================
STAGE 7: PROLOG KERNELS
================================================================================

Goal: Verify unification and rule matching on GPU.

Test 7.1 — Unify candidates (atom match):
  1. Write 10 facts to fact_store: 3 with value.v=42, 7 with other values
  2. Upload candidate offsets [0..9] to scratch_a
  3. Dispatch unify_candidates with query_type=ATOM, query_atom_id=42
  4. Read result_counts[0]: should be 3
  5. Read status_buf: exactly 3 entries with value 1

  PASS: unification found exactly 3 matching atoms.

Test 7.2 — Unify candidates (variable matches all):
  1. Same 10 facts
  2. Dispatch with query_type=VARIABLE
  3. Verify result_counts[0] == 10 (variable matches everything non-empty)
     (Subtract any empty facts)

  PASS: variable unification matches all non-empty facts.

Test 7.3 — Unify candidates (VDR exact match):
  1. Write facts with Q16 values: v=32768 (0.5), v=32769, v=32768
  2. Dispatch with query_type=VDR, query_vdr_v=32768, query_vdr_r0=0
  3. Verify exactly 2 matches (the two facts with v=32768)

  PASS: VDR comparison is exact, not approximate.

Test 7.4 — Rule match scan:
  1. Write 5 rules to rule_store with head terms:
     rule 0: head = atom(10)
     rule 1: head = atom(20)
     rule 2: head = atom(10)
     rule 3: head = compound(functor=5, argc=2)
     rule 4: head = variable(0)
  2. Write corresponding head terms to term_store
  3. Dispatch rule_match_scan with query_type=ATOM, query_atom_id=10
  4. Verify result: rules 0, 2, and 4 match
     (rule 4 matches because variable head matches any query)

  PASS: rule head matching with variable wildcard works.

Test 7.5 — Rule body eval:
  1. Write a rule with 2 body conditions: atom(100) and atom(200)
  2. Write facts: one with v=100, one with v=200, one with v=300
  3. Dispatch rule_body_eval with the matched rule
  4. Verify both body conditions satisfied (both found in facts)

  PASS: body condition evaluation scans fact store correctly.

Test 7.6 — Rule check satisfied:
  1. Set up 3 matched rules:
     rule A: 2 body conditions, both satisfied
     rule B: 2 body conditions, one NOT satisfied
     rule C: 0 body conditions (always satisfied)
  2. Dispatch rule_check_satisfied
  3. Verify: rules A and C fire, rule B does not

  PASS: AND reduction over body conditions is correct.

GATE: Prolog kernels produce correct unification and rule matching.

================================================================================
STAGE 8: LLM KERNELS (SMOKE TEST)
================================================================================

Goal: Verify LLM kernels execute without crashing on tiny dimensions.
NOT testing correctness of a real model — just kernel mechanics.

Use toy dimensions: d_model=8, n_heads=2, d_head=4, vocab_size=16,
mlp_dim=16, n_layers=1, seq_len=4, n_tokens=2.

Test 8.1 — Embedding lookup:
  1. Fill embedding_table with known pattern: table[token][dim] = token * 100 + dim
  2. Upload token_ids [3, 7] to scratch_a
  3. Dispatch embedding_lookup, n_tokens=2, d_model=8
  4. Download scratch_b
  5. Verify: scratch_b[0..7] = [300,301,302,...,307]
            scratch_b[8..15] = [700,701,702,...,707]

  PASS: embedding gather works.

Test 8.2 — Residual add (already tested in stage 4, skip or re-verify).

Test 8.3 — Softmax exact:
  1. Upload one row: [10000, 20000, 30000, 5536] to scratch_a
  2. Dispatch softmax_exact, row_length=4, n_rows=1, denominator=65536
  3. Download 4 values from scratch_a
  4. Verify: sum == 65536 EXACTLY
  5. Verify: largest input (30000) maps to largest probability
  6. Verify: all values >= 0

  PASS: softmax sums to D, ordering preserved, FRU remainder redistribution works.
  FAIL on sum != 65536: FRU logic broken. This is INVARIANT 1 violation.

Test 8.4 — Layer norm (smoke):
  1. Fill ln_params with gamma=65536 (1.0) for all dims
  2. Upload input: [65536, 0, 65536, 0, 65536, 0, 65536, 0] (alternating 1.0, 0.0)
  3. Dispatch layer_norm, n_tokens=1, d_model=8
  4. Download output
  5. Verify: output is non-zero, no NaN-equivalents (INT_MIN), reasonable range

  PASS: layer norm doesn't crash and produces finite values.
  This is a smoke test — correctness requires known-good reference values.

Test 8.5 — QKV project (smoke):
  1. Fill layer_weights with identity-like pattern
  2. Upload known input
  3. Dispatch qkv_project
  4. Verify output is non-zero and in reasonable range

  PASS: GEMM kernel doesn't crash on tiny dimensions.

Test 8.6 — Full forward pass sequence:
  1. Set up all buffers with toy data
  2. Dispatch the full layer sequence using dispatchSequence:
     embedding → layer_norm → qkv → kv_cache_append → attn_scores →
     softmax → attn_weighted_sum → output_project → residual_add →
     layer_norm → mlp → residual_add → layer_norm → lm_head
  3. Verify: no dispatch errors, no status buffer errors
  4. Download logits
  5. Verify: logits are finite i32 values (not zero, not INT_MIN)

  PASS: full forward pass executes without errors on toy model.
  This does NOT verify model correctness — just that the kernel
  sequence runs without crashing or producing garbage.

GATE: All LLM kernels execute on toy dimensions without errors.

================================================================================
STAGE 9: SORT AND MATMUL
================================================================================

Test 9.1 — Bitonic sort:
  1. Upload [50, 10, 40, 20, 30, 80, 60, 70] to scratch_a
  2. Dispatch builtin_sort, n=8, ascending=1
  3. Verify scratch_b: [10, 20, 30, 40, 50, 60, 70, 80]

  PASS: sort produces correct ascending order.

Test 9.2 — Sort descending:
  1. Same input
  2. Dispatch with ascending=0
  3. Verify scratch_b: [80, 70, 60, 50, 40, 30, 20, 10]

  PASS: descending sort works.

Test 9.3 — Matmul 2×2:
  1. Upload A = [65536, 0, 0, 65536] (Q16 identity matrix)
  2. Upload B = [131072, 65536, 0, 196608] (Q16 [2,1; 0,3])
  3. Dispatch builtin_matmul, m=2, n=2, k=2
  4. Verify C = B (identity × B = B)

  PASS: integer GEMM with identity matrix produces correct result.

Test 9.4 — Matmul non-trivial:
  1. A = [131072, 65536, 65536, 131072] (Q16 [2,1; 1,2])
  2. B = [65536, 0, 0, 65536] (Q16 identity)
  3. Dispatch, verify C = A
  4. B = [131072, 0, 0, 131072] (Q16 [2,0; 0,2])
  5. Dispatch, verify C = [262144, 131072, 131072, 262144] (Q16 [4,2; 2,4])

  PASS: matmul accumulation and Q16 division produce correct results.

GATE: Sort and matmul kernels produce correct results.

================================================================================
STAGE 10: ATOMICS AND SHARED MEMORY
================================================================================

Goal: Verify @atomicRmw and addrspace(.shared) work in Zig SPIR-V.

Test 10.1 — Atomic counter:
  1. Dispatch fact_scan_by_tag on 1000 facts, all matching
  2. Verify result_counts[0] == 1000
  3. Repeat 5 times — must get 1000 every time (deterministic)

  PASS: atomicAdd produces correct count, deterministic across runs.
  FAIL: count varies between runs → race condition in atomic implementation.

Test 10.2 — Shared memory reduction:
  1. Upload 256 values, all equal to 1 (i.e., sum should be 256)
  2. Dispatch builtin_reduction sub_op=0 (sum), n=256
  3. Verify scratch_b[0] == 256

  PASS: shared memory tree reduction works.

Test 10.3 — Shared memory reduction large:
  1. Upload 10000 values, all equal to 1
  2. Dispatch builtin_reduction, n=10000
  3. Verify scratch_b[0] == 10000
     (Requires multiple chunks per workgroup thread)

  PASS: chunked reduction with shared memory works for n > workgroup size.

GATE: Atomics and shared memory function correctly.

================================================================================
STAGE 11: DETERMINISM
================================================================================

Goal: Verify identical inputs produce identical outputs across runs.

Test 11.1 — Run every stage 4-10 test 10 times:
  Compare all outputs bit-by-bit across runs.
  Every output must be identical every time.
  No test may produce different results on re-run.

  PASS: all 10 runs produce identical outputs for every test.
  FAIL: identify which operation is non-deterministic.
  Integer arithmetic CANNOT be non-deterministic on the same hardware.
  If results vary: likely an uninitialized buffer read, missing barrier,
  or atomic ordering issue.

Test 11.2 — Softmax determinism (critical):
  1. Run softmax_exact on same input 100 times
  2. Verify output is bit-identical every time
  3. Verify sum == 65536 every time

  PASS: softmax is deterministic and sums to D. INVARIANT 1 holds.

GATE: All operations deterministic across repeated runs.

================================================================================
STAGE 12: CLEANUP
================================================================================

Save test results:
  Copy all PASS/FAIL output to a file.
  Note GPU device name, driver version, Zig version.

Destroy VM:
  gcloud compute instances delete vlp-gpu-test --zone=us-central1-a

Review:
  If all stages pass: GPU integration is validated. Proceed with
  VLP functional development using these kernels.

  If stages 0-4 pass but 5+ fail: arithmetic or kernel logic bugs.
  Fix in vlp_kernel.zig, re-test from the failing stage.

  If stage 1 fails: Zig SPIR-V backend cannot compile the kernel.
  Identify which op fails. Options:
    a) Simplify the failing op
    b) Split into smaller ops
    c) Fall back to GLSL for that specific op
    d) File bug against ziglang/zig with minimal reproducer

  If stage 2 fails: Vulkan environment issue. Check drivers.

  If stage 3 fails: buffer management bug in bridge. Fix host code.

================================================================================
FAILURE TRIAGE DECISION TREE
================================================================================

  Compilation fails?
    → Is it a Zig syntax error? Fix the code.
    → Is it a SPIR-V backend error? Simplify until it compiles.
      → @atomicRmw rejected? Try different atomic ordering (.monotonic).
      → @fence(.workgroup) rejected? Try @memoryBarrier(.workgroup).
      → shared memory rejected? Remove it, use scratch buffer instead.
      → switch with 28 cases rejected? Split into if-else chain.
      → i64 rejected? Check if Int64 capability is being emitted.

  spirv-val fails?
    → Missing decoration? Backend bug. Add decoration manually or simplify.
    → Invalid control flow? Simplify the op's branching.
    → Type mismatch? Check extern struct alignment.

  Dispatch fails?
    → Pipeline creation error? Shader module invalid. Check spirv-val.
    → Descriptor set error? Binding mismatch between kernel and host.
    → Timeout? Infinite loop in kernel. Check all loop bounds.

  Wrong results?
    → Compare host computation vs GPU computation for same input.
    → If they differ: which operation diverges? Binary search by disabling ops.
    → Check buffer offsets: off-by-one in FACT_INTS multiplication?
    → Check ping-pong: is the op reading scratch_a when it should read scratch_b?

  Results vary between runs?
    → Missing pipeline barrier between dependent dispatches.
    → Uninitialized buffer read (add buffer_fill before the operation).
    → Atomic ordering too weak (try .seq_cst).

================================================================================
TEST EXECUTION ORDER
================================================================================

  Stage 0: VM setup             (~15 min, includes reboot)
  Stage 1: SPIR-V compilation   (~5 min)
  Stage 2: Vulkan device init   (~2 min)
  Stage 3: Buffer round-trip    (~2 min)
  Stage 4: Simplest dispatch    (~5 min)
  Stage 5: Arithmetic kernels   (~10 min)
  Stage 6: Fact store kernels   (~10 min)
  Stage 7: Prolog kernels       (~10 min)
  Stage 8: LLM smoke test       (~10 min)
  Stage 9: Sort and matmul      (~5 min)
  Stage 10: Atomics + shared    (~5 min)
  Stage 11: Determinism          (~10 min)
  Stage 12: Cleanup              (~2 min)

  Total: ~90 minutes. Well within 2-hour spot VM budget.

  If any stage fails, stop. Fix. Re-run from the failing stage.
  Do not skip stages — each depends on the previous passing.
```
