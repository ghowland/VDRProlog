**7 turns. Bottom-up dependency order.**

---

**Turn 1 — Foundation types + memory** (~730 lines)
- `vlp_types.zig` (450) — everything else imports this
- `vlp_device_memory.zig` (80) — layout structs + sizing
- `vlp_gpu_params.zig` (200) — all kernel dispatch param structs

No dependencies between these three. Pure data definitions.

---

**Turn 2 — Bridge + LLM** (~250 lines)
- `vlp_bridge.zig` (150) — Vulkan init, dispatch, buffer management
- `vlp_llm.zig` (100) — forward pass orchestration, sampling, KV cache

Bridge depends on types + memory + params. LLM depends on bridge.

---

**Turn 3 — KB + Prolog** (~250 lines)
- `vlp_kb_store.zig` (130) — KB CRUD, fact ops, COW, path index, scoped search
- `vlp_prolog.zig` (120) — unification, query, rule fire, chain building

Prolog depends on KB store. Both depend on bridge.

---

**Turn 4 — Grammar + Builtins + Confidence** (~160 lines)
- `vlp_grammar.zig` (60) — compile, render, inherit
- `vlp_builtin.zig` (70) — dispatch table, IOSE validation
- `vlp_confidence.zig` (30) — assign, combine, chain, propagate

All depend on KB store. Grammar and builtins are independent of each other. Confidence uses bridge for large-N GPU dispatch.

---

**Turn 5 — Session + Snapshot + Safety** (~290 lines)
- `vlp_session.zig` (90) — create, destroy, clone, merge
- `vlp_snapshot.zig` (80) — save, load, diff, capture, restore
- `vlp_grant.zig` (60) — check, create, revoke, cleanup
- `vlp_access.zig` (20) — visibility walk
- `vlp_audit.zig` (40) — ring buffer write, query

Session depends on KB store + bridge. Snapshot depends on session. Grant/access/audit are leaf modules with minimal deps.

---

**Turn 6 — Command + Inference + Runner** (~250 lines)
- `vlp_command.zig` (70) — parse, execute, batch execute
- `vlp_inference.zig` (80) — full cycle, L1/L2/L3, context build, scratchpad
- `vlp_runner.zig` (100) — poller, processor, internal, batch, recycle

Command wires together KB + Prolog + grammar + builtins + grants + access + audit. Inference wires command + LLM + session. Runner wires session + inference.

---

**Turn 7 — System + Multi-device + Seed + Test** (~220 lines)
- `vlp_system.zig` (80) — top-level init, wires all modules
- `vlp_multi_device.zig` (50) — pipeline parallelism, KB replication
- `vlp_seed.zig` (50) — seed KB population
- `vlp_test.zig` (40) — determinism, roundtrip, isolation tests

System depends on everything. These are the capstone files.

---

**Total: ~2,150 lines across 7 turns. ~307 lines avg per turn.**

Ready for turn 1?
