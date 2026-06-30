# DSpark: KV safety refactor (prevent corruption without losing speed)

This document describes how to refactor DSpark speculative decoding so **target KV corruption and verify/KV race conditions are structurally impossible** under normal use. Performance optimizations (batched verify, defer, fused argmax) remain allowed only inside explicitly non-canonical buffers.

**Related:** root-cause writeup in [dspark-benchmark-qwen3.md](dspark-benchmark-qwen3.md) (Token match investigation). Pipeline shape in [dspark-refactor-pipeline.md](dspark-refactor-pipeline.md).

---

## Problem statement

### What went wrong (2026-06-30)

Batched target verify ran one parallel forward `[anchor, draft0, draft1, ...]` on **canonical target KV** (`ctx_tgt`, seq 0). That forward **wrote every hypothetical position into K/V tensors**. Rejected tail was dropped with `llama_memory_seq_rm()`, which clears **cell metadata only**, not K/V bytes in recycled cells. Attention later read stale K/V; greedy output diverged from vanilla after many steps.

Draft KV (`ctx_dft`) was never the issue. **Target canonical KV ownership was violated during verify.**

### Interim fix (current branch)

Scratch-sequence batched verify (`seq_id + 1`):

1. `llama_memory_seq_cp` canonical prefix to scratch stream
2. Batched forward on scratch only
3. Accept from logits; discard scratch tail
4. Append accepted tokens on main seq via `process_committed(..., kv_append_only=true)`

This restores the invariant on seq 0 but still shares one `llama_context`, relies on `n_parallel >= 2`, and batched logits can still diverge from one-token decode on edge prompts (e.g. `code01_think_off` @ gen 162). Sequential verify (`DSPARK_VERIFY_SEQ=1`) remains the correctness backstop.

---

## Design principle: three memory domains

Split all DSpark memory into **three domains** with hard API boundaries. No function may touch more than one domain per phase, except explicit copy/sync helpers.

| Domain | Owner | May write KV? | Purpose |
|--------|-------|---------------|---------|
| **Target canonical** | `ctx_tgt`, seq `S_main` | Yes, append-only, one token per commit | Ground-truth autoregressive prefix for output |
| **Target verify scratch** | `ctx_tgt`, seq `S_scratch` OR dedicated `ctx_verify` | Yes, ephemeral; discarded each step | Batched logits for draft block |
| **Draft working** | `ctx_dft`, seq 0 | Yes; rebuilt/trimmed from `n_past` | Draft proposals + injected target features |

**Frozen weights** are unchanged. Bugs are **cache ownership**, not matmul.

---

## Invariants (enforce in code and debug builds)

These should be `GGML_ASSERT` or cheap checks in debug/CI, not comments.

### I1 - Canonical append-only

After any successful propose step, for main target seq `S_main`:

```
kv_max(S_main) == n_past - 1
```

Positions `[0 .. n_past-1]` are exactly the committed output prefix. No holes, no draft hypotheticals.

### I2 - Verify never writes canonical KV

Any batched or parallel target forward used **only to read logits** must run on `S_scratch` (or a logits-only graph with KV writes disabled). Canonical seq must be unchanged across verify:

```
kv snapshot hash before verify == after verify   (canonical seq only)
```

### I3 - Commit is the only canonical mutator

Only `target_commit_token()` (new API, see below) may extend canonical KV. Exactly one `llama_decode` of one token per call.

### I4 - Scratch is reset every step

Before each verify, scratch domain is either:

- Full stream cleared + prefix copied from canonical, or
- Separate context reloaded from canonical snapshot

Never reuse scratch cells across steps without full buffer copy.

### I5 - Draft cannot affect target logits

Draft proposals are inputs to verify **accept logic only**. Wrong draft tokens must be rejected; they must never appear in canonical KV.

### I6 - Feature bridge reads committed target only

`dspark_process()` reads layer inputs from target decodes that correspond to **already committed** positions on canonical KV (or from the commit pass immediately after verify decision).

---

## Target API refactor (`common/speculative.cpp` / new header)

Replace the monolithic `common_speculative_dspark_verify_batched()` with explicit phases:

```cpp
// Phase A: logits only - MUST NOT touch canonical KV
struct dspark_verify_logits_result {
    std::vector<std::vector<float>> row_logits;  // or indices into ctx output buffer
    int n_rows;
};

bool dspark_target_verify_logits(
        dspark_memory_bundle * mem,
        llama_pos pos_verify,
        llama_token anchor,
        const llama_tokens & draft,
        dspark_verify_logits_result & out,
        llama_batch & batch);

// Phase B: accept - pure CPU / sampler, no llama_decode
llama_tokens dspark_target_accept_chain(
        common_sampler * smpl,
        const dspark_verify_logits_result & logits,
        const llama_tokens & draft,
        float temp);

// Phase C: commit - ONLY mutator of canonical KV
bool dspark_target_commit_tokens(
        dspark_memory_bundle * mem,
        llama_pos pos_verify,
        llama_token anchor,
        const llama_tokens & committed,
        llama_batch & batch);

// Phase D: draft feature inject (existing process(), scoped to draft domain)
bool dspark_draft_process_committed(
        common_speculative * spec,
        const llama_tokens & committed,
        llama_pos pos_first,
        llama_batch & batch);
```

**Rule:** `dspark_target_verify_logits` has no path to `seq_id == S_main` in its batch. Code review + assert guard.

### `dspark_memory_bundle`

Centralize contexts and seq ids:

```cpp
struct dspark_memory_bundle {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;
    llama_context * ctx_tgt_feat;  // optional, shared KV with ctx_tgt
    llama_seq_id    seq_main   = 0;
    llama_seq_id    seq_scratch = 1;
    llama_pos       n_past     = 0;
};
```

All DSpark code takes `dspark_memory_bundle*` instead of ad-hoc `ctx_tgt` + magic seq ids.

---

## Performance without sacrificing safety

### Keep batched verify (fast path)

Batched parallel forward stays on **scratch only**:

```
seq_cp(main -> scratch, full prefix)
batched_decode(scratch, [anchor, draft...])  -> logits[n_rows]
accept(logits, draft) -> committed[]
for t in committed: commit_one(main, t)
```

Cost vs old broken path:

| Component | Old (broken) | Safe scratch | Notes |
|-----------|--------------|--------------|-------|
| Batched logits | 1x forward on main | 1x forward on scratch + seq_cp | seq_cp is memcpy of KV buffers; cheap vs transformer |
| Canonical update | rm + re-decode all accepted | 1 decode per accepted token | Same as sequential verify |
| Rolling snapshot | full state get/set per step | None | Removed |

seq_cp cross-stream copies K/V tensors once per step. On GPU this is bandwidth, not extra transformer layers. Still much faster than sequential verify (N decodes for N draft rows).

### Optional faster paths (still safe)

| Optimization | Safe when | Unsafe when |
|--------------|-----------|-------------|
| `defer_layer_inp_extract` during verify | Never on canonical; optional on scratch if logits validated | Defer on canonical or without row-0 canary |
| Fused / GPU greedy argmax | After logits materialized on scratch; accept is read-only | Fused graph writes KV |
| Second target ctx (`ctx_tgt_feat`) for layer taps | Shares KV via `ctx_other`; taps only during **commit**, not verify logits | Layer taps during scratch verify without equivalence proof |
| Logits-only graph (no KV write) | llama.cpp adds `LLAMA_DECODE_FLAG_LOGITS_ONLY` or separate graph builder | - |

### Long-term: logits-only decode in llama.cpp

Best performance + safety: extend `llama_decode` / memory layer with **non-mutating verify forward**:

- Build attention graph that reads existing K/V for prefix
- Compute logits for hypothetical tokens using **temporary** K/V in separate tensors or stream
- Never insert hypotheticals into canonical cell pool

This removes seq_cp per step and scratch stream entirely. Track as upstream issue/PR to ggml-org/llama.cpp.

---

## Hardening llama.cpp KV layer (upstream)

DSpark cannot fully fix **global** stale-cell reuse. Recommend upstream changes (optional but valuable):

1. **Zero K/V on cell free** in `llama_kv_cache::seq_rm` when cell becomes empty (debug: always; release: optional flag).
2. **`llama_memory_seq_cp` validation** - assert cross-stream copy completes before next decode (`memory->apply()` fence).
3. **Debug mode `LLAMA_KV_CHECKSUM`** - per-seq hash of canonical prefix after each commit; CI asserts monotonic append.

DSpark CI should run with (3) on representative prompts.

---

## Debug and CI gates

### Runtime checks (env-gated)

| Flag | Behavior |
|------|----------|
| `DSPARK_KV_ASSERT=1` | Assert I1 after every commit; assert canonical unchanged across verify |
| `DSPARK_VERIFY_CANARY=1` | After scratch batched decode, row-0 greedy must match single-token forward on scratch; else fallback sequential |
| `DSPARK_TRACE_KV=1` | Log `kv_min/kv_max` per domain (existing) |

### CI tests (must pass before merge)

1. **Token match** - `compare_vanilla_speculative` vs vanilla, temp=0, all eval fixtures, n=200+
2. **KV monotonic** - custom test: canonical `kv_max == n_past-1` every step
3. **Verify isolation** - canonical KV hash identical before/after `dspark_target_verify_logits`
4. **smoke_batched_logits_repro** - batched row logits == sequential on clean snapshot
5. **No env workarounds** - pass without `DSPARK_VERIFY_PRE_SNAP`, `DSPARK_VERIFY_SEQ`

---

## Migration plan (phased)

### Phase 1 - API split (no behavior change)

- Introduce `dspark_memory_bundle`, `dspark_target_verify_logits`, `dspark_target_accept_chain`, `dspark_target_commit_tokens`
- Implement as thin wrappers around current scratch verify
- Single call site in decode loop

### Phase 2 - Asserts and canary

- I1/I2 checks in debug builds
- Row-0 canary with sequential fallback (correctness over speed on mismatch only)

### Phase 3 - Remove legacy paths

- Delete rolling snapshot / `verify_kv_canon`
- Remove batched verify on main seq (grep guard)
- Default batched = scratch only; sequential = explicit opt-in perf mode

### Phase 4 - Upstream logits-only verify

- Prototype `llama_decode` no-KV mode or second lightweight context
- Drop seq_cp from hot path when available

---

## Anti-patterns (never again)

1. **Batched target decode on canonical seq** for speculative verify
2. **`seq_rm` as "undo"** for rejected draft tail on canonical KV
3. **Mixing verify + commit** in one function without scratch isolation
4. **Defer / fused paths** that skip synchronizing logits before accept without equivalence tests
5. **Implicit seq id 0** everywhere; always use named `seq_main` / `seq_scratch`

---

## Summary

Corruption happened because verify **read logits by writing draft positions into canonical target KV**. Safety and speed are compatible if:

- **Canonical KV** = append-only, one token per commit API
- **Verify** = scratch or logits-only, discarded every step
- **Draft** = separate context; proposals never touch target KV

The scratch-sequence fix is a step toward this model. Full refactor makes violations **unrepresentable** in the API, not merely avoided by convention.
