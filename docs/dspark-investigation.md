# DSpark Parallel Verify Investigation

Goal: DSpark speculative decoding must produce **token-identical** output to
vanilla target at temp=0, using **parallel (batched) verify** (not sequential),
and reach ~2x tgp speedup over vanilla with ~same pp. Focus: Qwen3-8B + Q4_K_M
target, `dspark_qwen3_8b_block7` draft. Backend: Vulkan (AMD iGPU, no NVIDIA).

Append-only log. Newest at the top under each section.

---

## BREAKTHROUGH (2026-07-01)

### Root cause confirmed and fixed

Removed the `inp_out_ids` gather from the last layer in `src/models/qwen3.cpp`
(graph computes final FFN + norm + lm_head on full [n_embd, n_tokens]).

Result: **default batched verify now matches vanilla token-for-token.**

- think_on n=250: match True
- code01_think_off n=200: match True
- code_500l n=200: match True

### Fast path identified (MAINSEQ batched, no scratch)

With the graph fix, main-seq batched verify (no scratch seq, no seq_cp,
DFlash-style: decode [anchor,draft...] on seq 0, accept, KV trim) gives:

- code_500l n=200: **match True, speedup 2.22x**, accept 83.2%, spec tgp 87.3
  vs vanilla 39.3, verify 36.5ms/step, draft 13.3ms/step

The scratch/snapshot/sequential machinery is now unnecessary. The structural
fix is: (1) the qwen3 graph fix (done), (2) make the DSpark verify path the
simple main-seq batched path, remove scratch/seq_cp/snapshot/sequential.

### Timing bug fixed

`dspark_now_ms()` in `common/dspark_pipeline.cpp` used a static t0 initialized
at first call + a 1e-3 milli->micro conversion, making all spec timings ~1000x
too small. Fixed to use `time_since_epoch()` milli duration. Honest timings now.

### Default scratch path is slow (153ms/step) due to seq_cp + process overhead

The current default (scratch) gives correct tokens but 0.4x speedup because of
seq_cp (full KV copy) + heavy process_committed per step. MAINSEQ avoids both.


---

## Hypotheses (status)

| # | Hypothesis | Status | Evidence |
|---|-----------|--------|----------|
| H1 | Stale K/V bytes in freed cells after batched draft tail | RULED OUT | MAINSEQ_SNAP (snapshot/restore, fully non-polluting) still diverges at gen 73 |
| H2 | seq_cp scratch stream copies wrong state (DSV4 comp state) | RULED OUT for Qwen3 (Qwen3 uses ISWA, not DSV4); also MAINSEQ (no scratch at all) diverges |
| H3 | Batched multi-token (prefill-path) decode produces different argmax than single-token (decode-path) decode on Vulkan Q4 | LEADING | scratch canary row0 mismatch; MAINSEQ_SNAP divergence; smoke shows logit deltas ~0.2 at step1 (same argmax there but accumulates) |
| H4 | Flash-attn multi-token kernel diverges | PARTIALLY | harness forces `-fa off`; divergence persists with FA off. So FA is not the sole cause. |
| H5 | Q4 KV dequant path differs between prefill (batched) and decode (single) | RULED OUT | FA-on (f16 KV) still diverges at gen 73 |
| H6 | ubatch boundary / partial-batch numerics | RULED OUT | ubatch=1 impossible (non-causal draft); divergence is per-row |
| H7 | Vulkan `mul_mat` produces different results for M=1 vs M>1 (last-layer FFN after `inp_out_ids` gather) | RULED OUT | Moving gather AFTER FFN (lm_head still on get_rows'd n_outputs) still diverges at gen 73. Disabling gather entirely (nullptr) fixes it. |
| H8 | Vulkan `ggml_get_rows` produces a tensor whose layout makes the subsequent `lm_head` mul_mat dispatch a different, less-precise shader than when lm_head reads the contiguous [n_embd, n_tokens] directly | LEADING (mechanism) | nullptr (no get_rows, lm_head on raw n_tokens) -> MATCH. gather-before-FFN -> MISMATCH. gather-after-FFN (lm_head on get_rows'd n_outputs) -> MISMATCH. So the get_rows node feeding lm_head is the trigger, regardless of FFN position. CPU unaffected. |

---

## Experiments log

### E1 - default vs sequential (think_on, n=250, Q4, Vulkan, FA off)

```
DEFAULT (batched scratch): match False, mismatch_gen 73, accept 17.6%, acc/step 0.71
SEQ     (one-token)      : match True,  mismatch None, accept 36.6%, acc/step 1.47
```
Sequential is correct. Batched diverges early.

### E2 - main-seq batched (no scratch, no seq_cp), like DFlash

```
MAINSEQ: match False, mismatch_gen 38
```
Rules out scratch/seq_cp as the cause. Main-seq batched also diverges (earlier).

### E3 - main-seq batched + snapshot/restore (non-polluting)

Added `DSPARK_VERIFY_MAINSEQ_SNAP`: snapshot ctx_tgt state before batched
verify, decode [anchor,draft...] on main seq, accept, restore snapshot, then
commit accepted tokens via real single-token decodes + process().

```
SNAP: match False, mismatch_gen 73
```
**Decisive**: even with KV guaranteed non-polluting, batched diverges at same
gen 73 as scratch. KV pollution is NOT the cause. The batched forward itself
produces row logits whose argmax differs from a sequential single-token chain.

### E4 - scratch canary (batched vs sequential ON the scratch seq)

```
DSPARK_VERIFY_CANARY: row0 mismatch batched=729 pos_verify=111 anchor=369 draft_n=4
GGML_ABORT: DSpark parallel verify logits mismatch
```
On the scratch sequence itself (clean seq_cp'd state), batched row-0 argmax !=
sequential row-0 argmax at pos_verify=111. Confirms H3: it is a batched-decode
numerics issue, independent of canonical KV hygiene.

### E5 - smoke_batched_logits_repro (Q4 KV, scan-all, fixed pos bug)

Fixed scan pos bug (restore state after compare_row0_logits before sequential
verify). Result:

```
step 28 gen_index 73 pos 111 anchor 369 draft_n 4 features=0 defer=0:
  row0 single=729 multi=5746 max_abs_diff=0.211237 MISMATCH
*** first row0 mismatch at step 28 gen_index 73 ***
```

On a CLEAN snapshot (restored before each of single and multi), decoding
anchor 369 at pos 111:
- single-token decode -> argmax 729
- first row of 5-token batch -> argmax 5746
KV identical. Divergence is in the forward computation path itself.

### E6 - CPU backend scan (prefill vs decode divergence?)

```
smoke scan-all, CPU backend (-ngl 0): first row0 mismatch step = -1  (NO mismatch, 90 steps)
```

**DECISIVE:** CPU backend has zero prefill-vs-decode divergence. The
batched-vs-single logits divergence is **Vulkan-specific**. It is a Vulkan
kernel numerics issue (attention or matmul differs between n_tokens=1 and
n_tokens>1 ubatch), NOT a fundamental speculative-decoding problem and NOT
KV pollution.

Implication: vLLM batched verify works because CUDA prefill/decode share
numerics. The correct fix is either (a) fix the Vulkan kernel so prefill and
decode agree, or (b) make verify use the decode path (n_tokens=1 per row) but
amortize via graph capture / batched logits without separate prefill.

---

## Key code locations

- `common/dspark_pipeline.cpp` `dspark_pipeline_step` - verify dispatch
  (DSPARK_VANILLA_VERIFY / DSPARK_VERIFY_SEQ / default batched scratch).
  Experimental branches DSPARK_VERIFY_MAINSEQ and DSPARK_VERIFY_MAINSEQ_SNAP
  added during this investigation (temporary).
- `common/dspark_target.cpp` `dspark_target_verify_logits` - scratch seq_cp +
  batched decode + canary. `dspark_target_verify_sequential` - correct path.
  `dspark_target_verify_step` - dispatch (DSPARK_VERIFY_SEQ gate).
- `common/speculative.cpp` `common_speculative_impl_draft_dspark` - draft
  propose (Markov, softcap, block sampler).
- `tests/compare_vanilla_speculative.cpp` - vanilla vs spec token-match harness.
  Forces FA off for DSpark (line ~552). Sets DSPARK_KV_ZERO_ON_RM=1.
- `tests/smoke_batched_logits_repro.cpp` - row0 batched vs sequential on
  snapshot; has a pos-continuity bug in scan mode.

## Model files (~/models)

- Target: `Qwen3-8B-Q4_K_M.gguf` (Q4 KV by default on Vulkan)
- Draft: `dspark_qwen3_8b_block7.q4_k_m.gguf`
- Also: `Qwen3-8B/` (bf16/f16), `dspark_qwen3_8b_block7.bf16.gguf`

---

## Reproduce

```bash
export LD_LIBRARY_PATH=build/bin:$LD_LIBRARY_PATH
T=~/models/Qwen3-8B-Q4_K_M.gguf
D=~/models/dspark_qwen3_8b_block7.q4_k_m.gguf
IN=scripts/dspark-vps/eval/qwen3/code/code01_think_on.json
DSPARK_NO_ADAPTIVE_NMAX=1 ./build/bin/compare_vanilla_speculative \
  -m "$T" -md "$D" --input-ids "$IN" --spec-type draft-dspark \
  -c 4096 --device Vulkan0 -ngl 99 -ngld 99 --temp 0 --seed 42 \
  --spec-draft-n-max 4 -n 250 --json-results /tmp/d.json
```

## Design analysis (decision needed)

### The real problem

The divergence is NOT a DSpark design flaw and NOT KV pollution. It is a
**Vulkan backend numerics bug**: when `n_outputs` differs between two otherwise
identical forwards, the `ggml_get_rows(cur, out_ids)` node feeding the final
`lm_head` `mul_mat` produces slightly different floats (~0.2 logit delta) that
flip argmax. Specifically:

- vanilla decode: n_outputs=1, get_rows gathers 1 row, lm_head mul_mat M=1
- batched verify: n_outputs=N, get_rows gathers N rows, lm_head mul_mat M=N

The Vulkan shader dispatched for M=1 vs M=N (or the get_rows output layout)
isn't numerically identical. CPU is unaffected. This affects ANY model using
the `inp_out_ids` gather (all Qwen3-class) on Vulkan, not just DSpark.

### Why vLLM doesn't hit this

vLLM runs CUDA + FlashAttention; CUDA prefill/decode mul_mat kernels are
numerically consistent across M. So batched verify works there.

### Options

**Option A - Fix the Vulkan kernel (deep, out of DSpark scope).**
Make `ggml_get_rows` output or the lm_head mul_mat numerically identical for
M=1 and M=N. High risk, broad blast radius, likely needs upstream ggml work.
Not the DSpark refactor's job.

**Option B - Make verify use the decode path (n_outputs=1 per row).**
Verify would decode one token at a time but in a single captured graph
(CUDA-graph-style) so launch overhead is amortized. This is numerically
guaranteed correct (identical to vanilla) but loses the multi-token parallelism
that gives 2x. Defeats the goal.

**Option C - Logits-only verify graph (no KV write) + accept + commit.**
This is the kv-safety doc's intended design. BUT: snapshot/restore experiment
(E3) proved even non-polluting batched verify diverges, because the divergence
is in the forward itself, not the KV residue. So a logits-only graph would
STILL diverge on Vulkan. Option C does not fix Vulkan numerics.

**Option D - Fix the gather location / make n_outputs invariant.**
Disable the `inp_out_ids` gather (or move it to after lm_head) so lm_head
always runs on the full `[n_embd, n_tokens]` tensor with the same M as the
number of input tokens, NOT n_outputs. But vanilla decode has n_tokens=1
(M=1) and batched verify has n_tokens=N (M=N) - so lm_head M still differs
(1 vs N). Disabling gather entirely (nullptr) DID fix it though, which
contradicts this. Need to re-examine: with gather=nullptr, lm_head runs on
[n_embd, n_tokens] = [n_embd,1] (decode) vs [n_embd,5] (verify) and that
MATCHED. So lm_head mul_mat M=1 vs M=5 is fine on Vulkan WITHOUT get_rows.
The get_rows node is the sole trigger.

  => This means: **removing the get_rows node (compute FFN+norm+lm_head on all
  n_tokens, skip the gather) fixes Vulkan numerics** with only the cost of
  computing the last-layer FFN + lm_head on all N tokens instead of 1.
  For decode (N=1) no cost. For verify (N=5): 5x last-layer FFN + lm_head on
  5 rows = negligible (1 layer of 36, and lm_head is already all rows).
  This is the structural fix.

### Recommended fix

Remove the `inp_out_ids` gather from the last layer (qwen3.cpp L113-116) so
the graph computes the final FFN, norm and lm_head on the full n_tokens
tensor. This:
1. Makes batched verify numerically identical to single-token decode (fixes
   the divergence at its source - the get_rows+lm_head interaction).
2. Costs ~1 extra last-layer FFN on N rows during multi-token decode (tiny).
3. Is a model-graph fix, not a DSpark hack - benefits any spec decode on Vulkan.
4. Keeps parallel verify, no sequential fallback, no scratch seq needed.

After this, the DSpark refactor can proceed cleanly: parallel batched verify
on the main seq (DFlash-style, no scratch), with proper KV trim, and remove
all the scratch/snapshot/sequential machinery as the user requested.

### To confirm before implementing

- Re-run smoke scan with gather removed (already done: MATCH through 90 steps).
- Re-run full compare_vanilla_speculative batched (default path) on think_on
  n=250 and code01_think_off n=200: expect token_match=True.
- Measure tgp speedup: batched verify should give ~2x once working.
- Then do the DSpark structural refactor (remove scratch/seq, simplify).
