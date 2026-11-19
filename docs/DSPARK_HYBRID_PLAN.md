# Hybrid DSpark implementation plan

**Status:** partly implemented (host-side policy, pack tooling, Terminal and engine help). The engine placement, the CPU miss lane and all measurements are not started. No hybrid speed or memory measurements yet. See [Implementation status](#implementation-status).

## Goal and scope

Make DSpark a practical option when its routed experts compete with the DeepSeek verifier for VRAM. Keep the dense work of all three draft layers on the GPU, run nonresident 4-bit experts on the CPU, and keep the most-used experts in a small GPU cache across turns. Start with a single discrete GPU and a GGUF-based DeepSeek V4.1-Flash pack; qualify CUDA and HIP explicitly. Leave existing behavior unchanged by default. Multi-GPU placement, unified memory tuning, Metal, and automatic enablement are follow-ups.

| Proposed interface | Behavior |
| --- | --- |
| `--dspark on --dspark-expert-cache-mib 0` | Opt into hybrid placement. All routed draft experts run on the CPU; the dense path stays on the GPU. |
| `--dspark on --dspark-expert-cache-mib N` | Same placement, with at most `N` MiB for routed draft expert slots. The cap includes spare slots; the rest of the expert budget goes to the verifier. |
| `--dspark-depth N` (existing) | Verify at most `N` draft tokens per round. Start trials at 1, 2, and 3. Confidence, remaining token budget, and the pack's block width can lower the effective depth. |

Omitting the new flag preserves current placement and `--dspark auto`. Require explicit `--dspark on` for hybrid mode, and reject invalid sizes or unsupported configurations with an actionable message. Keep `--dspark-block` and `DS4_DSPARK_MAX` compatible; neither is a compute-width setting today. `--dspark none` is the unloaded baseline; `--dspark off` still loads the draft model.

## 1. Establish the 4-bit pack and baseline

- Start with the checkpoint's native `MXFP4` routed weights. Keep the installed drafter's dense precision for the first placement comparison so acceptance changes do not confound the offload result. Use the same pinned target model and draft weights in the reference and hybrid arms.
- Build a 4-bit-heavy variant: retain `MXFP4` experts and quantize eligible large dense matrices to `Q4_K` with [`drafter_pack.py`](../tools/dsv41/drafter_pack.py). Keep the gate, norms, confidence and Markov heads, and tensors that do not fit the quant's block shape at their supported precision. Do not requantize the target. Record effective types, source revision, and bytes per tensor class. Verify the loader can handle the chosen types and the attention output projection slices in the actual pack, not just in isolation.
- Test [`pack_gguf.py`](../tools/dsv41/pack_gguf.py) with both draft artifacts present: its GGUF auto-discovery can currently take precedence over `--drafter`. Make explicit selection unambiguous, with a packer regression test, while preserving discovery when no choice is given.
- Use [`dspark_ref.py`](../tools/dsv41/dspark_ref.py) and the engine's `--spec-eval` / `DS4_ROUND_TIMES` hooks to capture a reference for proposals, confidence, acceptance by depth, and draft/verify cost. Measure the native-dense and `Q4_K` variants separately before shipping either configuration.

## 2. Land the CPU-miss path with zero slots

- In [`ds4_main.cpp`](../src/dsv41/ds4_main.cpp) and [`ds4_model.hpp`](../src/dsv41/ds4_model.hpp), plumb an optional DSpark-only cache budget; distinguish unset from zero. In [`ds4_model.cu`](../src/dsv41/ds4_model.cu), make `place_experts` respect that budget for `mtp.*`. Audit both the first allocation pass and the leftover-slot pass so unused bytes cannot silently refill draft layers. Keep existing dense placement, context/KV allocation, and VRAM reserves. Log the effective bytes for dense draft tensors, draft slots, verifier slots, and host draft weights.
- Give the drafter a read-only RAM backing for all three layers' routed experts, including resident ones, so eviction never requires a GPU-to-host weight copy. Budget it before sizing the verifier's host store, check memory/commit headroom, and fail explicitly if it cannot fit. Do not silently turn draft misses into SSD reads. Keep only the small copy staging buffers pinned in host memory.
- Reuse `layer()` and the CPU expert lane: route on the GPU, group cold rows by expert, run misses on the CPU, and merge with the GPU shared and routed outputs using the existing weights. Check both the async mailbox path and the sync fallback. Give draft matrix lookup and promotion reads the same RAM source; do not put the draft weights in the verifier's RAM/SSD store, lookahead queue, or generic warm-up list. The store currently describes trunk layers only.
- **Gate:** a real-pack greedy run crosses several draft/verify rounds with zero draft expert slots, no draft weight reads from the model file during decode, and no additional GPU allocation on a miss. Check rollback, EOS, and context-boundary behavior against the reference before adding swaps.

## 3. Add the live histogram and hot slots

- Allocate deterministic per-layer quotas from the byte budget using actual aligned expert blob sizes. Reserve one spare per partially cached layer; a quota too small for a usable slot plus a spare gets no adaptive slots. A fully resident layer needs no spare. Leave unused budget to the verifier. Use deterministic startup ordering, with a compatible draft profile if available.
- Count GPU hits and CPU misses for each `(draft layer, expert)` in both routing paths. Count executed routes even if a proposal later gets rejected: the expert already cost time. Use decayed frequency for admission, last-use order as the eviction tie-breaker, and a minimum reuse threshold and score margin to avoid churn. Age by draft rounds rather than wall time so a pause for a tool call does not erase locality. Keep the histogram and residency on the loaded model, not in conversation/KV snapshots. Clear them when the pack is unloaded or replaced.
- Reuse the existing cache worker and spare-slot publication pattern. Copy a hot expert from RAM off the miss path; after completion, publish the host/GPU slot mapping at a safe window boundary. Do not reuse an evicted slot until kernels that could read it have finished. After a copy failure, keep the old mapping and continue to serve misses from RAM. Bound concurrent copies and bytes promoted per interval; reuse idle time where available without blocking the next request. Honor `--no-expert-cache` by keeping placement fixed.
- Update draft cache maintenance at a completed draft boundary even when confidence rejects every proposal and the engine falls back to a plain step. Keep draft hit/byte/promotion counters distinct from trunk counts. Add synthetic tests for stable ties, decay, byte caps, swaps in flight, failed copies, and cache retention across turn/KV resets.

## 4. Expose depth and qualify the complete path

- Document `--dspark-depth` in the engine help and Terminal parameter list, and test that it and the new budget survive launcher/desktop argument filtering. Keep depth process-scoped for this release. Depth 1 verifies up to two target rows (the seed and one proposal); it does **not** skip draft layers or reduce DSpark's full-block CPU expert work. A shorter compute block is a separate experiment because every draft row attends to the whole block. Do not change the meaning of the current flag.
- Add tests for CLI bounds and incompatible options, quant metadata/byte layout, 0/tiny/partial/full slot budgets, CPU/GPU output tolerances on identical packed experts, and exact greedy verification fixtures with fixed verifier placement. Include rejection/rollback, sampled verification, context limits, cancellation, and a server sequence with two turns, an idle tool gap, and a conversation reset. When the verifier's CPU/GPU expert placement changes, check logits and token mismatches separately: numeric differences between lanes are not evidence of a speculative verification bug by themselves. Run the touched backend builds and confirm that omitting the opt-in flag matches the unchanged engine path.

## Benchmark and go/no-go

Run paired cases with one pinned target pack, fixed total VRAM reserve, context, fixture, sampling, and CPU thread count. Compare: **unloaded**, **current DSpark placement**, **hybrid with zero slots**, and **hybrid with a small adaptive cache**; add a fixed-placement arm at the same draft budget to isolate the benefit of promotion. Sweep depths 1–3 with identical quantization first, then compare `Q4_K` dense against the reference at the best settings. For an isolation run, hold verifier placement fixed too. Use fresh loads for cold runs and separate warm multi-turn chat, code, and tool-call runs. Follow the [benchmark protocol](BENCHMARK_PROTOCOL.md) for timing and report the multi-turn replay as a separate experiment.

Record emitted tokens/s, p95 inter-token latency, time to first token, draft/verify time, acceptance by position, separate VRAM and RAM usage, and draft cache hit rate, distinct CPU expert bytes per emitted token, and promotion bytes/time. Note whether the extra draft RAM causes more verifier storage reads. Test on a VRAM-constrained discrete GPU first and repeat on a second supported backend before claiming broad gains.

**Proposed gate:** keep the adaptive cache only if it beats zero-slot and fixed placement in end-to-end decode on warm runs. Promote the hybrid mode beyond an experimental flag only on hardware where it clears correctness, stays within memory limits, and gives a repeatable useful win (target: at least 5% more emitted tokens/s with no more than 5% p95 latency regression against the faster available baseline). If all variants lose, report whether draft CPU time, promotion traffic, or verifier expert misses consumed the gain; leave `auto` and installer defaults unchanged.

## Implementation status

Stages 1 and 4 (the Terminal and help parts) and the host policy are as described below. Stages 2 and 3 are written and
compile-checked, but **have not run on a GPU**, so none of their behaviour or speed is measured. Nothing here changes the default
path, `--dspark auto`, or the installer: every new branch is behind `--dspark-expert-cache-mib`.

- **Stage 1, pack tooling.** `pack_gguf.py`: an explicit `--drafter` now wins over a `dspark-*.gguf` found beside the model, which
  used to be ignored without a message; naming both `--dspark-gguf` and `--drafter` is an error, and so is an explicit drafter
  whose `.bin` is missing. Discovery is unchanged when no choice is given. `drafter_pack.py --requant`: a dense matrix whose rows
  are not whole blocks of the quant keeps its own type (it used to abort the run), norms, gates and heads are never requantized,
  and the run prints the bytes and types per class. The checkpoint route did not carry the drafter's geometry (block size, target
  layers, noise token, markov rank, expert counts), so a pack built from it loaded no drafter: `drafter_pack.py --config` (default
  `<dir>/config.json`) writes those values from DeepSeek's `config.json` into `dspark.json`, requantize carries them over, and
  `pack_gguf.py` puts them in the pack. A drafter without them gets a warning. Tests: `tests/terminal/test_pack_gguf_drafter.py`,
  `tests/terminal/test_drafter_pack_requant.py`, `tests/terminal/test_drafter_geometry.py`. The requant path was also run end to end on a synthetic drafter with a shared
  `libggml-base` built from the pinned ggml: `HEAD` aborts on the non-fitting matrix, the new code keeps it and quantizes the rest.
- **Stage 2 and 3, engine (`src/dsv41/ds4_model.cu`, `ds4_main.cpp`, `ds4_model.hpp`).** What was written:
  - *Flag.* `--dspark-expert-cache-mib N` (unset: -1; `0` is the zero-slot arm). It needs `--dspark on` (refused with `auto`,
    `off` or `none`), and the engine refuses a pack without a drafter, a non-MXFP4 (EXL3) pack, and Metal.
  - *Placement.* `place_experts` gives the drafter's layers their share from `plan_draft_quotas` under the cap (equal shares, whole
    aligned slots, a spare per partial layer); the verifier gets what is left. The leftover pass does not refill draft layers.
    The placement logs the draft slots and the verifier's share.
  - *RAM backing.* `draft_backing` reads every routed draft expert, resident or not, once into anonymous memory of its own (not
    pinned, not a verifier copy, store or RAM-tier slot). It runs before the verifier's copies are sized and refuses with the
    numbers when the RAM does not hold it beside the 8 GB margin. `mx_mat`, `read_mx`, the CPU lane's prefetch and `file_read`
    (which decides the lookahead's file reads) all read the backing first, so no draft expert is read from the model file during
    decode, as far as reading the code can show.
  - *Cache.* The draft cache (`DraftExpertCache`, host policy above) starts from the placement's residents. Each route the drafter
    executes is counted in `layer()`, on the GPU or the CPU lane. At the end of every draft it proposes up to `DS4_DRAFT_SWAPS`
    (default 2) copies, which the existing cache thread makes from the RAM backing; `draft_publish` switches the mapping at the next
    cache tick, as the trunk does, and reports a failed copy as failed. The trunk's heat rule skips draft layers when the hybrid is on.
  - *Counters.* `draft_gpu_routes`, `draft_cpu_routes`, `draft_promotions`, `draft_promo_bytes`, `draft_failed`, printed at the end
    of a run. The trunk's `swaps` are not touched.
  - *Checks run.* The engine compiles with clang-19's CUDA frontend and device path (`-fsyntax-only`, and `--cuda-device-only -S`)
    against the CUDA 12.9 headers from the pip wheels, with no warnings beyond the three that `HEAD` already has. `ds4_main.cpp`
    passes a host syntax check. The policy test passes, including the new adopted-resident case. This is a compile check only:
    clang's CUDA frontend is not `nvcc`, and nothing was linked or run.
  - *Not verified.* The publish path, the cache thread, the CPU lane for draft layers, the byte-identical default, the greedy
    gate and every speed number. The reuse of an evicted slot relies on the window boundary, as the trunk does; that has not been
    checked on a device.
- **Stage 3, host policy.** `include/strata/core/dspark_expert_plan.hpp` and `src/core/dspark_expert_plan.cpp`:
  `plan_draft_quotas`, `DraftExpertCache` (decayed routing frequency by draft rounds, last use as the tie-break, a minimum score and
  a margin against churn, failed copies cool down and keep the old mapping, `adopt_resident` for the placement's experts). Its test,
  `tests/core/dspark_expert_plan_test.cpp`, runs under CTest as `dspark_expert_plan_test` in the CPU-only test section. The knobs
  (decay 0.97 a round, minimum 2, margin 1.25, cooldown 16 rounds; env `DS4_DRAFT_*`) are starting points that nothing has measured.
- **Stage 4, help text and plumbing.** `--dspark-depth N` and `--dspark-expert-cache-mib N` are in the Terminal's parameter table and
  the engine's usage line; the desktop app's value-flag list has both. Tests: `tests/terminal/test_dspark_depth_flag.py`,
  `tests/terminal/test_dspark_expert_cache_flag.py`. The desktop filter's `npm test` was not run (Node is not installed here).

Not done, and why:

- **Real packs and the gate.** No `Q4_K` pack has been built from a checkpoint, no loader check has run on one (including the
  attention output projection slices), `dspark_ref.py` has not captured a reference, and the go/no-go runs and the benchmark matrix
  have not run. All of that needs a GPU and the model.
- **Stage 4, qualification.** Depth 1 to 3 sweeps, CLI bounds for the budget beyond the checks above, the two-turn server sequence,
  and the idle-gap and reset tests are not written.
- **Chunked prefill.** The drafter's blocks never run in the chunked prefill path (it loops over the trunk's layers only), so
  its routes are counted only on the window path, which is where the drafter runs.
- **Logging gaps.** The placement logs draft expert slots, not the dense draft tensors' bytes.

Pre-existing, found while reading and not changed (the trunk's behaviour, which the default path keeps):

- `cache_loop` queues a trunk swap whether or not its copy succeeded (`ds4_model.cu`, `cache_loop` and `cache_tick_swaps`); a
  failed copy is still published, so the mapping points at whatever the spare held. The draft path publishes only good copies.
- The trunk's publish writes `tab_pin` and then copies from it asynchronously. An async copy reads its source when it runs, so when
  two swaps publish in one tick the first copy may read the second swap's words. Not reproduced (no GPU here). The draft path
  synchronises its own writes.
