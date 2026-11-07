# Expert tier for the DeepSeek-V4.1 engine (strata-dsv41): plan

Status: phases D1-D6 are in the tree and host-tested (no GPU, no model: `ds4_expert_tier_test` plus the shared tier
tests pass; the fit and setup tests cover the flags). Phase D7 (byte-identical answers on the real model) and D8
(speed) wait for hardware with the model — still no speed number here. Every number marked **target** is a goal to
test, not a result. This plan is the DeepSeek companion of
[EXPERT_TIER_PLAN.md](EXPERT_TIER_PLAN.md). Read that plan first. This document covers only what differs for
`src/dsv41`, and says where the two plans share code.

## 1. Summary

The DeepSeek-V4.1 engine already has the three bands that the expert tier sits between: VRAM experts, a RAM store, and
file reads. The tier adds a fourth band below the RAM store and above the file reads. It holds the experts the store
cannot keep. Each device is configured on its own. Its contents are placed once from the routing profile, dealt
uniformly across devices, and kept across starts.

Applicability: **applicable**. The DeepSeek model is larger than the Strata model, so the tier matters more here.

| Model | Experts | Size per expert | All routed experts |
|---|---|---|---|
| DeepSeek-V4.1-Flash (`src/dsv41`) | 15,360 (40 layers) | 18.8 MB | about 289 GB |
| Qwen3.8-Flash-Next (`src/core`, the main plan) | 24,576 (48 layers × 512) | about 1.4 MB | smaller |

Source: `strata_control/dsv41_fit.py:34-37` (`LAYERS`, `EXPERTS`, `EXPERT_GB`).

## 2. What the engine already has

These facts are read from the code. They set the design.

- **The RAM store is a demand cache of whole experts.** `ExpertStore` (`src/dsv41/ds4_store.hpp`) keeps experts in
  fixed slots that a replacement policy owns (`kLru` or `kLfu`, `ds4_store.hpp:38-60`). It fills on demand
  (`ensure`, `begin`, `wait_slots`) and ahead of time (`prefetch`, `preload`). It is not a static placement.
- **Each expert is three matrices in three GGUF shards.** `LayerGeo` (`ds4_store.hpp:40-47`) gives, for each matrix
  `r`, the file index `file[r]`, the offset `off[r] + e * stride[r]`, its size `bytes[r]`, and its place in the slot
  `tr[r]`. The slot holds the blob with 8192 bytes of gaps (`kSlack`), so aligned reads can spill into the gaps.
- **Reads are unbuffered, into aligned memory.** The store reads through `dfds_` (direct) and `pfds_` (page cache)
  handles (`ds4_store.hpp:485-500`). `ds4_os.hpp` provides `open_direct`, `pread_direct`, `open_unbuffered`,
  `map_ro`, `prefetch`, `resident` and `range_reserve`.
- **The IO queue already has a memory-copy job.** `IoQueue::Job` (`ds4_pool.hpp:139-151`) has a `mem` field: when it
  is set, the job copies `len` bytes from `mem` to `dst` instead of reading a file (`ds4_pool.hpp:217-218`). A tier
  hit can use this job unchanged.
- **The GPU reads store slots in place.** `map_chunk` maps each store chunk for the GPU (`ds4_store.hpp:60-66`), and
  `dev_blob(slot)` gives the device address. When the chunk is mapped, the device kernels read the slot over PCIe.
- **The store shrinks under RAM pressure.** `store_watch()` (`src/dsv41/ds4_model.cu:7240-7250`) gives chunks back when
  free RAM falls below 4 GB (`shrink_chunks`). The store's size is not fixed for the run.
- **The profile is written back on every run.** `save_profile()` (`ds4_main.cpp:10642`, `ds4_model.cu`) rewrites
  `expert-profile.bin` unless `--profile-readonly` is given. The shipped profiles are `data/dsv41/profile-2.00.bin`
  and `profile-2.25.bin`, each with a `.sha256` (`setup.py:794`).
- **The store's preload fills the RAM store at start** from the profile's hottest experts (`ds4_model.cu:5525-5582`).
  Its reads run behind the first token.
- **The prompt path writes into the store too.** VRAM evictions are copied into store slots through `put_begin` and
  `put_done` (`ds4_model.cu:7116-7124`).
- **Build and platforms.** The engine is built from `CMakeLists.txt` and its `strata-dsv41` target. Its platform layer
  is `ds4_os.hpp` (Linux and Windows) and `ds4_os_apple.h` (macOS, Metal). The launch dispatch is in `ds4_launch.h`
  (CUDA, HIP, Metal).
- **Flags.** `--ssd-cache GB` sets the store's budget (`ds4_main.cpp:1325`), `--ssd-policy` picks `lru`, `lfu` or
  `lfulru`, `--vram-experts F` sets the VRAM share (`:1321`), `--threads` sets the store's IO threads (`IoQueue(c.threads)`,
  default 8), `--expert-profile` and `--profile-readonly` (`:1323-1324`), and `--mmap-experts` (`:1348`).

## 3. What differs from the Strata engine

| Topic | Strata engine (`src/core`) | DeepSeek engine (`src/dsv41`) | Effect on the tier |
|---|---|---|---|
| RAM band | Static copy, budget fixed at start | Demand cache with a replacement policy; can shrink | The tier must cover ranks the store might lose, not only the ranks it holds at start (§5.2). |
| Expert storage | One `experts.bin` per layer (blob) | Three matrices in three GGUF shards | The tier stores each expert in the store's slot layout (§6.1). |
| Blob size | Per layer | One slot size for all (the largest blob) | Capacity is counted in whole slots, not bytes (§6.2). |
| Profile | Read once | Written back every run | The profile is not part of the tier's identity (§7.2). |
| GPU read | Tier not GPU-mapped in v1 | Store chunks are GPU-mapped | The tier copies into store slots, which the GPU already reads (§8.3). |
| Pack identity | `experts.bin` hash | Set of GGUF shard files, hashes from setup | Identity covers all shards (§7.1). |
| Platforms | CUDA, HIP, Windows, SYCL, Metal (partial) | CUDA, HIP, Linux, Windows; Apple separate | The tier targets CUDA and HIP first. macOS is out of scope (§12). |
| Prompt path | Stager, `read_direct` batches | `ensure` and `put_begin` into the store | Tier copies go through the store's own IO queue (§8.2). |

## 4. Decisions

Inherited from the Strata plan, unchanged:

- **D1 Host-visible.** The tier is mapped into the host address space.
- **D2 Static placement, persistent.** Placed once, then reused across starts while the identity checks pass.
- **D3 Uniform deal.** Experts are dealt in rank order, round-robin across devices. Larger devices keep taking experts
  after smaller ones are full.
- **D4 Name.** "expert tier": `ExpertTier`, `ExpertTierSet`, `--expert-tier`, `plan_expert_tiers`.
- **D5 Both paths.** Decode and prompt both read the tier.
- **D6 Persistence.** The placement is kept on the devices.
- **D7 Integrity.** Fill-state flag written last, identity checks on start, chunk checksums, `verify=full` opt-in.
- **D8 No hardware yet.** Correctness is developed on synthetic data and on `/tmp`. No speed claim before hardware.

Decided for this engine:

- **E1 Share the planner and the header format.** `plan_expert_tiers` and the header structs in
  `include/strata/core/expert_tier_plan.hpp` and `expert_tier_layout.hpp` are pure code with no engine dependency. The
  DeepSeek engine includes them, as it already includes `strata/dsv41/exl3_cpu.hpp`. Slot sizes come from the caller,
  so the planner does not change.
- **E2 Tier slots match store slots.** A tier slot has the store's `slot_bytes_` and its layout (`tr[]`, `kSlack`). The
  tier fill and the store's reads then use the same job builder, and a tier hit copies bytes that are already in the
  store's layout.
- **E3 One IO queue per device.** Each tier device gets its own `IoQueue` (the class takes a thread count). This keeps
  each narrow link busy with its own queue, and keeps a slow device from blocking the others.
- **E4 Opt-in, default off.** `--expert-tier` absent: byte-identical behaviour, as in the Strata plan. The store's
  feature switches use the existing `store_feature_default(..., defaults_026())` pattern, so the default stays as it is.

## 5. Bands and the guard rule

### 5.1 The bands

Order, hottest first, using the profile's ranking:

1. **VRAM experts.** The GPU keeps the `--vram-experts` share (`ds4_model.cu:4600-4640` for the budget). Which experts
   those are follows the profile order (`ds4_model.cu:4688`).
2. **RAM store.** The store takes the next experts the profile ranks, up to its slot count (preload `:5525`, and the
   demand cache after that).
3. **Expert tiers.** The planner deals the ranks below both, across the devices (§6.2).
4. **File reads.** Everything else, through `dfds_` and `pfds_`.

### 5.2 The guard rule

The store and VRAM sizes can change at start and during the run:

- The store's size is set from free RAM at start (`ram_avail`, `ds4_model.cu:201-256`) and can shrink later
  (`store_watch`, `:7240`).
- The VRAM share can be `auto` (`ds4_model.cu:4619`).

If the tier were planned against the sizes at fill time, a later shrink would leave a gap: ranks that neither the store
nor the tier holds, which are read from the file. The gap is correct, but slower.

**Rule.** The tier is planned against the *guaranteed minimums*, not the sizes at fill time:

- `vram_guard` = the VRAM expert count at the configured `--vram-experts` value, or 0 when it is `auto`.
- `store_guard` = the store size the run is guaranteed to keep. With `--ssd-cache GB` given, that is the configured
  value. With `auto`, the code I read sets no minimum: `shrink_chunks` (`ds4_store.hpp`) can give back chunks down to
  zero. So `auto` needs a chosen minimum, which is open item 1 (§19). Until it is chosen, `auto` is not allowed with
  `--expert-tier`, and the engine says so.

The planner's input is `profile order minus the first vram_guard minus the first store_guard`. Then a store that holds
more than its guard duplicates some tier experts, which costs RAM but is correct. A store that shrinks never leaves a
gap, because the tier covers every rank below the guard.

This rule means the tier may hold a few experts that the store also holds. That is the price of never leaving a gap.
It is measured in P3 (§13).

## 6. Placement and storage

### 6.1 The slot layout

Each tier slot is a store slot: `slot_bytes` bytes, with each matrix at `tr[r]` and the store's `kSlack` gaps. The
blob sizes differ per layer in the geometry, but the slot size is the largest blob (`ds4_store.hpp:init`). The tier
uses that same size, so capacity is counted in whole slots.

Device offset of slot `k` = `header_bytes + k * slot_bytes`, with `header_bytes` rounded up to 4096.

### 6.2 The deal

The planner is the one in the Strata plan (§4.2 of that plan), with one change: capacity is in slots, not bytes.
`TierDevice.capacity` = `floor(usable_bytes / slot_bytes)`. The rest of the algorithm is the same: round-robin over the
ranked list, skipping full devices, stopping when no device has room.

Edge cases as in the Strata plan: a device with zero slots is dropped with a log line, an empty list opens no device
for the fill, duplicate paths are an error, and the device order is part of the placement.

### 6.3 Why the tier does not hold what the store holds

Experts in the store's guard are not in the tier's list (§5.2). Experts the store evicts later are read from the tier
when they are in it, and from the file otherwise. The tier is therefore a second-level cache with a fixed content, and
it needs no replacement policy.

## 7. Identity, reuse and the header

### 7.1 Pack identity

The pack is the set of GGUF shards. Setup already has a SHA-256 for each file (`setup.py:806`, `sha256` list). The
identity is:

- If setup recorded the hashes: the SHA-256 of the sorted list of (file name, file SHA-256). No file is re-hashed.
- Otherwise: the list of (file name, size, modification time).

### 7.2 Profile is not part of identity

The Strata plan put the profile in the identity. For this engine that is wrong, for two reasons:

- The profile is rewritten after every run (`save_profile`), so the identity would change on almost every start.
- A placement made from an older profile is slower, not incorrect. All copies are the same bytes.

So the header records the profile's SHA-256 for information only. A changed profile does not force a refill. A refill
happens when the user asks for one (`--expert-tier-refill`), or when the header is missing or invalid.

The Strata plan should make the same change: profile recorded, not part of identity (§17 of this plan lists it).

### 7.3 Header fields

The header follows `EXPERT_TIER_PLAN.md` §5.2, with these fields added or changed:

| Field | Meaning |
|---|---|
| `pack_id` | §7.1. |
| `profile_sha` | Information only (§7.2). |
| `slot_bytes` | The store's slot size at fill time. Must equal the start's slot size. |
| `layout_version` | Version of the store's slot layout (`tr[]`, `kSlack`). A change refills. |
| `n_layers`, `n_expert` | Model shape. |
| `vram_guard`, `store_guard` | The guard values used to plan (§5.2). |
| `slots_per_device` | Each device's slot count. |
| `placement` | Per expert: device index and slot index. 15,360 experts × 9 bytes ≈ 138 KB. |
| `fill_state`, `header_crc`, `chunk_crc` | As in the Strata plan. |

### 7.4 Reuse on start

A placement is reused when all of these hold:

1. Every device's header is valid (§7.5 of the Strata plan: header A, then B).
2. `pack_id`, `slot_bytes`, `layout_version`, shape and device count match.
3. The start's guards equal the planned guards (`vram_guard`, `store_guard`, §5.2). A guard is a configured value or a
   fixed minimum, so it normally matches. A start whose guard differs refills, because the planned ranks would be wrong.
   A larger actual VRAM or store at start is fine: the extra experts duplicate tier copies, which is correct.
4. Every device has `fill_state == 2`.

If any check fails, the whole tier refills (as in the Strata plan §6.3).

## 8. Serving reads

### 8.1 Lookup

The store is checked first, as today (`ds4_store.hpp:find`). On a miss, the tier is checked. A tier hit is served by
a copy job into a store slot. A miss in both is a file read, as today.

### 8.2 The copy job

On a tier hit for `(layer, e)` with device `d` and tier slot `k`, the store builds three memory-copy jobs, one per
matrix:

```
job[r].mem  = tier_base[d] + header_bytes + k * slot_bytes + g.tr[r]
job[r].dst  = sp.base + g.tr[r]            # the store slot's matrix r
job[r].len  = g.bytes[r]
job[r].keep = nullptr                      # a copy, not a read
job[r].left = &sp.left;  job[r].bad = &sp.bad
```

These go to `tier_io[d]`, the device's own `IoQueue`, not to the store's queue. The existing `loop()` copies them
(`ds4_pool.hpp:217-218`). The store's `try_ready` and `finish` need no change, because the job counts down `sp.left`
like a read does.

Statistics: a tier copy adds to `tier_hits` and `tier_bytes` (new counters next to `hits`, `misses` and `bytes_read` in
`Stats`). It does not add to `bytes_read` (file bytes), so the file-read figure stays meaningful.

### 8.3 GPU

The copies land in store slots. Store chunks are GPU-mapped as they are today, so the device kernels read tier-served
experts exactly as they read file-served ones. The tier itself is not GPU-mapped in v1. Mapping it for direct device
reads (through `map_chunk`, as the store does) is a later option (§16, open item 2).

### 8.4 Priority

The store puts demand reads ahead of guesses (`io_->push(jobs, 3, !demand)`). The tier's queues follow the same rule:
a demand copy is pushed at the front, a guess at the back, and a guess that is cancelled (`sp.cancel`) is dropped. The
`cancel` pointer is set only for guesses, as in the store (`ds4_store.hpp:485-500`).

### 8.5 Preload and the prompt path

- `preload` (`ds4_model.cu:5578`) reads the profile's hottest experts into the store at start. With a tier, its
  experts in the tier are copied from the tier. The file read is used only for the rest.
- The prompt path's writes into the store (`put_begin`, `put_done`, `ds4_model.cu:7116`) are not affected. They copy
  from VRAM, not from the tier.
- The prompt path's reads of store misses go through `ensure` and `begin`, so the tier serves them as it serves decode.
  This is to be confirmed in P2 by reading the prompt path's call sites.

### 8.6 Bounded parallelism

Each tier device has `threads` workers on its queue (default 2, §13). Parallel reads across devices come from the
separate queues. A device's queue never holds more than its own workers, so a narrow link is not oversubscribed.

## 9. Fill

The fill places every tier-held expert once, at start, before the first decode token. It runs behind the preload.

- The fill reads each expert with the store's own read job (§2, `ds4_store.hpp:485-500`), with its destination inside
  the tier mapping instead of a store slot. Tier slots are aligned like store slots, and they have `kSlack` after each
  blob, so the job builder works unchanged. The `alignment` rule of `os::kDirectAlign` is met by construction.
- Each device has its own queue (E3), so the fill runs on all devices at once.
- When every expert of a device is written, the device flushes (`msync`) and then writes its header (fill state 2,
  in header A and then B). A torn fill is refilled at the next start (§7.4 of the Strata plan).
- The fill's progress is printed per device, with the experts placed and the time spent.
- The fill is counted in load statistics, not in decode statistics.

## 10. Memory accounting

- `ram_avail()` (`ds4_os.hpp:613`) and the store's budget do not count tier memory. Tier memory is a device, not RAM.
- The store's budget is not reduced by the tier. The store takes what it did before.
- A tier file on `/tmp` (tmpfs) is RAM, as in the Strata plan (§10 there). Test sizes must respect that.

## 11. Setup and launch

- **Flags.** `--expert-tier PATH,mib=N[,reserve=M][,verify=header|full]`, repeatable, parsed in `ds4_main.cpp` beside
  `--ssd-cache` (`:1325`). The help text sits next to it. The same flag is passed by `strata launch --family deepseek41`
  through `args`, as the Strata plan describes (§5.1 there).
- **Fit.** `strata_control/dsv41_fit.py` decides the VRAM and RAM split from the card and RAM sizes. The tier adds a term:
  the experts that the store and VRAM do not cover, against the tier's total slots. Its output names the tier size the
  model needs, and whether the configured devices cover it. Setup never adds devices by itself.
- **Setup wiring.** The `deepseek41` family is in `setup.py:798-810`. Tier options are carried through the same
  path as `--ssd-cache`.
- **Tests.** `tools/test_setup_dsv41.py` gets the same cases as the Strata plan §12.7: default config unchanged, one flag
  per device in order, and the fit with and without tiers.
- **Pack info.** `strata-dsv41 --pack-info` (`ds4_packinfo.cpp`) reports tier sizes already. It gets a line for the
  configured tiers: the slots each device holds, and the placement that would be written.

## 12. Platforms

- **CUDA and HIP (Linux, Windows).** In scope. Copies to store slots are plain `memcpy` on the host, so the tier does
  not depend on the backend. Only the store's GPU mapping is backend-specific, and the tier does not use it in v1.
- **macOS and Metal.** Out of scope. The unified-memory path (`ds4_os_apple.h`, `arena_budget.h`) has no PCIe tier.
  The engine stops with a clear message when `--expert-tier` is given on macOS.
- **Windows.** `ds4_os.hpp` has the Windows variants (`wide`, `MapViewOfFile`-style mapping). The device open and the
  mapping need the same two paths as the Strata plan §5.3. Test on Windows before any claim.

Every touched backend builds before review (AGENTS.md).

## 13. Measurement

Nothing here is claimed without a measurement on hardware with the model. No hardware is connected, so the phases
below stop before any speed claim.

Measure on the target hardware, with the same-day baseline (`--expert-tier` absent):

- Random-read bandwidth of each device at store-slot size (18.8 MB per expert), at queue depths 1, 2 and 4.
- Bandwidth with 2, 3 and 4 devices at once, each with 1 and 2 workers.
- Decode tokens per second, prompt tokens per second, and file bytes read per token, with and without the tier.
- Fill time for the whole tier and the bytes written.
- Store shrink events (`store_watch`) and their effect on the tier's hit rate.

The existing reference numbers for the engine are in [DEEPSEEK41.md](DEEPSEEK41.md) (for example, decode 40.2 tokens/s on
an RTX 4090 with 48 GB and 128 GB of RAM, `--ssd-cache 108`, as the owner's reported run). They are a starting point
for the same-day baseline, not a comparison.

## 14. Development without hardware

What can be built and tested now:

- **The store with a synthetic geometry.** `ExpertStore` takes a `LayerGeo` list and file handles. A test can give it
  a small synthetic geometry (a few layers, a few experts, small matrices) and synthetic files on `/tmp`. The store's
  logic (hits, misses, eviction, guards, shrink, tier copy jobs, cancellation) runs without the model and without a
  GPU, since `map_chunk` is optional in `Cfg`.
- **The tier.** Open, map, fill, reuse, torn fill and checksum tests use synthetic slots. A file on `/tmp` is accepted
  (tmpfs is RAM, so RAM numbers in such tests are not valid, §10).
- **The planner and the header.** Shared with the Strata plan. Its tests run without hardware.

What cannot be tested without the model and a GPU:

- Byte-identical tokens with and without the tier, on the real model (both decode and prompt).
- Pack identity over the real shard set.
- Any speed claim.

These are the release gates. They wait for hardware and the model files.

## 15. Phases

| Phase | Work | Gate |
|---|---|---|
| D1 | Shared planner and header, used by this engine (E1) | Strata plan P1 and P2 tests pass; the DeepSeek build includes them. |
| D2 | Store change: tier copy jobs, tier stats, per-device queues (§8) | Synthetic-geometry tests: a tier hit gives the same bytes as a file read. |
| D3 | Tier storage: open, map, slot layout, fill through the store's job builder, header, reuse (§6-§9) | Fill and reuse match §7.4; torn fill refills. |
| D4 | Guard rule and shrink tests (§5.2) | A shrink never leaves a gap in the tier's ranks. |
| D5 | Flags and launch (§11) | `strata launch --family deepseek41 --expert-tier ...` reaches the engine. |
| D6 | Setup fit and pack-info lines (§11) | `tools/test_setup_dsv41.py` passes; default config unchanged. |
| D7 | Full-model identity on hardware: tokens identical with and without the tier, decode and prompt | Release gate. Needs the model and a GPU. |
| D8 | Speed measurements (§13) and docs | Numbers with hardware and baseline. |

D1 to D6 need no model and no GPU. D7 and D8 need both.

## 16. Acceptance criteria

1. With no `--expert-tier`, the engine's output, logs and config are byte-identical to the release.
2. With tiers, tokens are identical to the run without tiers, on decode and on the prompt path, for the same prompt and
   seed.
3. Every expert the planner put in a tier is served from the tier (tier hit counts match the plan).
4. A store shrink does not leave a rank that is neither in the store nor in a tier, unless the rank is below the guard.
5. A restart with an unchanged pack, layout and guard reuses the placement and copies no expert.
6. A profile change does not refill the tier.
7. Any pack, layout or guard change refills; a torn fill refills; a corrupt header falls back to its twin.
8. Each touched backend builds (CUDA, HIP on Linux; Windows where touched).
9. Every performance claim has a measurement on hardware with the model, the hardware named, and the same-day baseline.

## 17. Risks

| Risk | Effect | Response |
|---|---|---|
| Guards are small, so the tier covers little | Little gain. | The fit output (§11) says how much tier the model needs. |
| The store's guard duplicates tier experts | Wasted RAM. | Measured in D7; the guard is the price of no gaps. |
| Narrow links make tier copies slower than file reads for some experts | Slower than today for them. | Measured per device (§13); the cost model uses the measured rate. |
| Tier copies compete with the store's demand reads | Decode waits. | Separate queues per device (E3); demand before guesses (§8.4). |
| Profile order is poor for a prompt | Prompt misses go to file. | The profile is the model's own routing history; measured per prompt type. |
| Shard set changes (a new download) | Identity fails; refill. | Expected; the refill is logged with its cost. |
| `/tmp` results look like hardware results | False speed claims. | §14 rules; no speed claim before D8. |
| The model is too large for the devices the user has | Tier is partial; the rest is read from file. | Correct and measured; setup reports the coverage (§11). |

## 18. Changes to the Strata plan

Two decisions made here should go back into [EXPERT_TIER_PLAN.md](EXPERT_TIER_PLAN.md):

1. **Profile out of identity.** Placement affects speed, not correctness, so a changed profile should not force a refill.
   The header records the profile's hash for information. Refill on request or on an invalid header.
2. **Guard rule.** A tier planned against the sizes at fill time leaves a gap when the RAM copy or VRAM shrinks. Plan
   against the guaranteed minimums instead.

## 19. Open items

1. **Guard values for `auto`.** With `--ssd-cache auto`, no minimum store size exists in the code read so far. A minimum
   has to be chosen. Measure how often and how far `store_watch` shrinks the store on real runs before choosing it. A
   large minimum leaves less RAM for the OS; a small one leaves gaps the tier cannot cover.
2. **Direct GPU reads of the tier.** The tier could be mapped for the GPU through `map_chunk`, which would remove the
   copy into the store. It needs the device mapping to be registered and is not in v1.
3. **Engram tables.** The 202.8 GB n-gram table (`DEEPSEEK41.md`) is also read from the SSD. It is out of scope here.
   It could use the same tier later, as it reads fixed-size rows. This would be a separate plan.
4. **Copy method.** A CPU copy into store slots is the v1 method. A DMA copy from the mapping is an option, to be decided
   by measurement, as in the Strata plan.
5. **Prompt path confirmation.** §8.5 says the prompt path's reads go through `ensure` and `begin`. That is to be
   confirmed by reading its call sites in D2.
