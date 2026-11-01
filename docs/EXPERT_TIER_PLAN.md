# Expert tier: plan

Status: plan, not implemented. No hardware is connected yet, so no performance number exists. Every number marked
**target** is a goal to test, not a result. Development and correctness testing run on a RAM-backed file under `/tmp`
(§10).

## 1. Summary

An **expert tier** is a set of large, persistent, host-visible memory devices that hold the experts RAM does not keep.
Each device is configured on its own. Experts are placed at startup from the routing profile, dealt evenly across all
devices, and the placement is kept on the devices so that a later start can reuse it without refilling.

The tier exists to make very large models run. Its devices have a lot of capacity on a narrow link, so each device is
slow for one read but many devices together give enough read bandwidth. The tier also holds the experts that the
prompt path needs, not only decode.

The tier sits below the RAM copy and above the file tier:

```
VRAM expert cache      hottest      (adaptive, existing)
RAM copy (complement)  next         (existing; budget-limited)
expert tiers           next         (NEW; static after fill; persistent; one or more devices)
file tier              the rest     (existing; OS page cache / direct reads)
```

The tier holds only what RAM did not keep, so it holds colder experts than RAM does.

With no `--expert-tier` flag, the engine, the setup output and the config files are byte-identical to today.

## 2. Decisions

| # | Question | Decision | Why |
|---|---|---|---|
| D1 | How is a device reached? | Host-visible. The device is mapped into the host address space; blob pointers point into the mapping. | Matches `ExpertSource::blob()`, which returns a pointer. No staging copy is needed. |
| D2 | When are experts placed? | Static after the fill. Placed once per model and profile, then reused across starts while the identity checks pass (§6). | Same model as `RemoteExperts`, plus persistence, because the devices hold their data across restarts. |
| D3 | How are experts spread? | Uniform deal across devices, in rank order, round-robin. Capacity is counted in bytes. Larger devices keep taking experts after smaller ones are full. | Every device serves a similar mix of hot and cold experts, so reads run in parallel. Confirmed by the owner. |
| D4 | What is the tier called? | "expert tier" everywhere: `ExpertTier`, `ExpertTierSet`, `--expert-tier`, `plan_expert_tiers`, `expert_tier.hpp`. | The storage type is not part of the name. |
| D5 | Does the tier serve the prompt path? | Yes, both decode and prompt. The tier exists to fit very large models, so every read path must be able to use it. | Confirmed by the owner. §7. |
| D6 | Is the placement kept across starts? | Yes. The devices keep a header with the placement and an identity for the pack and the profile. | The devices are persistent and the fill is slow on a narrow link. §6. |
| D7 | How is a stale or torn fill detected? | A fill-state flag written last, identity checks on every start, and checksums per chunk (`verify=full` checks them all). | A wrong expert is a wrong token with no error. §6.4. |
| D8 | What happens before hardware exists? | Develop on a RAM-backed file under `/tmp`. Bandwidth and speed claims wait for hardware (P0-hw). | Confirmed by the owner. §10. |

## 3. Terms

- **Ranked list.** The profile's ordering of `(layer, expert)` pairs, hottest first. It is read by
  `read_expert_profile` (`src/core/expert_cache.cpp:163`).
- **Band.** A part of the ranked list held by one level of the hierarchy (VRAM, RAM, tier, file).
- **Blob.** One expert's bytes in the pack's layout. Its size is `layer_blob_bytes_[layer]`, which can differ per layer.
- **Device.** One configured tier entry: a host-visible path (a device node, or a file on a host-visible mount).
- **Placement.** For every tier-held expert, the device index and the byte offset on that device.
- **Identity.** The pack, the profile, the layout version and the device list that a placement was made for.
- **Fill.** Copying the tier-held blobs from the file tier onto the devices, and writing the header.

## 4. Placement

### 4.1 Band order

The planner runs at startup, after the RAM copy is ready, in this order:

1. **VRAM.** The GPU cache takes its share of the ranked list, as it does today.
2. **RAM copy.** The RAM copy takes what its budget allows from what is left, as it does today
   (`choose_resident_keep_from`, `src/core/expert_source.cpp:288`).
3. **Expert tiers.** The planner deals the remainder across the devices (§4.2).
4. **File tier.** Whatever no band holds is read from the files, as it is today.

Because step 3 sees only what RAM did not keep, every tier-held expert is colder than every RAM-held expert.

### 4.2 Uniform deal

Input:

- `R`: the ranked list minus VRAM- and RAM-held experts, hottest first.
- `D = [d_1 .. d_n]`: the configured devices, in the order given on the command line. Each has a capacity `C_i` in
  bytes, after its reserve.
- `b(l)`: the blob size for layer `l`.

Algorithm (deterministic):

```
cursor = 0                                   # position in D, round-robin
for e in R (hottest first):
    tried = 0
    while tried < n:
        d = D[cursor]
        cursor = (cursor + 1) mod n
        if d.used + b(e.layer) <= d.capacity:
            d.place(e)                       # record (device, byte offset)
            break
        tried += 1
    if tried == n:                           # no device has room for this blob
        stop                                 # the rest is left to the file tier
```

Properties:

- **Balanced.** Each device gets every n-th expert of the remaining ranked list, so each device holds the same mix of
  ranks until it runs out of room.
- **Byte-correct.** A device is charged for the blob's real size, so a device with larger blobs does not overfill.
- **Full devices drop out.** A device that cannot take the next blob is skipped. The rotation continues with the
  others. A small device fills first and then leaves the rotation, so the larger devices keep taking experts to the
  end. This is the confirmed behaviour (D3).
- **Stop rule.** The planner stops when no device has room for the current blob. A later, smaller blob could still
  fit. The rule is kept strict so that a colder expert never takes a place before a hotter one. The unused bytes at
  the end are at most one blob per device.

Edge cases:

- A device with capacity `0`, or smaller than the smallest blob, is dropped from the rotation with a log line.
- An empty `R` (everything fits in VRAM and RAM) places nothing and opens no devices for the fill.
- Two `--expert-tier` entries with the same path are an error.
- The device order is part of the placement. Reordering the flags is a different placement and forces a refill (§6.3).

### 4.3 Weighting (not in v1)

Equal shares assume equal speed. On a narrow link the devices may differ. A weighted deal by measured bandwidth is a
later change. It needs the measurement of P0-hw (§11). The planner's input already has room for a weight per device,
so adding one later does not change the interface.

### 4.4 Fixed for the run

The tier's contents are fixed after the fill. Two consequences:

- The GPU cache and the RAM copy still adapt during a conversation. An adaptive swap that moves an expert from the tier
  into VRAM or RAM is allowed. The tier copy stays valid, because it is never written after the fill.
- The RAM copy's swaps (`replace_resident`, `src/core/expert_source.cpp:4372`) ignore the tier in v1. This can keep
  copies of an expert in both RAM and the tier. All copies are identical, so it is a waste of RAM, not an error.
  Measure it on hardware before changing it.

## 5. Devices

### 5.1 Configuration

One flag per device. The flag is repeatable:

```
--expert-tier PATH,mib=N[,reserve=M][,verify=header|full]
```

- `PATH`: a host-visible device node, or a file on a host-visible mount (a DAX-backed or tmpfs file is accepted for
  testing; §10).
- `mib=N`: the capacity to use, in MiB. A device smaller than this is an error.
- `reserve=M`: MiB kept unused on the device, default 0.
- `verify=header`: the default. Checks the header and the placement on start (§6.4).
- `verify=full`: also checks every chunk's checksum on start. Reads the whole tier once.

Each entry is independent. There is no shared setting.

The configuration is stored in the config's `args` as repeated flags. The same mechanism keeps other user-set engine
flags across reconfigures (`strata_control/user_args.py`, `setup.py:6571`).

### 5.2 Layout of a device

A device holds a header and then the blobs. The header is written by the fill, so its size depends on the placement.

```
offset 0        header A (64 KiB, fixed fields + offsets to the tables below)
offset 64 KiB   header B (copy of A, written after A is complete; used if A is torn)
...             placement table (all devices' placements, identical on every device)
...             chunk checksum table (one CRC32C per 4 MiB of this device's data)
...             data: this device's blobs, each blob at its recorded offset, aligned to 4 KiB
```

Fixed fields in the header:

| Field | Size | Meaning |
|---|---|---|
| `magic` | 8 B | `STRPTIER` |
| `version` | 4 B | Layout version. Starts at 1. |
| `device_index` | 4 B | This device's position in the device list. |
| `device_count` | 4 B | How many devices the placement spans. |
| `pack_id` | 32 B | Identity of the experts file (§6.2). |
| `profile_id` | 32 B | SHA-256 of the profile file's bytes. |
| `n_layers`, `n_expert` | 8 B each | Model shape. |
| `capacity_bytes` | 8 B | This device's capacity at fill time. |
| `placement_off`, `placement_len`, `placement_crc` | 8 B + 8 B + 4 B | Where the placement table is, and its checksum. |
| `chunk_off`, `chunk_count` | 8 B + 8 B | Where the checksum table is. |
| `fill_state` | 4 B | 0 = empty, 1 = filling, 2 = complete. |
| `header_crc` | 4 B | CRC32C of the header's other fields. |

The placement table holds, for each tier-held expert, the device index and the byte offset. The header is sized from the
placement at fill time, not fixed. At 9 bytes per expert, the Flash-Next model (48 layers × 512 experts) needs about
221 KB for its table, which is well under the 1 MiB that the header area provides at that size.

### 5.3 Opening a device

For each device, in order:

1. Open the path read-write. A fill needs write access. If the open fails, the start stops and the error names the
   path and the errno.
2. Check the size against `mib`. A device smaller than `mib` is an error, not a silent cut.
3. Map it with `mmap(MAP_SHARED)` on POSIX, or `MapViewOfFile` on Windows. Keep the base and the length.
4. Read the header (§6.4) and decide reuse or refill (§6.3).
5. Try `cudaHostRegister` on the mapping, with the portable flag, so copies to the GPU can be DMA. If it fails, keep
   the device and mark it unregistered. Copies then go through the CPU, as pageable memory does today (#1250,
   `src/core/expert_source.cpp:3964`). The log line says which case applies.

### 5.4 Failure rules

- A device that fails to open, map or check stops the start with a clear error. The engine does not start with less
  tier than the user asked for. Setup can offer a fallback; the engine does not guess.
- A device that fails after startup (a read error, or the device disappearing) makes its blobs fall back to the file
  tier. The engine logs once per device and keeps running. The file tier must still be present for this to work,
  so a tier-only mode (§9.3) is refused at start unless the user accepts the risk.

## 6. Persistence

### 6.1 What is kept

Each device keeps the full placement table, not only its own part, so that any device can tell what the whole tier
should hold. The data is written only by the fill.

### 6.2 Identity

A placement is valid only for the pack, the profile, the layout and the device list it was made for.

- **Pack id.** The SHA-256 of the experts file, when setup has already verified it (setup checks the hash at download,
  `setup.py`, `verify_sha256`). If it is not known, the pack id is the file's size and modification time. Hashing a
  whole model at every start is too slow, so the engine does not do it.
- **Profile id.** The SHA-256 of the profile's bytes. The profile is small, so this is cheap.
- **Layout version and shape.** `version`, `n_layers`, `n_expert`, and the per-layer blob sizes (checked through the
  pack's layout, not stored separately).
- **Device list.** The count, the order, and each device's `capacity_bytes`. Changing any of these changes the
  placement, so it forces a refill.

### 6.3 Reuse or refill

On start, after the devices are opened:

1. Every device's header must have a valid `header_crc`, the same `pack_id`, `profile_id`, shape, `device_count` and
   the same placement table, and `device_index` must match its position in the list.
2. Every device must have `fill_state == 2`.
3. If all checks pass, the tiers are reused. No blob is copied. The startup log says "tier reused" and the fill time
   saved is not counted.
4. If any check fails, the whole tier is refilled. v1 has no partial reuse: a single changed device refills all of them.
   This is simpler and avoids a placement that mixes two fills. Partial reuse is a later option.

### 6.4 Torn fills and verification

- **Crash safety.** The fill writes the data and the chunk checksums first, flushes them (`msync`), then sets
  `fill_state = 2` in header A and then header B, and flushes again. A device found at `fill_state == 1` is refilled.
  A header whose `header_crc` does not match is read from its twin (B if A fails). If both fail, the device is refilled.
- **`verify=header` (default).** Checks the header, the placement table's CRC, and the identity. It does not read the
  data. A bit flip in the data goes unnoticed. This is the same risk any cache has, and it is why `verify=full` exists.
- **`verify=full`.** Also reads every chunk and compares its CRC32C with the table. It costs one full read of the tier at
  every start. On a narrow link that is a real cost, so it is opt-in.
- A checksum mismatch found at read time (only with `verify=full`, or on a chunk the engine reads for the first time)
  is an error for that blob. The blob is read from the file tier instead, and the device is marked for refill at the
  next start. The engine logs the chunk.

### 6.5 Endurance and fill cost

- Each refill writes the whole tier once. Persistent devices may have a write limit, so the design avoids refills: the
  fill runs only when the identity changes.
- The fill is slow on a narrow link. The first start after a change is the expensive one. The log records the fill time
  and the bytes written, so the cost is visible.

## 7. Serving reads

### 7.1 Lookup order

Every read path checks the bands in order: VRAM (for the decode kernels), RAM copy, tier, file tier. A blob found in a
tier is returned as a pointer into its device mapping. Tier memory is never recycled, so a pointer stays valid as long
as the tier is open.

### 7.2 Decode

- `blob()` and `blob_stable()` return the tier pointer for a held expert.
- `blob_parts()` returns 0 for a tier-held expert, because the blob is contiguous there.
- The decode copies (`copy_blob`) read from the tier. The copy costs a read over the narrow link, so the decode
  scheduler should count these bytes separately from file reads (§7.4).

### 7.3 Prompt path (D5)

The prompt path reads many experts per chunk. Today those reads go through `copy_blobs` in the prefill stager
(`src/prefill/prefill.cpp:692`), which batches file reads in file order (`read_direct`, `expert_source.cpp:3411`).

Changes for the tier:

- `FileExpertSource::copy_blobs` (or a new wrapper above it) splits a batch into the tier share and the file share.
- The tier share is grouped by device. Each device gets one worker thread. The worker copies its blobs in layer order.
  Parallel workers across devices are the point of the tier: each device's link is used at the same time as the others.
- The file share keeps its existing path, unchanged.
- The two shares run at the same time, so the prompt's wait is the slower of the two.
- The worker count per device is capped. A narrow link saturates at a small queue depth, so more threads do not help.
  The cap is a measured value (P0-hw), with a default of 2 per device for now.

Copy mechanism. Two options, to be chosen by measurement on hardware (P0-hw):

- **CPU copy** from the mapping (`memcpy`). Portable. Uses CPU time, and on a narrow link the CPU may wait.
- **DMA copy** with `cudaMemcpyAsync` from the registered mapping. Uses the copy engine, not the CPU. Needs the
  registration of §5.3 to succeed.

Development uses the CPU copy only (§10).

### 7.4 Accounting

The decode and prompt statistics gain three counters: tier bytes read, tier reads, and tier wait time. The file counters
stay as they are. The planner's cost model (`prefill_plan.hpp`, `ssd_gbps`, `file_tier`) gets a tier term, so the
prompt plan knows which bytes arrive over the tier and which over the file.

### 7.5 Prefetch

- Tier-held experts need no prefetch: their data is already in the mapping. Prefetch and stage paths skip them, as they
  skip RAM-held experts (`expert_source.cpp:2183`, `:2209`).
- Whether a read from the device needs a warm-up (a `touch` pass) is a question for P0-hw. Not assumed in v1.

### 7.6 Other read methods

- `cached_bytes()` (`expert_source.cpp:1045`): tier-held blobs count as cached, so the planner does not treat them as
  file reads.
- `transient()` (`:3344`): false for tier-held experts. Their pointers are stable.
- `release()`: returns 0 for tier-held experts. Their pages are not in the page cache.
- `has_resident()` (`:4218`): unchanged. It means the RAM copy. A new `has_tier()` answers for the tier.
- `pinned()`: true when the device is registered.
- `device_alias()`: unchanged (null). Only the `PeerExperts` path uses it.
- `note_async_read()`: no-op. Tier memory is never recycled.

## 8. Engine changes

### 8.1 New files

| File | Content |
|---|---|
| `include/strata/core/expert_tier_plan.hpp` | `plan_expert_tiers` and its types. Pure, no I/O. |
| `src/core/expert_tier_plan.cpp` | The planner in §4.2. |
| `include/strata/core/expert_tier.hpp` | `ExpertTier` (one device) and `ExpertTierSet` (all devices). |
| `src/core/expert_tier.cpp` | Open, map, header, identity, reuse or refill, fill, lookup, copy workers. |
| `include/strata/core/expert_tier_layout.hpp` | Header and table structs, with `static_assert`s on their sizes. |

Keeping the planner and the layout in files with no I/O makes them testable without hardware.

### 8.2 Interfaces

```cpp
// expert_tier_plan.hpp
struct TierDevice { uint64_t capacity_bytes; };             // after reserve
struct TierPlacement { int32_t device; uint64_t offset; };   // device -1 = not placed
std::vector<TierPlacement> plan_expert_tiers(
    const std::vector<std::pair<int32_t,int32_t>>& ranked_rest,  // hottest first, minus VRAM and RAM
    const std::vector<uint64_t>& layer_blob_bytes,
    const std::vector<TierDevice>& devices);

// expert_tier.hpp
struct TierSpec { std::string path; uint64_t mib; uint64_t reserve_mib; bool verify_full; };

class ExpertTier {                                            // one device
public:
    bool open(const TierSpec& s, std::string& err);
    void close();
    uint64_t capacity() const;
    const uint8_t* base() const;
    bool registered() const;                                  // cudaHostRegister succeeded
    bool read_header(TierHeader& h, std::string& why) const;  // header A, else B
    bool write_header(const TierHeader& h, std::string& err); // A then B, flushed
};

class ExpertTierSet {
public:
    // Opens every device in order, checks identity, and returns whether the tier is reusable.
    bool open(const std::vector<TierSpec>& specs, const TierIdentity& id, std::string& err, bool& reused);
    // Plans and copies every tier-held blob once, then writes the headers (fill_state 2 last).
    bool fill(ExpertSource& src, const std::vector<std::pair<int32_t,int32_t>>& ranked_rest,
              const TierIdentity& id, std::string& err);
    const uint8_t* blob(int64_t layer, int64_t expert) const;  // nullptr if not held
    bool has(int64_t layer, int64_t expert) const;
    // Prompt path: copies the held blobs of a batch, one worker per device, in layer order.
    bool copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n,
                    uint64_t& bytes_out);
    uint64_t bytes_held() const;
    void close();
};
```

Lookup is one array: `int32_t device_[layer * n_expert + expert]` and a `uint64_t` offset array. For the Flash-Next
model that is 12 bytes per expert, 288 KiB in all, held in RAM. The index is in RAM, not on the device, because it is
read on every access.

### 8.3 `ExpertSource` integration

`FileExpertSource` (`src/core/expert_source.cpp`) gets an optional `ExpertTierSet*`. The order is RAM copy, tier, file.
Methods that change are listed in §7.2 and §7.6. The `copy_blobs` split is §7.3.

### 8.4 Memory accounting

- `host_available_memory()` (`:401`), `clamp_resident_budget()` (`:121`) and the cgroup math (`:357`) do not count tier
  capacity. Tier memory is not in `MemAvailable` (on Linux) and not in the process's commit (on Windows). Each tier
  reports its own size.
- The RAM copy's budget is not reduced by the tier. The tier takes only experts the RAM copy did not keep.
- The fill's reads are counted in load stats, not in decode statistics.
- Dev caveat: a file in `/tmp` (tmpfs) is RAM. It counts against `MemAvailable`, so tests with a large tier in `/tmp`
  distort the RAM budget numbers (§10).

### 8.5 Startup log

Per device: path, capacity, registration result, reused or refilled, experts placed, bytes placed, and fill time (if
filled). Per start: how many experts sit in each band (VRAM, RAM, tiers, file). One line per device, one summary line.

## 9. Setup and large models

### 9.1 What the tier changes for fit

A model fits when every expert has a home: VRAM, RAM copy, tier, or the file tier. Today the low-RAM rules
(`low_ram_needed`, `setup.py:6000`; `low_ram_resident`, `:6025`; `low_ram_fits`, `:6033`) assume the file tier is the
only place for experts that do not fit in VRAM and RAM. With tiers, the room beside the RAM includes the tier capacity.

Change: the functions that compare the rest of the experts with RAM take a `tier_gb` argument (the sum of the configured
tier capacities, minus reserves). Their default is 0, so today's answers do not change.

### 9.2 Recommendation rule

Setup can tell the user how much tier they need for a model to fit with the file tier only for the remainder:

- `needed = arena − GPU share − RAM copy budget`, the experts that the file tier would otherwise carry.
- If the user gives devices, setup reports whether their total capacity covers `needed`, and how much of the file
  tier's read load they take.
- Setup never adds devices by itself. It recommends, the user chooses (the owner's rule that setup recommends and does
  not force).

### 9.3 Tier-only mode

When the tiers hold every expert the GPU and RAM do not, the file tier is read only for refills. In that case the
engine may unmap the file tier after the fill, to free its page cache. v1 does not do this. It is a later option,
because a device failure (§5.4) then has no fallback. If it is added, it is opt-in and the engine logs the risk.

### 9.4 Setup checks

- Each device can be opened, and its size covers `mib`. A failed check is a refusal with the reason.
- An existing header on the device is read and reported (reusable or not), so the user sees whether the next start
  will refill.
- Setup does not scan for devices. The user gives them.

### 9.5 Default

No tier by default. Setup mentions tiers only when the model does not fit without them, and only as an option.

## 10. Development without hardware

No device is connected. Development uses a file in `/tmp`, which is a tmpfs of 31 GB on this machine (RAM-backed).
What that does and does not show:

| Aspect | With `/tmp` | Caveat |
|---|---|---|
| Correctness: mapping, header, identity, fill, reuse, torn fill, planner, copy workers | Valid. | None. |
| Byte-identical output with and without tiers | Valid. | None. |
| Persistence across restarts | Valid (the file survives the process, not a reboot). | Reboot persistence is not tested until hardware. |
| Narrow-link timing | Not valid. tmpfs is at memory speed. | No speed claims. |
| RAM budget | Distorted. A tier file counts as RAM. | Keep tier sizes small in tests that check RAM numbers, or check them with the tier off. |
| `cudaHostRegister` on a tmpfs file | Not representative of a device. | Test registration separately. |

Rules during development:

- Test files go under `/tmp/strata-tier/` and are removed after each run.
- Tests that check RAM numbers run with the tier off, or with a tier smaller than the run's margin.
- A test-only shim (`tests/core/tier_sim.hpp`) may add a delay per blob copy, set by a bandwidth and a latency. It shapes
  the scheduler and the worker counts, so their logic can be checked. It is not in the engine and not in a release
  build. Its output is labelled "simulated" wherever it appears. No speed claim is made from it.
- No environment variable in the engine turns on simulation.

## 11. Measurement

Nothing is claimed from `/tmp`. Two gates:

### P0-sim (now)

The shim (§10) checks that the worker counts and the uniform deal behave as designed. It produces no claim.

### P0-hw (when the hardware is connected, before any speed claim)

Measure each device on its own, and then several together:

- Random-read bandwidth in blob-sized reads (about 1.4 MB at Flash-Next's blob size, from the expert layout).
- Read latency at queue depth 1.
- Bandwidth with 2, 3 and 4 devices at once, with 1, 2 and 4 workers per device.
- CPU copy against DMA copy (§7.3).
- Whether a `touch` pass before a read changes the rate (§7.5).
- Fill time for the whole model, written bytes, and the first-start cost.

Record: the hardware, the driver, the PCIe slot and lane count per device, the host's memory bandwidth on the same day,
and the link width the device negotiated.

The result decides the worker count, the copy method, and whether the uniform deal is worth it on that hardware.

## 12. Tests

### 12.1 Planner (`tests/core/expert_tier_plan_test.cpp`)

| Case | Expectation |
|---|---|
| Empty ranked list | No placements. |
| One device, room for all | Every expert placed, in rank order. |
| Two equal devices | Experts alternate between the devices, hottest first. |
| Two devices, unequal capacity | Alternation until the smaller is full; the rest go to the larger. |
| Layers with different blob sizes | Each device is charged the real size. No device overfills. |
| Device smaller than one blob | Dropped from the rotation. |
| All devices full | Planning stops; the rest is left to the file tier. |
| Same input, run twice | Identical output. |
| Band order | The caller passes only the experts RAM did not keep. |
| Device order reversed | A different placement, as expected. |

### 12.2 Layout (`tests/core/expert_tier_layout_test.cpp`)

- `static_assert`s on the header and table sizes.
- Header A and B written, A corrupted, B read.
- Both headers corrupted: the device is refilled.
- A placement table whose CRC does not match: refill.

### 12.3 Tier (`tests/core/expert_tier_test.cpp`, on `/tmp`)

- Open, map, fill, and byte-compare every held blob against the file tier.
- Reuse: a second open with the same identity reuses and copies nothing.
- Identity changes (profile bytes, device order, capacity): each forces a refill.
- Torn fill (`fill_state == 1`) forces a refill.
- A device too small for `mib` is an error.
- `verify=full` detects a flipped byte in a chunk. The blob is served from the file tier and the chunk is logged.
- Duplicate path is rejected.
- `close()` unmaps, and `has()` is false afterwards.

### 12.4 Copy workers (`tests/core/expert_tier_copy_test.cpp`)

- A batch with tier and file blobs gives the same bytes as the file path alone.
- Worker count 1, 2 and 4 give the same bytes.
- The shim's delay shows the workers overlapping across devices (checked by the timing order, not by a speed number).

### 12.5 Integration (`tests/core/split_lend_complement_test.cpp`, extended)

- The tier and the RAM copy never hold the same expert at fill time.
- The file tier is read only for experts no band holds.

### 12.6 Engine identity (on `/tmp`, small model)

- Same prompt, same seed: identical tokens with the tier and without it, for both the decode and the prompt path.
- Every expert the planner placed in a tier is served from the tier in decode (hit counts).
- Restart with the tier present: reused, no fill, identical tokens.

### 12.7 Setup (`tools/test_setup_lowram.py`, `tools/test_setup_unsloth.py`)

- No tier by default: the generated config matches the golden file.
- One `--expert-tier` per device, in order, in the args.
- The fit calculation with and without tiers (§9.1); defaults unchanged.
- The device check refuses a too-small device with the reason.

### 12.8 Terminal (`tests/terminal/`)

- `--expert-tier` given to `strata launch` reaches the engine.
- A repeated flag keeps its order across a reconfigure (`user_args`).

## 13. Other backends and docs

- **SYCL.** Mirror `expert_source.hpp`, `expert_source.cpp`, `generate.cpp` under `sycl/`. The tier code is backend-neutral.
  Only the registration call is backend-specific. The SYCL build uses its own call, or none, and logs which.
- **HIP.** The registration uses the HIP equivalent. Check it on the target card before claiming it (P0-hw).
- **Metal.** Check for direct `copy_blob` callers. A Metal tier is out of scope for v1.
- **Windows.** `MapViewOfFile` with the same semantics. Test on Windows before any claim there.

AGENTS.md requires every backend a file touches to build before review. Each backend's build is part of the phase that
touches it.

Docs:

- `docs/DETAILS.md`: the flag, the placement order, the header and verification levels, and the log lines.
- `docs/AI_REFERENCE.md`: the memory tiers, the fit rule, and measured numbers with their hardware.
- `docs/MODELS.md`: only if a setup choice changes.
- A user page, written for non-technical users first: what a tier is, when it makes a large model run, how long the
  first start takes, and what a refill is.

Every number names its hardware, its driver, and its date.

## 14. Phases

| Phase | Work | Gate |
|---|---|---|
| P1 | `plan_expert_tiers` and its tests (§12.1) | All cases pass. |
| P2 | Layout: header, tables, CRCs (§12.2) | Round trip and corruption cases pass. |
| P3 | `ExpertTier` and `ExpertTierSet`: open, identity, reuse, fill, verify (§12.3) | Bytes match; reuse and refill decisions match §6.3. |
| P4 | Copy workers and prompt-path split (§7.3, §12.4) | Same bytes as the file path. |
| P5 | `ExpertSource` integration and accounting (§7.2, §7.6, §8.4) | Identical tokens, both paths, with and without tiers (§12.6). |
| P6 | Flags, parser, `user_args`, help text (§5.1) | §12.8 passes. |
| P7 | Setup: device checks, fit rule, args, budget line (§9) | §12.7 passes; default config unchanged. |
| P8 | Mirror for SYCL and HIP; builds (§13) | Each touched backend builds. |
| P9 | Hardware: P0-hw (§11), then speed measurements | Numbers recorded with hardware and baseline. |
| P10 | Docs (§13) | Claims match the measurements. |

P1 to P8 need no hardware. P9 does. No speed claim is made before P9.

## 15. Acceptance criteria

The feature ships only when all of these hold:

1. With no `--expert-tier`, the engine's output, logs and config are byte-identical to the last release.
2. With tiers, the output tokens are identical to the run without tiers, on both the decode and the prompt path, for the
   same prompt and seed.
3. Every expert the planner placed in a tier is served from the tier.
4. No expert is in both the tier and the RAM copy at fill time.
5. The planner matches §4.2 for every §12.1 case.
6. A restart with an unchanged identity reuses the placement and copies no blob.
7. Any identity change refills; a torn fill refills; a corrupt header falls back to its twin.
8. Each touched backend builds.
9. Every performance claim has a measurement on hardware, the hardware named, and the same-day baseline.

## 16. Risks

| Risk | Effect | Response |
|---|---|---|
| The narrow link is slower than the file tier's direct reads | The tier is slower than the file for some reads. | P0-hw measures it per device; the planner's cost model uses the measured rate. |
| Devices share one link or one root port | Parallel workers add nothing. | P0-hw measures 2, 3 and 4 devices together. |
| `cudaHostRegister` fails on the device | Copies go through the CPU and are slower. | Keep the device, mark it unregistered, measure the cost. |
| Persistent data goes stale | Wrong experts, wrong tokens, no error. | Identity checks on start; `verify=full`; refill on any mismatch. |
| Checksums miss a flip (`verify=header`) | A bit flip in the data goes unnoticed. | Default is documented; `verify=full` is the answer for critical use. |
| A refill is slow on the narrow link | The first start after a change is long. | The fill time is logged; the refill happens only when the identity changes. |
| Write endurance | Each refill writes the whole tier. | Refills are rare by design (§6.5). |
| Device fails after the start | Blobs fall back to the file tier, slower. | Log once; the file tier is required (§5.4). |
| Device sizes differ a lot | The larger devices take most experts at the end. | Accepted (D3). Weighting is §4.3. |
| Dev results from `/tmp` look like hardware results | False speed claims. | §10 rules; no speed claim before P9. |
| RAM swaps duplicate tier experts | Wasted RAM. | Measure on hardware; not a correctness issue. |
| Setup and engine disagree on capacity | A tier is refused at start. | Setup and the engine use the same check. |

## 17. Decisions taken and open items

Taken, from the owner's answers:

- Larger devices keep taking experts after smaller ones are full (D3).
- The tier serves the prompt path as well as decode (D5).
- The devices are persistent, so the placement is kept across starts (D6).
- Development uses `/tmp` until hardware is connected (D8).

Defaults I chose, for the owner to confirm or change:

1. `verify=header` is the default; `verify=full` is opt-in. Reading the whole tier at every start is costly on a narrow
   link.
2. Any identity change refills the whole tier. Partial reuse is later work.
3. The worker count per device starts at 2, until P0-hw measures it.
4. The file tier stays mapped even when the tiers hold every expert (§9.3). Unmapping it is later work.
5. The RAM copy's swaps ignore the tier in v1 (§4.4).
6. The copy method (CPU or DMA) is decided by P0-hw, not now.

Open until hardware exists:

- The copy method, the worker count, and whether a `touch` pass helps (P0-hw).
- Whether reboot persistence behaves as the header assumes on the real device (P9).
- The weighting of uneven devices (§4.3), once their bandwidth is measured.
