// include/strata/core/dspark_expert_plan.hpp - DSpark's hybrid expert placement, the host side (docs/DSPARK_HYBRID_PLAN.md §3).
//
// Pure: no I/O, no device, no CUDA.  The engine owns the memory: the slot buffers, the copies from RAM, and publishing a new
// slot mapping at a window boundary.  This header decides two things only:
//   - plan_draft_quotas: how many expert slots each draft layer may hold under a byte budget;
//   - DraftExpertCache:  which experts take those slots, by decayed routing frequency with last use as the tie-break.
//
// Ages are draft rounds, never wall time, so a pause for a tool call does not age a layer's experts and the locality of a
// turn survives the pause.  Every routed draft expert counts, whether the GPU or the CPU ran it: either one cost the time, and
// a rejected proposal does not undo that.
//
// Slot accounting (§3): a partially cached layer has `usable` slots that can hold residents and `spare` extra slot(s).  A swap
// copies the new expert into a free slot and releases the victim's slot only after the copy completes, so a kernel that may
// still read the victim is never raced.  Residents plus copies in flight never exceed usable + spare, so every copy has a free
// slot to land in; a victim counts as resident until its copy completes.
#pragma once

#include <cstdint>
#include <vector>

namespace strata::core {

/// One draft layer as the quota sees it.
struct DraftLayerShape {
    uint64_t blob_bytes = 0;   // one routed expert's bytes (before the slot alignment)
    int32_t n_experts = 0;     // routed experts in the layer
};

/// The slots one draft layer gets.  `slot_bytes` is blob_bytes rounded up to 4 KiB.
struct DraftLayerQuota {
    uint64_t slot_bytes = 0;
    int32_t n_experts = 0;
    int32_t usable = 0;          // experts that can be resident at once (adaptive); 0 for an empty layer
    int32_t spare = 0;           // 1 for a partially cached layer, else 0
    bool fully_resident = false; // every expert fits: no CPU miss path and no swaps for this layer

    uint64_t bytes() const { return slot_bytes * (uint64_t) (usable + spare); }
};

/// The budget split across the layers.  `leftover_bytes` is what no layer could use and goes back to the verifier.
struct DraftQuotaPlan {
    std::vector<DraftLayerQuota> layers;
    uint64_t leftover_bytes = 0;
};

/// Deterministic quotas (§3).  Each layer gets an equal share of what is left, in whole slots.  A layer whose experts all fit
/// in its share is fully resident: it takes exactly its own size, and the rest is shared again.  A partially cached layer
/// needs a usable slot plus its spare, so a share that holds fewer than two slots gives it nothing.  Layers with no experts
/// or no bytes take nothing.  The sum of bytes() never exceeds `budget_bytes`.
DraftQuotaPlan plan_draft_quotas(uint64_t budget_bytes, const std::vector<DraftLayerShape>& layers);

/// The policy's knobs.  The defaults are starting points only: nothing has measured them yet, and the benchmark gate decides.
struct DraftCacheConfig {
    double decay_per_round = 0.97;   // a route's weight is multiplied by this for each draft round it ages
    double min_score = 2.0;          // a candidate needs at least this decayed count to be admitted (must be > 0)
    double margin = 1.25;            // a victim is replaced only when the candidate scores at least this many times the victim
    uint64_t retry_cooldown_rounds = 16;   // an expert whose copy failed is not proposed again for this many rounds
};

/// One proposed copy: `expert` goes into a free slot of `layer`; when `victim >= 0` the victim's slot is released after the
/// copy completes (`victim == -1`: the layer has a free usable slot and nothing is evicted).
struct DraftSwap {
    int32_t layer = -1;
    int32_t expert = -1;
    int32_t victim = -1;

    bool operator==(const DraftSwap& o) const { return layer == o.layer && expert == o.expert && victim == o.victim; }
};

/// Counters per draft layer, kept apart from the trunk's (§3).
struct DraftLayerStats {
    uint64_t gpu_hits = 0;      // routes that ran on the GPU
    uint64_t cpu_misses = 0;    // routes that ran on the CPU
    uint64_t promotions = 0;    // copies completed and published
    uint64_t promoted_bytes = 0;
    uint64_t failed_copies = 0;
};

/// The adaptive draft expert cache: the histogram of routes and the resident set, per layer.
///
/// Usage per draft round: record() every routed draft expert, then propose() at a completed draft boundary (even when confidence
/// rejected every proposal), start the copies, and call complete() for each when it finishes.  The caller owns the slots; this
/// class tracks only which expert is resident, which is being copied, and which failed.  Not thread-safe.
class DraftExpertCache {
public:
    explicit DraftExpertCache(const DraftQuotaPlan& quota, DraftCacheConfig cfg = {});

    /// One routed draft expert, GPU hit or CPU miss. `round` is the draft round it ran in; for the same expert it must not go
    /// backwards (std::logic_error otherwise).
    void record(int32_t layer, int32_t expert, uint64_t round, bool gpu_hit);

    /// Up to `max_swaps` new copies (the caller turns its byte budget per interval into this count).  Candidates are the
    /// hottest experts not resident and not in flight, hottest first; a free usable slot is taken first, and otherwise the
    /// coldest resident is the victim, only when the candidate beats it by the margin.  Layers are visited in index order.
    std::vector<DraftSwap> propose(uint64_t round, int32_t max_swaps);

    /// A proposed copy finished: `ok` publishes it (the victim, if any, is released); otherwise the old mapping stays and the
    /// candidate waits out the cooldown.  Throws std::logic_error for a swap that is not in flight.
    void complete(const DraftSwap& swap, bool ok, uint64_t round);

    /// An expert the placement already holds in a usable slot (the engine's initial VRAM experts): it becomes an adaptive
    /// resident with no routes behind it yet.  Throws std::logic_error for a fully resident layer, an expert already held,
    /// or no usable slot left.
    void adopt_resident(int32_t layer, int32_t expert);

    /// Whether the expert is held on the GPU: always for a fully resident layer, else resident in the adaptive cache.
    bool resident(int32_t layer, int32_t expert) const;

    /// Adaptive residents of a layer (0 for a fully resident layer, which has no adaptive set).
    int32_t adaptive_resident(int32_t layer) const;

    /// The decayed score of an expert at `round`.
    double score(int32_t layer, int32_t expert, uint64_t round) const;

    const DraftLayerStats& stats(int32_t layer) const;
    const DraftQuotaPlan& quota() const { return quota_; }

    /// Forget everything: residents, scores, in-flight copies and counters.  Called when the pack is unloaded or replaced.
    void reset();

private:
    struct Entry {
        double weight = 0.0;        // the decayed count as of `stamp`
        uint64_t stamp = 0;
        uint64_t last_use = 0;      // the round of the last route (ties in score go to the more recent)
        uint64_t retry_round = 0;   // not proposed before this round (after a failed copy)
        bool resident = false;
        bool in_flight = false;     // a copy of this expert is pending
        bool evicting = false;      // this resident is the victim of a pending copy
    };

    struct Layer {
        std::vector<Entry> entries;
        int32_t adaptive = 0;       // residents
        int32_t pending = 0;        // copies in flight
        DraftLayerStats stats;
    };

    double score_at(const Entry& e, uint64_t round) const;
    Entry& at(int32_t layer, int32_t expert);
    const Entry& at(int32_t layer, int32_t expert) const;

    DraftQuotaPlan quota_;
    DraftCacheConfig cfg_;
    std::vector<Layer> layers_;
    std::vector<DraftSwap> in_flight_;
};

}  // namespace strata::core
