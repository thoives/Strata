// include/strata/core/expert_tier_plan.hpp - expert tier plan v1 §4.2: the uniform deal.
//
// Pure: no I/O, no device, no CUDA.  It turns the ranked list (hottest first) minus the VRAM- and RAM-held experts
// into a placement: for each expert, the device and the byte offset (into that device's data region) where its
// blob is kept.  Keeping it here, and keeping `expert_tier_layout.hpp` free of I/O too, is what makes the tier
// testable without hardware (docs/EXPERT_TIER_PLAN.md §8.1).
//
// The deal (§4.2): round-robin over the devices in the order configured, charging each device the blob's real
// size; a device that cannot take the next blob is skipped and the rotation continues with the others, so the
// larger devices keep taking experts after the smaller ones are full (decision D3).  The planner stops when no
// device has room for the current blob: a colder expert never takes a place before a hotter one, and what is
// left of the ranked list falls to the file tier.
#pragma once

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace strata::core {

/// One configured tier device, with `capacity_bytes` already reduced by its reserve (docs/EXPERT_TIER_PLAN.md §5.1).
/// The planner does not open or look at anything; a capacity of 0 means "dropped from the rotation" (§4.2).
struct TierDevice {
    uint64_t capacity_bytes = 0;
};

/// Where one ranked expert's blob goes. `device == -1`: not placed, the expert stays in the file tier.
struct TierPlacement {
    int32_t device = -1;
    uint64_t offset = 0;    // byte offset into the device's data region; always a multiple of 4 KiB
};

/// Deal `ranked_rest` (layer, expert pairs, hottest first, already minus the VRAM- and RAM-held experts; §4.1
/// band order is the caller's) across `devices` (the §5.1 configuration order, which is part of the placement:
/// reordering the flags forces a refill, §6.3).  `layer_blob_bytes[layer]` is the blob's real size (per-layer
/// native packs differ, §3), so a device is never overfilled by a larger blob.
///
/// Deterministic (§12.1): the same input always gives the same output.  Offsets grow per device from 0; every
/// blob starts at a 4 KiB boundary, so a later refill that skips an expert never shifts another expert's home.
std::vector<TierPlacement> plan_expert_tiers(
    const std::vector<std::pair<int32_t, int32_t>>& ranked_rest,
    const std::vector<uint64_t>& layer_blob_bytes,
    const std::vector<TierDevice>& devices);

/// The planner's input list (§4.1): the profile's experts in rank order, then every expert the profile does not rank
/// (all of them without a profile) in layer, expert order, each once, minus what `held_elsewhere(layer, expert)` says
/// the GPU cache, the remote tiers or the RAM copy already hold.  A profile entry outside the shape is ignored.
std::vector<std::pair<int32_t, int32_t>> tier_ranked_rest(
    const std::vector<std::pair<int32_t, int32_t>>& profile, int64_t n_layers, int64_t n_expert,
    const std::function<bool(int64_t, int64_t)>& held_elsewhere);

}  // namespace strata::core
