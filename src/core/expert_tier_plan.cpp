// src/core/expert_tier_plan.cpp - the uniform deal, exactly §4.2 of docs/EXPERT_TIER_PLAN.md.
#include "strata/core/expert_tier_plan.hpp"

#include <algorithm>
#include <functional>

namespace strata::core {

static_assert(sizeof(TierDevice) == 8, "TierDevice is the capacity the header records as capacity_bytes");
static_assert(sizeof(TierPlacement) == 16, "TierPlacement is the (device, offset) pair the placement arrays hold");

static inline uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

std::vector<TierPlacement> plan_expert_tiers(
    const std::vector<std::pair<int32_t, int32_t>>& ranked_rest,
    const std::vector<uint64_t>& layer_blob_bytes,
    const std::vector<TierDevice>& devices) {
    std::vector<TierPlacement> out(ranked_rest.size());   // every expert starts unplaced (device -1)
    if (devices.empty() || ranked_rest.empty()) return out;

    // §4.2 edge case: a device with capacity 0, or smaller than the smallest blob in `ranked_rest`, can never take
    // one; it drops out of the rotation (the engine logs the line, §4.2/§8.5).  The planner keeps it simple and
    // leaves it out; with the whole list's minimum blob this is the same decision the rotation would make.
    uint64_t min_blob = UINT64_MAX;
    for (const auto& pr : ranked_rest) {
        const int32_t l = pr.first;
        if (l >= 0 && (size_t) l < layer_blob_bytes.size()) min_blob = std::min(min_blob, layer_blob_bytes[(size_t) l]);
    }
    std::vector<uint64_t> used(devices.size(), 0);
    std::vector<char> live(devices.size(), 1);
    int live_n = 0;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (devices[i].capacity_bytes == 0 || devices[i].capacity_bytes < min_blob) live[i] = 0;
        else ++live_n;
    }
    if (live_n == 0) return out;

    // The round-robin cursor skips dead devices; the rotation is over the live devices only, in configuration
    // order, so the deal stays the "every n-th expert of the remainder to each device" of §4.2's properties.
    std::vector<size_t> rot;
    for (size_t i = 0; i < devices.size(); ++i)
        if (live[i]) rot.push_back(i);
    size_t cursor = 0;

    for (size_t i = 0; i < ranked_rest.size(); ++i) {
        const int32_t l = ranked_rest[i].first;
        const uint64_t blob = (l >= 0 && (size_t) l < layer_blob_bytes.size()) ? layer_blob_bytes[(size_t) l] : 0;
        if (blob == 0) continue;   // a layer with no blob bytes: nothing to place, not a stop
        int tried = 0;
        bool placed = false;
        while (tried < (int) rot.size()) {
            const size_t di = rot[cursor];
            cursor = (cursor + 1) % rot.size();
            // the blob starts at the next 4 KiB boundary, so the fit is judged from THAT offset: checking `used + blob`
            // first would let a blob end up to 4095 bytes past the capacity (the reserve, or the end of the device)
            const uint64_t at = align_up(used[di], 4096);
            if (at + blob <= devices[di].capacity_bytes) {
                out[i].device = (int32_t) di;
                out[i].offset = at;
                used[di] = at + blob;
                placed = true;
                break;
            }
            ++tried;
        }
        if (!placed) break;   // §4.2 stop rule: no device has room for this blob; the rest falls to the file tier
    }
    return out;
}

std::vector<std::pair<int32_t, int32_t>> tier_ranked_rest(
    const std::vector<std::pair<int32_t, int32_t>>& profile, int64_t n_layers, int64_t n_expert,
    const std::function<bool(int64_t, int64_t)>& held_elsewhere) {
    std::vector<std::pair<int32_t, int32_t>> out;
    if (n_layers <= 0 || n_expert <= 0) return out;
    std::vector<char> seen((size_t) n_layers * (size_t) n_expert, 0);
    // 1. the profile's own order, hottest first: a repeat of an expert (the profile may list one twice) counts once,
    //    an entry outside the shape is ignored, and one the GPU or the RAM copy holds never enters the tier (§15.4)
    for (const auto& pr : profile) {
        if (pr.first < 0 || pr.first >= n_layers || pr.second < 0 || pr.second >= n_expert) continue;
        char& sn = seen[(size_t) pr.first * (size_t) n_expert + (size_t) pr.second];
        if (sn) continue;
        sn = 1;
        if (!held_elsewhere(pr.first, pr.second)) out.push_back(pr);
    }
    // 2. every expert the profile did not rank (all of them when there is no profile), in layer then expert order:
    //    the tier takes these only where it still has room after the ranked ones (§4.2's stop rule)
    for (int64_t l = 0; l < n_layers; ++l) {
        for (int64_t e = 0; e < n_expert; ++e) {
            if (seen[(size_t) l * (size_t) n_expert + (size_t) e]) continue;
            if (!held_elsewhere(l, e)) out.emplace_back((int32_t) l, (int32_t) e);
        }
    }
    return out;
}

}  // namespace strata::core
