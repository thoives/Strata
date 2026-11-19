// src/core/dspark_expert_plan.cpp - the DSpark draft expert quota and cache policy (see the header and docs/DSPARK_HYBRID_PLAN.md §3).
#include "strata/core/dspark_expert_plan.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::core {

namespace {

constexpr uint64_t kPage = 4096;

uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

// Scores order as fixed-point keys (steps of 2^-20): a strict order, so equal scores are exactly equal and std::sort is well defined.
int64_t score_key(double s) {
    return (int64_t) std::llround(s * 1048576.0);
}

}  // namespace

DraftQuotaPlan plan_draft_quotas(uint64_t budget, const std::vector<DraftLayerShape>& layers) {
    DraftQuotaPlan out;
    out.layers.resize(layers.size());
    std::vector<char> open(layers.size(), 0);   // not decided yet
    size_t n_open = 0;
    for (size_t i = 0; i < layers.size(); ++i) {
        auto& q = out.layers[i];
        q.n_experts = std::max<int32_t>(0, layers[i].n_experts);
        q.slot_bytes = layers[i].blob_bytes > 0 ? align_up(layers[i].blob_bytes, kPage) : 0;
        if (q.n_experts > 0 && q.slot_bytes > 0) {
            open[i] = 1;
            ++n_open;
        }
    }

    // 1. a layer whose every expert fits in an equal share is fully resident: it takes its own size, and the rest is shared
    //    again.  Settling a layer only ever raises the share of the others, so a share computed at the top of a pass stays valid
    //    for the rest of it; the passes repeat until nothing settles.
    uint64_t left = budget;
    bool settled = true;
    while (settled && n_open > 0) {
        settled = false;
        const uint64_t share = left / n_open;
        for (size_t i = 0; i < layers.size(); ++i) {
            if (!open[i]) continue;
            auto& q = out.layers[i];
            const uint64_t all = (uint64_t) q.n_experts * q.slot_bytes;
            if (all > share) continue;
            q.usable = q.n_experts;
            q.spare = 0;
            q.fully_resident = true;
            left -= all;
            open[i] = 0;
            --n_open;
            settled = true;
        }
    }

    // 2. what is left is shared equally, in whole slots.  A usable slot needs its spare beside it, so fewer than two slots give
    //    nothing.  Every open layer has fewer experts than its share holds (step 1 settled the others), so `slots` < n_experts.
    if (n_open > 0) {
        const uint64_t share = left / n_open;
        for (size_t i = 0; i < layers.size(); ++i) {
            if (!open[i]) continue;
            auto& q = out.layers[i];
            const uint64_t slots = share / q.slot_bytes;
            if (slots >= 2) {
                q.usable = (int32_t) (slots - 1);
                q.spare = 1;
            }
        }
    }

    uint64_t used = 0;
    for (const auto& q : out.layers) used += q.bytes();
    out.leftover_bytes = budget - used;   // used <= budget by construction
    return out;
}

DraftExpertCache::DraftExpertCache(const DraftQuotaPlan& quota, DraftCacheConfig cfg) : quota_(quota), cfg_(cfg) {
    if (!(cfg_.decay_per_round > 0.0 && cfg_.decay_per_round <= 1.0)) throw std::invalid_argument("decay_per_round must be in (0, 1]");
    if (!(cfg_.min_score > 0.0)) throw std::invalid_argument("min_score must be above 0");
    if (!(cfg_.margin >= 1.0)) throw std::invalid_argument("margin must be at least 1");
    layers_.resize(quota_.layers.size());
    for (size_t l = 0; l < layers_.size(); ++l)
        layers_[l].entries.resize((size_t) std::max<int32_t>(0, quota_.layers[l].n_experts));
}

double DraftExpertCache::score_at(const Entry& e, uint64_t round) const {
    if (e.weight == 0.0) return 0.0;
    if (round <= e.stamp) return e.weight;
    return e.weight * std::pow(cfg_.decay_per_round, (double) (round - e.stamp));
}

DraftExpertCache::Entry& DraftExpertCache::at(int32_t layer, int32_t expert) {
    if (layer < 0 || (size_t) layer >= layers_.size()) throw std::out_of_range("draft layer " + std::to_string(layer));
    auto& es = layers_[(size_t) layer].entries;
    if (expert < 0 || (size_t) expert >= es.size()) throw std::out_of_range("draft expert " + std::to_string(expert));
    return es[(size_t) expert];
}

const DraftExpertCache::Entry& DraftExpertCache::at(int32_t layer, int32_t expert) const {
    if (layer < 0 || (size_t) layer >= layers_.size()) throw std::out_of_range("draft layer " + std::to_string(layer));
    const auto& es = layers_[(size_t) layer].entries;
    if (expert < 0 || (size_t) expert >= es.size()) throw std::out_of_range("draft expert " + std::to_string(expert));
    return es[(size_t) expert];
}

void DraftExpertCache::record(int32_t layer, int32_t expert, uint64_t round, bool gpu_hit) {
    Entry& e = at(layer, expert);
    if (round < e.stamp) throw std::logic_error("draft route recorded out of round order");
    e.weight = score_at(e, round) + 1.0;
    e.stamp = round;
    e.last_use = round;
    DraftLayerStats& st = layers_[(size_t) layer].stats;
    if (gpu_hit) ++st.gpu_hits;
    else ++st.cpu_misses;
}

std::vector<DraftSwap> DraftExpertCache::propose(uint64_t round, int32_t max_swaps) {
    struct Cand {
        int64_t s;
        uint64_t last;
        int32_t e;
    };
    std::vector<DraftSwap> out;
    for (size_t li = 0; li < layers_.size() && (int32_t) out.size() < max_swaps; ++li) {
        const DraftLayerQuota& q = quota_.layers[li];
        if (q.fully_resident || q.usable <= 0) continue;
        Layer& L = layers_[li];
        const int32_t phys = q.usable + q.spare;

        std::vector<Cand> cands;    // hottest first: score, then the more recent use, then the lower expert
        std::vector<Cand> victims;  // coldest first: score, then the less recent use, then the lower expert
        for (int32_t e = 0; e < (int32_t) L.entries.size(); ++e) {
            const Entry& en = L.entries[(size_t) e];
            const double s = score_at(en, round);
            if (en.resident) {
                if (!en.evicting) victims.push_back({score_key(s), en.last_use, e});
                continue;
            }
            if (en.in_flight || en.retry_round > round || s < cfg_.min_score) continue;
            cands.push_back({score_key(s), en.last_use, e});
        }
        std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
            if (a.s != b.s) return a.s > b.s;
            if (a.last != b.last) return a.last > b.last;
            return a.e < b.e;
        });
        std::sort(victims.begin(), victims.end(), [](const Cand& a, const Cand& b) {
            if (a.s != b.s) return a.s < b.s;
            if (a.last != b.last) return a.last < b.last;
            return a.e < b.e;
        });

        size_t vi = 0;
        for (const Cand& c : cands) {
            if ((int32_t) out.size() >= max_swaps) break;
            DraftSwap sw{(int32_t) li, c.e, -1};
            // a free usable slot takes the copy with nothing evicted, so the margin does not apply to it
            if (L.adaptive + L.pending >= q.usable) {
                // every usable slot is taken, so the copy needs a spare slot (usable + spare in all) and the coldest victim keeps
                // its slot until the copy completes.  The candidate must beat the victim by the margin.  The hottest candidate
                // failing this means no later one passes.
                const bool room = L.adaptive + L.pending < phys && vi < victims.size();
                if (!room || (double) c.s < cfg_.margin * (double) victims[vi].s) break;
                sw.victim = victims[vi].e;
                L.entries[(size_t) sw.victim].evicting = true;
                ++vi;
            }
            L.entries[(size_t) c.e].in_flight = true;
            ++L.pending;
            out.push_back(sw);
            in_flight_.push_back(sw);
        }
    }
    return out;
}

void DraftExpertCache::complete(const DraftSwap& swap, bool ok, uint64_t round) {
    auto it = std::find(in_flight_.begin(), in_flight_.end(), swap);
    if (it == in_flight_.end()) throw std::logic_error("DraftExpertCache::complete: that swap is not in flight");
    in_flight_.erase(it);

    Layer& L = layers_[(size_t) swap.layer];
    Entry& cand = L.entries[(size_t) swap.expert];
    cand.in_flight = false;
    --L.pending;
    if (swap.victim >= 0) L.entries[(size_t) swap.victim].evicting = false;

    if (ok) {
        cand.resident = true;
        ++L.adaptive;
        if (swap.victim >= 0) {
            L.entries[(size_t) swap.victim].resident = false;
            --L.adaptive;
        }
        ++L.stats.promotions;
        L.stats.promoted_bytes += quota_.layers[(size_t) swap.layer].slot_bytes;
    } else {
        // the old mapping stays (the victim is still resident); the candidate waits out the cooldown before it is proposed again
        cand.retry_round = round + cfg_.retry_cooldown_rounds;
        ++L.stats.failed_copies;
    }
}

void DraftExpertCache::adopt_resident(int32_t layer, int32_t expert) {
    Entry& en = at(layer, expert);
    const DraftLayerQuota& q = quota_.layers[(size_t) layer];
    if (q.fully_resident) throw std::logic_error("adopt_resident: the layer is fully resident");
    if (en.resident) throw std::logic_error("adopt_resident: the expert is already resident");
    Layer& L = layers_[(size_t) layer];
    if (L.adaptive + L.pending >= q.usable) throw std::logic_error("adopt_resident: the layer has no usable slot left");
    en.resident = true;
    ++L.adaptive;
}

bool DraftExpertCache::resident(int32_t layer, int32_t expert) const {
    const Entry& e = at(layer, expert);
    return quota_.layers[(size_t) layer].fully_resident || e.resident;
}

int32_t DraftExpertCache::adaptive_resident(int32_t layer) const {
    if (layer < 0 || (size_t) layer >= layers_.size()) throw std::out_of_range("draft layer " + std::to_string(layer));
    return layers_[(size_t) layer].adaptive;
}

double DraftExpertCache::score(int32_t layer, int32_t expert, uint64_t round) const {
    return score_at(at(layer, expert), round);
}

const DraftLayerStats& DraftExpertCache::stats(int32_t layer) const {
    if (layer < 0 || (size_t) layer >= layers_.size()) throw std::out_of_range("draft layer " + std::to_string(layer));
    return layers_[(size_t) layer].stats;
}

void DraftExpertCache::reset() {
    for (auto& L : layers_) {
        L.entries.assign(L.entries.size(), Entry{});
        L.adaptive = 0;
        L.pending = 0;
        L.stats = DraftLayerStats{};
    }
    in_flight_.clear();
}

}  // namespace strata::core
