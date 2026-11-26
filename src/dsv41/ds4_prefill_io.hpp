// Bounded chunked-prefill reads. Destinations and their reuse events belong to the caller's staging ring.
#pragma once

#include "ds4_pool.hpp"

#include <limits>

namespace strata::ds4 {

struct PrefillRead {
    std::atomic<int> left{0};
    std::atomic<bool> bad{false};
    bool submitted = false;
    bool tier = false;

    bool ready() const {
        return left.load(std::memory_order_acquire) == 0 && !(tier && bad.load(std::memory_order_relaxed));
    }
};

struct PrefillRange {
    uint64_t off, len;       // matrix (or sub-range) in the tier blob, not a model-file offset
    uint8_t* dst;
    uint8_t* keep = nullptr; // optional layer-cache destination
};

class PrefillReads {
public:
    uint64_t tier_bytes = 0;  // payload submitted; includes stripes of any expert subsequently retried
    uint64_t tier_reads = 0, tier_retries = 0;

    // Build and validate all jobs before publishing any of them. The single counter covers every device,
    // including jobs on queues not yet published to when the first workers start.
    template <class Tier>
    bool submit_tier(Tier& tier, int layer, int expert, const PrefillRange* ranges, size_t n,
                     PrefillRead& read, bool low) {
        std::vector<typename Tier::StripeRef> refs;
        if (!tier.stripe_refs(layer, expert, refs)) return false;
        const size_t nd = tier.device_count();
        std::vector<std::vector<IoQueue::Job>> per_dev(nd);
        int count = 0;
        uint64_t bytes = 0;
        for (size_t i = 0; i < n; ++i) {
            const auto& r = ranges[i];
            uint64_t pos = r.off;
            for (const auto& s : refs) {
                if (s.blob_off + s.len <= pos) continue;
                if (pos - r.off == r.len) break;
                if (s.blob_off > pos || s.dev < 0 || (size_t) s.dev >= nd || !s.src || !tier.queue(s.dev) ||
                    count == std::numeric_limits<int>::max()) return false;
                const uint64_t len = std::min(s.blob_off + s.len - pos, r.len - (pos - r.off));
                auto& j = per_dev[(size_t) s.dev].emplace_back();
                j.mem = s.src + (pos - s.blob_off);
                j.dst = r.dst + (pos - r.off);
                j.keep = r.keep ? r.keep + (pos - r.off) : nullptr;
                j.len = len;
                j.tier_copy = j.tier_stripe = true;
                attach(j, read, low);
                pos += len;
                ++count;
            }
            if (pos - r.off != r.len) return false;
            bytes += r.len;
        }
        if (count == 0) return false;
        begin(read, count, true);
        for (size_t d = 0; d < nd; ++d)
            if (!per_dev[d].empty()) tier.queue((int) d)->push(per_dev[d].data(), (int) per_dev[d].size(), low);
        ++tier_reads;
        tier_bytes += bytes;
        return true;
    }

    void submit_file(IoQueue& queue, IoQueue::Job* jobs, int n, PrefillRead& read, bool low = false) {
        begin(read, n, false);
        for (int i = 0; i < n; ++i) attach(jobs[i], read, low);
        queue.push(jobs, n, low);
    }

    template <class SubmitFile>
    bool retry_file(PrefillRead& read, SubmitFile submit) {
        if (!read.tier || read.left.load(std::memory_order_acquire) != 0 ||
            !read.bad.load(std::memory_order_relaxed) || cancelled_.load(std::memory_order_relaxed)) return false;
        // The acquire above follows all of the tier writes. The retry can now overwrite staging and the layer cache.
        ++tier_retries;
        submit();
        return true;
    }

    uint64_t epoch() const {
        std::lock_guard<std::mutex> lk(done_->mu);
        return done_->epoch;
    }
    void wait_change(uint64_t seen, int us) const {
        std::unique_lock<std::mutex> lk(done_->mu);
        done_->cv.wait_for(lk, std::chrono::microseconds(us), [&] { return done_->epoch != seen; });
    }
    void wait(const PrefillRead& read) const {
        std::unique_lock<std::mutex> lk(done_->mu);
        done_->cv.wait(lk, [&] { return read.left.load(std::memory_order_acquire) == 0; });
    }
    void cancel() { cancelled_.store(true, std::memory_order_release); }

private:
    static void begin(PrefillRead& read, int n, bool tier) {
        read.tier = tier;
        read.bad.store(false, std::memory_order_relaxed);
        read.left.store(n, std::memory_order_relaxed);
    }
    void attach(IoQueue::Job& j, PrefillRead& read, bool low) {
        j.left = &read.left;
        j.bad = &read.bad;
        j.cancel = &cancelled_;
        j.expert_done = done_;
        j.demand = !low;
    }
    std::shared_ptr<CompletionHub> done_ = std::make_shared<CompletionHub>();
    std::atomic<bool> cancelled_{false};
};

// The first S staged items from `first` have distinct slots (ordinal modulo S). A later claimant cannot
// use its slot until one of those items has DMA'd. Check every claimant in that window even if an earlier
// one is blocked on a cache dependency or a previous DMA. `first` advances only through a contiguous prefix.
template <class ItemAt, class TrySubmit>
void submit_prefill_window(size_t& first, size_t end, size_t slots, ItemAt at, TrySubmit submit) {
    size_t staged = 0;
    for (size_t i = first; i < end; ++i) {
        auto& it = at(i);
        if (it.stage >= 0 && staged++ == slots) break;
        if (!it.submitted) it.submitted = submit(i, it);
        if (i == first && it.submitted) ++first;
    }
}

// The routing histogram is indexed by virtual slot. Visit the distinct routed experts in stream order,
// regardless of how many rows picked each one. Speculative slots have already been put in the stream.
template <class Expert>
void prefill_routed(const std::vector<int>& host_list, const int* vs, int first, const int* bounds, Expert add) {
    for (int e : host_list) {
        const int v = vs[e];
        if (v >= first && bounds[v + 1] > bounds[v]) add(e, v);
    }
}

}  // namespace strata::ds4
