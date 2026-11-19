// Strata. src/dsv41/ds4_store.hpp - the RAM tier of the DeepSeek-V4.1 routed experts as a cache of its own: whole experts read
// from the model's files by unbuffered requests (the OS page cache neither used nor disturbed), kept in a fixed arena of slots
// that a replacement policy owns, filled on demand (the CPU lane's picks) and ahead of time (the lookahead's guesses).
//
// A slot holds an expert in the engine's blob layout (gate | up | down at the layer's offsets tr[r], the same as a VRAM slot), so
// the CPU lane's rows read it in place and - when the arena is registered with the GPU - the device kernels read it over PCIe
// (zero-copy, 24 GB/s on a 5070) through the same plan pointers as a VRAM expert.  An expert is three matrices at three file
// offsets (the GGUF's one tensor a role and layer): each is read aligned (4096) into the slot and moved into place (the read's
// lead of up to 4095 bytes), the blob layout leaving 8192 bytes between matrices (and around the blob) for the aligned spans.
//
// Memory: slots are allocated in chunks as they are first needed (private memory), never past the budget nor past the RAM this
// process can still get less `floor_bytes` (the owner's session and the system keep theirs).
//
// Policy (the slot to give up, among the unpinned, not GPU-visible ones): lru (oldest touch), lfu (lowest decayed hit count).
// A slot whose expert a window is computing is pinned from ensure() to release(); a slot the GPU may read is `visible` until
// sync() takes it back at a window boundary (the GPU idle), which also publishes the newly ready ones.
#pragma once

#include "ds4_os.hpp"
#include "ds4_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::ds4 {

class ExpertStore {
public:
    struct LayerGeo {
        int n_exp = 0;
        int file[3] = {-1, -1, -1};
        uint64_t off[3] = {0, 0, 0}, stride[3] = {0, 0, 0};   // matrix r of expert e: file[r], off[r] + e stride[r]
        uint64_t bytes[3] = {0, 0, 0};
        uint64_t tr[3] = {0, 0, 0};                            // matrix r's place in the blob
        uint64_t blob = 0;                                     // the blob's bytes (gaps of 8192 between matrices)
    };
    enum Policy { kLru = 0, kLfu = 1 };
    struct Stats {
        std::atomic<int64_t> hits{0}, misses{0}, prefetched{0}, prefetch_used{0}, prefetch_wasted{0}, evictions{0}, fails{0};
        std::atomic<int64_t> bytes_read{0}, wait_ns{0}, no_slot{0}, cancelled{0}, put{0};
        // the expert tier (ds4_tier.hpp, §8.2): copies served from a tier slot into a store slot.  Not counted in bytes_read
        // (file bytes), so the file-read figure stays meaningful (§8.2 there).
        std::atomic<int64_t> tier_hits{0}, tier_bytes{0};
        std::atomic<int64_t> tier_direct{0};   // (DS4_TIER_DIRECT: tier hits submitted as one whole-slot direct read; the rest are mapped copies)
    };
    /// DS4_TIER_DIRECT (ds4_tier.hpp): a tier slot's whole read, if its device can take one now.  `fd` is the device's unbuffered
    /// descriptor (kept open until the device's queue has drained); [off, off + len) is the slot in that device; `align` and
    /// `mem_align` are its unbuffered requirements (offset and length; memory); `refused` is set by the queue when the device
    /// refuses unbuffered reads (then no more direct reads from it).
    struct TierDirect {
        os::fd_t fd = os::kBadFd;
        uint64_t off = 0, len = 0, align = 0, mem_align = 0;
        std::atomic<bool>* refused = nullptr;
    };
    struct Vis {                 // a slot the GPU may read from now on (dev != 0), or no longer (dev == 0)
        int layer = 0, e = 0;
        unsigned long long dev = 0;
    };

    struct Cfg {
        int threads = 8;
        size_t budget_bytes = 0;           // the arena at most (grown in chunks as slots are first needed)
        size_t floor_bytes = 8ull << 30;   // RAM left alone
        Policy policy = kLru;
        float decay = 0.97f;               // lfu: heat x decay every `decay_every` accesses of its layer
        int decay_every = 240;
        std::function<uint64_t()> ram_avail;   // the RAM this process can still get (~0: unknown)
        // a new chunk of the arena for the GPU to read in place: returns the device address of `host` (null: not visible)
        std::function<void*(void* host, size_t bytes)> map_chunk;
        std::function<void(void* host)> unmap_chunk;
        size_t reserve = 192;              // slots kept evictable (not visible) for the window's demand and guesses
        int vis_percent = 100;             // the share of experts the GPU may take (by a fixed hash): the rest stay the CPU lane's, so both lanes work
        // the expert tier (ds4_tier.hpp): four hooks, set after init() once the tier decided (null: no tier).
        // find gives (device, slot) of an expert the tier holds and has placed (§8.1: store first, tier on a miss);
        // base is that slot's bytes (the store's slot layout: E2), queue is that device's own IoQueue (E3: a tier copy
        // runs on its device's queue, never the store's).  fill_read builds the store's own read job builder for a
        // tier fill (§9): three aligned reads into a tier slot, the same jobs as a store fill, different destination.
        std::function<bool(int layer, int e, int& dev, uint64_t& slot)> tier_find;
        std::function<uint8_t*(int dev, uint64_t slot)> tier_base;
        std::function<IoQueue*(int dev)> tier_queue;
        // DS4_TIER_LOG: a tier copy started (bytes, demand vs guess) and the time a caller actually waited for one.
        // Both are null unless DS4_TIER_LOG is set; the normal path pays no counters.
        std::function<void(int dev, uint64_t bytes, bool demand)> tier_note_start;
        std::function<void(int dev, uint64_t ns)> tier_note_wait;
        // a copy out of a tier device failed (its own read error): the caller marks that device, so the tier refills at
        // the next start and its experts fall back to the files (ds4_tier.hpp note_refill, §9)
        std::function<void(int dev)> tier_fail;
        // run first when the store goes away: every job the store pushed to a tier device's queue holds a pointer into this
        // store's arena and slots (the destination, `left`, `bad`, `fail`), so those queues must finish them before the arena
        // is released.  The tier owns the queues; the store only asks for them to be drained.
        std::function<void()> tier_drain;
        // DS4_TIER_DIRECT (null: every tier hit is a mapped copy, as before): the whole slot of a tier hit as one direct read,
        // when the device can take it now (false: the mapped copies, as before)
        std::function<bool(int dev, uint64_t slot, TierDirect& out)> tier_direct;
        // DS4_TIER_WAKE (opt-in; env read once in init(), "1" on, "0" off; unset keeps this value, false): a caller waiting for a
        // tier-backed slot sleeps on that device's own queue until the slot's reads are done (IoQueue::wait_until_zero), where
        // the copies run, and not on the store's queue, which none of them signals.  Off: the store queue's 2 ms poll, as in the
        // last release.  Not yet measured faster: see docs/EXPERT_TIER_DSV41_PLAN.md §8.2.
        bool tier_wake = false;
    };

    void set_pfds(const std::vector<os::fd_t>& pfds) { pfds_ = pfds; }   // (the fallback handles were reopened)
    void set_tier_direct(std::function<bool(int, uint64_t, TierDirect&)> f) { cfg_.tier_direct = std::move(f); }
    /// DS4_TIER_DIRECT: the arena is not inherited by a fork.  A direct read into private memory that a fork shares copy-on-write
    /// can land in the wrong copy (the kernel's O_DIRECT and fork hazard), so the arena's range is MADV_DONTFORK.  Call it before
    /// the first direct read (the tier does, when a device is direct).  False: the advice was refused (the caller stays mapped).
    bool no_fork_arena() {
#if defined(__linux__)
        return arena_ != nullptr && ::madvise(arena_, arena_bytes_, MADV_DONTFORK) == 0;
#else
        return false;
#endif
    }
    void set_tier(std::function<bool(int, int, int&, uint64_t&)> find, std::function<uint8_t*(int, uint64_t)> base,
                  std::function<IoQueue*(int)> q, std::function<void(int)> fail,
                  std::function<void()> drain = nullptr) {   // (after the tier decided: reuse or fill placed bytes first, §8.1/§8.5)
        cfg_.tier_find = std::move(find);
        cfg_.tier_base = std::move(base);
        cfg_.tier_queue = std::move(q);
        cfg_.tier_fail = std::move(fail);
        cfg_.tier_drain = std::move(drain);
    }
    void set_tier_trace(std::function<void(int, uint64_t, bool)> start, std::function<void(int, uint64_t)> wait) {
        cfg_.tier_note_start = std::move(start);
        cfg_.tier_note_wait = std::move(wait);
    }
    /// §9: a tier fill read: (layer, e)'s three matrices read exactly as a store fill reads them (the same aligned job
    /// builder, E2), landing inside a tier mapping at `slot_base` on that device's own queue (E3).  The caller owns `left`
    /// (set to 3 by the caller before this) and `bad`; bytes_out counts what was asked for, not the store's file bytes
    /// (the fill is a load statistic, §9).  False: the layer/expert is outside the geometry, or the device has no queue.
    bool tier_fill_read(int layer, int e, uint8_t* slot_base, int dev, std::atomic<int>& left, std::atomic<bool>& bad,
                        uint64_t& bytes_out) {
        if (layer < 0 || layer >= (int) geo_.size() || e < 0 || e >= geo_[(size_t) layer].n_exp || !cfg_.tier_queue) return false;
        IoQueue* tq = cfg_.tier_queue(dev);
        if (tq == nullptr) return false;
        const LayerGeo& g = geo_[(size_t) layer];
        IoQueue::Job jobs[3];
        for (int r = 0; r < 3; ++r) {
            const uint64_t off = g.off[r] + (uint64_t) e * g.stride[r];
            const uint64_t lead = off % os::kDirectAlign;
            uint8_t* fin = slot_base + g.tr[r];
            // the same aligned landing as a store fill (dst is fin - lead rounded down to 4096: the tier slot keeps kSlack
            // on both sides like a store slot, E2), then the matrix moved up to fin
            uint8_t* dst = (uint8_t*) ((uintptr_t) (fin - lead) & ~(uintptr_t) (os::kDirectAlign - 1));
            IoQueue::Job& j = jobs[r];
            j.fd = pfds_[(size_t) g.file[r]];
            j.dfd = dfds_[(size_t) g.file[r]];
            j.off = off;
            j.len = g.bytes[r];
            j.dst = dst;
            j.keep = fin;
            j.left = &left;
            j.bad = &bad;
            j.cancel = nullptr;   // (a fill read is never a guess: it is placed once, §9)
        }
        bytes_out += (uint64_t) (g.bytes[0] + g.bytes[1] + g.bytes[2]);
        tq->push(jobs, 3, false);   // (the fill runs in rank order on the device's queue: hottest first, §9)
        return true;
    }
    bool init(const Cfg& c, const std::vector<LayerGeo>& layers, const std::vector<os::fd_t>& dfds, const std::vector<os::fd_t>& pfds) {
        cfg_ = c;
        if (const char* tw = std::getenv("DS4_TIER_WAKE")) cfg_.tier_wake = std::atoi(tw) != 0;
        if (const char* ds = std::getenv("DS4_TIER_DIRECT_SPAN")) direct_span_ = std::atoi(ds) != 0;   // (opt-in: direct_range)
        if (const char* pf = std::getenv("DS4_STORE_PIN_FIRST")) pin_first_ = std::atoi(pf) != 0;   // (0: the order before 1B, for an A/B)
        if (const char* cs = std::getenv("DS4_STORE_CANCEL_SCAN")) cancel_scan_ = std::atoi(cs) != 0;   // (1: the walk on every call, for an A/B)
        geo_ = layers;
        dfds_ = dfds;
        pfds_ = pfds;
        size_t blob = 0;
        for (const LayerGeo& g : geo_) blob = std::max<size_t>(blob, g.blob);
        slot_bytes_ = (blob + 2 * kSlack + os::kDirectAlign - 1) & ~(size_t) (os::kDirectAlign - 1);
        if (blob == 0 || c.budget_bytes < slot_bytes_) return false;
        nmax_ = c.budget_bytes / slot_bytes_;
        // one reserved range for the whole arena (chunks committed as they are needed), 64 KB aligned chunks
        arena_bytes_ = ((nmax_ + kChunk - 1) / kChunk) * kChunk * slot_bytes_;
        arena_ = (uint8_t*) os::range_reserve(arena_bytes_);
        if (arena_ == nullptr) return false;
        slots_.reset(new Slot[nmax_]);
        where_.resize(geo_.size());
        heat_.resize(geo_.size());
        hits_in_layer_.assign(geo_.size(), 0);
        for (size_t l = 0; l < geo_.size(); ++l) {
            where_[l].assign((size_t) geo_[l].n_exp, -1);
            heat_[l].assign((size_t) geo_[l].n_exp, 0.0f);
        }
        io_.reset(new IoQueue(std::max(1, c.threads)));
        // DS4_LANE_WAKE (opt-in; "1" on, "0" off): the store's completion signal, fed by this queue, by each tier device's queue
        // (set when a slot's reads are pushed there) and by put_done.  Off: no hub, and nothing below changes.
        if (const char* lw = std::getenv("DS4_LANE_WAKE"); lw != nullptr && std::atoi(lw) != 0) {
            hub_ = std::make_shared<CompletionHub>();
            io_->set_hub(hub_);
        }
        grow();   // (the first chunk now: whether the GPU can map it is known from the start)
        // DS4_STORE_PREGROW=1 (default on Linux with the GPU mapping, =0 off): the whole arena now.  cuMemHostRegister (the GPU mapping of a chunk) taken while a kernel of
        // the decode is running waited on it for good on Linux (H100, driver 580): registering all chunks before the first
        // kernel avoids it
#if defined(__linux__)
        const bool pregrow = cfg_.map_chunk && !(std::getenv("DS4_STORE_PREGROW") != nullptr && std::atoi(std::getenv("DS4_STORE_PREGROW")) == 0);   // (Linux: on with the GPU mapping)
#else
        const bool pregrow = std::getenv("DS4_STORE_PREGROW") != nullptr && std::atoi(std::getenv("DS4_STORE_PREGROW")) != 0;
#endif
        if (pregrow)
            while (grow()) {
            }
        return true;
    }
    ~ExpertStore() {
        // 1. the tier devices' queues finish every job that lands in this arena (and counts down this store's slots);
        // 2. then the store's own queue; 3. only then the arena goes.  Nothing queued can touch a released slot.
        if (cfg_.tier_drain) cfg_.tier_drain();
        io_.reset();
        for (size_t i = 0; i < chunks_.size(); ++i)
            if (cfg_.unmap_chunk && chunk_dev_[i] != nullptr) cfg_.unmap_chunk(chunks_[i].first);
        os::range_release(arena_, arena_bytes_);
    }

    size_t slot_bytes() const { return slot_bytes_; }
    size_t slots_allocated() const { return nalloc_; }
    size_t slots_max() const { return nmax_; }
    /// The slots the store has committed now: the most it can promise to hold (a chunk is committed only when the RAM floor allows,
    /// so the nominal slots_max() can be more than the store can keep). Slots are never given back, so this is a guarantee.
    size_t committed_slots() const { return nalloc_; }
    const Stats& stats() const { return st_; }
    /// cancel_stale's arena walks so far (a test's count: a call with no guess loading does not walk).  Read it when no store
    /// call is running on another thread.
    uint64_t cancel_walks() const { return cancel_walks_; }
    double busy_ms() const { return io_ ? io_->busy_ms() : 0.0; }
    static constexpr size_t kSlack = 8192;

    /// The slot holding (layer, e) ready to read, or -1.  (Any thread; a slot found stays valid while it is pinned or visible.)
    int find(int layer, int e) const {
        if (layer < 0 || layer >= (int) where_.size()) return -1;   // (a layer the store does not cover: the DSpark drafter's)
        const int s = where_[(size_t) layer][(size_t) e];
        if (s < 0) return -1;
        Slot& sp = slots_[(size_t) s];
        try_ready(sp);
        return sp.state.load(std::memory_order_acquire) == kReady && sp.layer == layer && sp.expert == e ? s : -1;
    }
    const uint8_t* mat(int slot, int r) const { return slots_[(size_t) slot].base + geo_[(size_t) slots_[(size_t) slot].layer].tr[r]; }
    const uint8_t* blob(int slot) const { return slots_[(size_t) slot].base; }
    /// The device address of the slot's blob (0: the arena is not mapped for the GPU)
    unsigned long long dev_blob(int slot) const {
        const size_t c = (size_t) slot / kChunk;
        if (c >= chunk_dev_.size() || chunk_dev_[c] == nullptr) return 0;
        return (unsigned long long) ((uint8_t*) chunk_dev_[c] + (slots_[(size_t) slot].base - (uint8_t*) chunks_[c].first));
    }

    /// Makes the experts es[0..n) of `layer` ready and pinned (reading what is not here), waiting for the reads; the
    /// caller releases them.  False if a read failed or no slot could be had for one (ok[i] says which are fine; the caller
    /// reads the others another way).
    bool ensure(int layer, const int* es, int n, std::vector<char>& ok) {
        std::vector<int> sl;
        begin(layer, es, n, ok, sl);
        return wait_slots(sl, ok, 0, n);
    }
    /// The first half of ensure(): the experts' slots found or their reads started (pinned), no waiting.  sl[i] is the slot or -1.
    void begin(int layer, const int* es, int n, std::vector<char>& ok, std::vector<int>& sl) {
        ok.assign((size_t) n, 0);
        sl.assign((size_t) n, -1);
        if (layer < 0 || layer >= (int) geo_.size()) return;
        std::lock_guard<std::mutex> lk(mu_);
        if (!pin_first_) {   // DS4_STORE_PIN_FIRST=0: each entry in turn, as before 1B (a miss may then evict a hit that comes later)
            for (int i = 0; i < n; ++i) {
                touch_heat(layer, es[i]);
                const int s = where_[(size_t) layer][(size_t) es[i]];
                if (s >= 0) {
                    sl[(size_t) i] = pin_held(s);
                } else {
                    st_.misses.fetch_add(1, std::memory_order_relaxed);
                    sl[(size_t) i] = start_fill(layer, es[i], true);
                }
            }
            return;
        }
        // Pass 1: every entry the store holds (ready, or its reads on the way) is pinned before any miss is filled.  A fill
        // takes the policy's victim, and that can be a later entry of this same request: it would then be read again.
        for (int i = 0; i < n; ++i) {
            touch_heat(layer, es[i]);
            const int s = where_[(size_t) layer][(size_t) es[i]];
            if (s >= 0) sl[(size_t) i] = pin_held(s);
        }
        // Pass 2: the misses, in request order.  A repeat of a miss met earlier in this pass is held as a hit (its read is on
        // the way), so the same expert is never read twice.
        for (int i = 0; i < n; ++i) {
            if (sl[(size_t) i] >= 0) continue;
            const int e = es[i];
            const int s = where_[(size_t) layer][(size_t) e];
            if (s >= 0) {
                sl[(size_t) i] = pin_held(s);
                continue;
            }
            st_.misses.fetch_add(1, std::memory_order_relaxed);
            sl[(size_t) i] = start_fill(layer, e, true);
        }
    }
    /// True when the reads of slot sl (from begin) are all done.
    bool loaded(int s) const { return s >= 0 && slots_[(size_t) s].left.load(std::memory_order_acquire) <= 0; }
    /// DS4_LANE_WAKE on (the store has its completion signal): a lane may sleep in wait_any_loaded().  Off: false.
    bool lane_wake_on() const { return hub_ != nullptr; }
    /// DS4_LANE_WAKE: sleeps until one of the slots sl[0..n) has its last read done, or `timeout_us` passes (the caller looks again).
    /// Returns true when one is done.  Only the slots' own counts end the wait: a completion of any other slot, or of one of several
    /// reads of a slot, wakes it to test and it sleeps again.  The test is made under the hub's mutex, and the sleep is entered
    /// with that mutex held, so a read that finishes just before the sleep is seen (its notify waits for the sleep).  The caller
    /// passes slots it has pinned (begin), so none is given up and refilled meanwhile; a -1 is skipped.  Call it only when
    /// lane_wake_on().
    bool wait_any_loaded(const int* sl, int n, int timeout_us = 2000) {
        if (hub_ == nullptr || n <= 0) return false;
        auto any_done = [&] {
            for (int i = 0; i < n; ++i)
                if (sl[i] >= 0 && slots_[(size_t) sl[i]].left.load(std::memory_order_acquire) <= 0) return true;
            return false;
        };
        std::unique_lock<std::mutex> lk(hub_->mu);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(timeout_us);
        for (;;) {
            if (any_done()) return true;
#ifdef DS4_IOQUEUE_TEST_HOOKS
            if (CompletionHub::WaitHook h = CompletionHub::wait_hook()) h(hub_.get());
#endif
            if (hub_->cv.wait_until(lk, deadline) == std::cv_status::timeout) return any_done();
        }
    }
    /// The second half: waits for the slots sl[a..b) and marks ok[i] for the good ones.  False if any failed or had no slot.
    bool wait_slots(std::vector<int>& sl, std::vector<char>& ok, int a, int b) {
        const auto t0 = std::chrono::steady_clock::now();
        bool all = true;
        for (int i = a; i < b; ++i) {
            const int s = sl[(size_t) i];
            if (s < 0) {
                all = false;
                continue;
            }
            Slot& sp = slots_[(size_t) s];
            // (only a slot this batch must still wait for counts: a store hit is already done (left 0), and its old
            //  src_dev would otherwise look like a tier wait that never happened)
            if (sp.left.load(std::memory_order_acquire) > 0) {
                const int tdev = sp.src_dev;
                const auto tw0 = std::chrono::steady_clock::now();
                wait_left(sp);
                if (tdev >= 0 && cfg_.tier_note_wait)
                    cfg_.tier_note_wait(tdev, (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - tw0).count());
            }
            std::lock_guard<std::mutex> lk(mu_);
            finish(sp);
            if (sp.state.load() == kReady) ok[(size_t) i] = 1;
            else {
                sp.pin--;
                sl[(size_t) i] = -1;
                all = false;
            }
        }
        st_.wait_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(),
                              std::memory_order_relaxed);
        return all;
    }
    void release(int layer, const int* es, int n, const std::vector<char>& ok) {
        if (layer < 0 || layer >= (int) geo_.size()) return;
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < n; ++i)
            if (ok[(size_t) i]) {
                const int s = where_[(size_t) layer][(size_t) es[i]];
                if (s >= 0 && slots_[(size_t) s].pin > 0) slots_[(size_t) s].pin--;
            }
    }
    /// Starts reading (layer, e) if it is neither here nor on its way (a guess: unpinned, may be given up before it is used).
    void prefetch(int layer, int e) {
        if (layer < 0 || layer >= (int) geo_.size() || e < 0 || e >= geo_[(size_t) layer].n_exp) return;
        if (where_[(size_t) layer][(size_t) e] >= 0) return;
        std::lock_guard<std::mutex> lk(mu_);
        if (where_[(size_t) layer][(size_t) e] >= 0) return;
        if (start_fill(layer, e, false) >= 0) st_.prefetched.fetch_add(1, std::memory_order_relaxed);
    }
    /// Reads the experts of `list` (layer, expert; hottest first) into free slots and returns when they are in (a start that
    /// is already warm: the profile's hottest, so the first tokens find them).  The first `list.size()` fit the arena less
    /// `leave` slots; returns how many were read.
    size_t preload(const std::vector<std::pair<int, int>>& list, size_t leave) {
        size_t n = 0;
        std::vector<int> order;   // the prefix's slots, hottest first: each is pinned from here until it is in
        {
            std::lock_guard<std::mutex> lk(mu_);
            const size_t room = nmax_ > leave ? nmax_ - leave : 0;
            const size_t take = std::min(list.size(), room);
            order.reserve(take);
            // Hottest first, each pinned as it is taken, so no fill of this call gives up a hotter entry of it.  When no slot can be
            // had (a chunk refused by the RAM floor and nothing unpinned to give up) the list stops there: the colder entries are the
            // ones dropped.  An entry already held is pinned and kept the same way, and does not count as read.
            for (size_t k = 0; k < take; ++k) {
                const int layer = list[k].first, e = list[k].second;
                if (layer < 0 || layer >= (int) geo_.size() || e < 0 || e >= geo_[(size_t) layer].n_exp) continue;
                int s = where_[(size_t) layer][(size_t) e];
                if (s >= 0) {
                    slots_[(size_t) s].pin++;
                } else {
                    s = start_fill(layer, e, false);
                    if (s < 0) break;
                    Slot& sp = slots_[(size_t) s];
                    sp.prefetch = false;   // (warm, not a guess)
                    --guesses_loading_;    // (start_fill counted it as one: it is not)
                    sp.pin = 1;
                    ++n;
                }
                order.push_back(s);
            }
            // The explicit LRU order: the prefix's hottest gets the newest tick, so the policy gives up the colder ones first.
            const uint64_t t0 = clock_;
            clock_ += order.size();
            for (size_t j = 0; j < order.size(); ++j) slots_[(size_t) order[j]].tick = t0 + (order.size() - j);
        }
        // Each is waited for on its own (outside the lock; pinned, so it stays put), then made ready.  The old wait took every slot
        // of the arena through again at each wake (nalloc_ slots a wake: quadratic in a long warm list).
        for (const int s : order) wait_left(slots_[(size_t) s]);
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (const int s : order) {
                Slot& sp = slots_[(size_t) s];
                if (sp.pin > 0) sp.pin--;   // (unpinned before any finish: a failed slot goes free, and no pin may outlive it)
                if (sp.state.load(std::memory_order_acquire) == kLoading && sp.left.load(std::memory_order_acquire) == 0) {
                    try_ready(sp);
                    if (sp.state.load(std::memory_order_acquire) == kLoading) finish(sp);
                }
            }
        }
        for (;;) {   // until every read is done: still every loading slot of the arena, as before (the ones above are ready, so
                     // this finds nothing busy; it holds for another thread's reads, and for a slot given up and filled again)
            bool busy = false;
            {
                std::lock_guard<std::mutex> lk(mu_);
                for (size_t j = 0; j < nalloc_; ++j) {
                    Slot& sp = slots_[j];
                    if (sp.state.load(std::memory_order_acquire) != kLoading) continue;
                    if (sp.left.load(std::memory_order_acquire) > 0) busy = true;
                    else {
                        try_ready(sp);
                        if (sp.state.load(std::memory_order_acquire) == kLoading) finish(sp);
                    }
                }
            }
            if (!busy) break;
            io_->wait_any(5000);
        }
        return n;
    }
    /// A slot for (layer, e) that the caller fills with the expert's blob itself (an expert the VRAM cache gives back to the CPU lane: its bytes
    /// are in memory already, so no read from the SSD while the lane waits).  Returns the blob's address, or nullptr (it is here already, or
    /// no slot can be had).  The slot is pinned and not findable until put_done().
    uint8_t* put_begin(int layer, int e) {
        if (layer < 0 || layer >= (int) geo_.size() || e < 0 || e >= geo_[(size_t) layer].n_exp) return nullptr;
        std::lock_guard<std::mutex> lk(mu_);
        if (where_[(size_t) layer][(size_t) e] >= 0) return nullptr;
        const int s = take_slot(false);
        if (s < 0) return nullptr;
        Slot& sp = slots_[(size_t) s];
        if (sp.layer >= 0 && sp.expert >= 0) {
            where_[(size_t) sp.layer][(size_t) sp.expert] = -1;
            if (sp.prefetch) st_.prefetch_wasted.fetch_add(1, std::memory_order_relaxed);
            st_.evictions.fetch_add(1, std::memory_order_relaxed);
        }
        sp.layer = layer;
        sp.expert = e;
        sp.pin = 1;
        sp.prefetch = false;
        sp.visible = false;
        sp.tick = ++clock_;
        sp.bad.store(false);
        sp.cancel.store(false);
        sp.fail.store(false);
        sp.src_dev = -1;
        sp.left.store(1, std::memory_order_release);
        sp.state.store(kLoading, std::memory_order_release);
        where_[(size_t) layer][(size_t) e] = s;
        st_.put.fetch_add(1, std::memory_order_relaxed);
        return sp.base;
    }
    void put_done(int layer, int e, bool ok) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            const int s = where_[(size_t) layer][(size_t) e];
            if (s < 0) return;
            Slot& sp = slots_[(size_t) s];
            if (sp.pin > 0) sp.pin--;
            if (ok) {
                sp.left.store(0, std::memory_order_release);
                sp.state.store(kReady, std::memory_order_release);
            } else {
                where_[(size_t) layer][(size_t) e] = -1;
                sp.layer = sp.expert = -1;
                sp.left.store(0, std::memory_order_release);
                sp.state.store(kFree, std::memory_order_release);
                free_hint_ = std::min<size_t>(free_hint_, (size_t) s);
            }
        }
        // No job of the store's queue counted this slot down: a caller waiting for it (wait_left on the store queue) is woken now
        // and not at its next 2 ms look.  With DS4_TIER_WAKE only: the default path is as in the last release.
        if (cfg_.tier_wake && io_) io_->notify_done();
        if (hub_ != nullptr) hub_->notify();   // (DS4_LANE_WAKE: a lane sleeping on this slot is woken the same way)
    }
    /// Guesses still waiting in the queue for a layer the token is past: dropped unread (their slots free once the queue has gone by)
    void cancel_stale(int layer_now) {
        std::lock_guard<std::mutex> lk(mu_);
        // No guess loading (guesses_loading_ bounds them from above): nothing to cancel, and the arena is not walked
        // (DS4_STORE_CANCEL_SCAN=1 walks it every time, as before)
        if (!cancel_scan_ && guesses_loading_ == 0) return;
        ++cancel_walks_;
        size_t loading = 0;
        for (size_t i = 0; i < nalloc_; ++i) {
            Slot& sp = slots_[i];
            const bool guess = sp.prefetch && sp.state.load(std::memory_order_acquire) == kLoading;
            if (guess) ++loading;
            if (guess && sp.pin == 0 && sp.layer >= 0 && sp.layer <= layer_now) sp.cancel.store(true, std::memory_order_release);
        }
        guesses_loading_ = loading;   // exact now: a guess that lands later only lowers the true count, so the bound stays one
    }
    /// An access the host saw for an expert the GPU computed (keeps it warm in the policy's eyes).
    void touch(int layer, int e) {
        if (layer < 0 || layer >= (int) where_.size()) return;
        std::lock_guard<std::mutex> lk(mu_);
        touch_heat(layer, e);
        const int s = where_[(size_t) layer][(size_t) e];
        if (s >= 0) slots_[(size_t) s].tick = ++clock_;
    }
    bool gpu_mapped() const { return !chunk_dev_.empty() && chunk_dev_[0] != nullptr; }
    /// The arena's address range (the device sees its host pointers as they are when the mapping is the identity)
    unsigned long long arena_lo() const { return (unsigned long long) (uintptr_t) arena_; }
    unsigned long long arena_hi() const { return (unsigned long long) (uintptr_t) (arena_ + arena_bytes_); }
    bool identity_mapped() const {
        for (size_t i = 0; i < chunks_.size(); ++i)
            if (chunk_dev_[i] != nullptr && chunk_dev_[i] != chunks_[i].first) return false;
        return true;
    }
    size_t max_blob() const { return slot_bytes_ - 2 * kSlack; }

    /// At a window boundary (nothing reading the slots; the caller applies `ev` to the device's table before the next window):
    /// every ready slot not visible yet becomes visible (`gpu_ok(layer, e)`: false for an expert VRAM holds), and visible
    /// slots of the lowest score are taken back until `cfg.reserve` slots can be given up.  Returns the number of events.
    size_t sync(std::vector<Vis>& ev, const std::function<bool(int, int)>& gpu_ok) {
        std::lock_guard<std::mutex> lk(mu_);
        size_t nvis = 0, nfree = 0;
        for (size_t i = 0; i < nalloc_; ++i) {
            Slot& sp = slots_[i];
            if (sp.state.load(std::memory_order_acquire) == kLoading && sp.left.load(std::memory_order_acquire) == 0) {
                try_ready(sp);
                if (sp.state.load(std::memory_order_acquire) == kLoading) finish(sp);
            }
            const int st = sp.state.load(std::memory_order_acquire);
            if (st == kFree) {
                ++nfree;
                continue;
            }
            if (st != kReady) continue;
            if (sp.visible) {
                ++nvis;
                continue;
            }
            if (sp.pin == 0) ++nfree;
            if (sp.pin == 0 && dev_blob((int) i) != 0 && gpu_ok(sp.layer, sp.expert) && !sp.prefetch && vis_hash(sp.layer, sp.expert) < cfg_.vis_percent) {
                sp.visible = true;
                ++nvis;
                if (sp.pin == 0) --nfree;
                ev.push_back({sp.layer, sp.expert, dev_blob((int) i)});
            }
        }
        // give slots back until the reserve is there: the visible ones with the lowest score
        if (nfree < cfg_.reserve && nvis > 0) {
            std::vector<std::pair<double, size_t>> cand;
            cand.reserve(nvis);
            for (size_t i = 0; i < nalloc_; ++i)
                if (slots_[i].visible && slots_[i].state.load(std::memory_order_acquire) == kReady) cand.push_back({score(slots_[i]), i});
            const size_t want = std::min(cand.size(), cfg_.reserve - nfree);
            std::partial_sort(cand.begin(), cand.begin() + (long) want, cand.end());
            for (size_t k = 0; k < want; ++k) {
                Slot& sp = slots_[cand[k].second];
                sp.visible = false;
                ev.push_back({sp.layer, sp.expert, 0ull});
            }
        }
        return ev.size();
    }
    /// RAM is short: gives back whole chunks from the end of the arena (their experts forgotten) - up to `want`, stopping at one
    /// with a slot in use (pinned, loading, or GPU-visible).  Returns the chunks given back.
    size_t shrink_chunks(size_t want) {
        std::lock_guard<std::mutex> lk(mu_);
        size_t done = 0;
        while (done < want && drop_last_chunk()) ++done;
        return done;
    }
    /// The same by bytes: gives back chunks from the end until at least `want_bytes` are freed - counted from each chunk's own size
    /// (the last one may be short), so a chunk of many slots counts as what it is - or until none can go.  Returns the chunks given
    /// back; `freed`, when set, gets their bytes.
    size_t shrink_bytes(size_t want_bytes, size_t* freed = nullptr) {
        std::lock_guard<std::mutex> lk(mu_);
        size_t done = 0, bytes = 0;
        while (bytes < want_bytes && !chunks_.empty()) {
            const size_t cb = chunks_.back().second;
            if (!drop_last_chunk()) break;
            bytes += cb;
            ++done;
        }
        if (freed != nullptr) *freed = bytes;
        return done;
    }
    size_t shrunk() const { return shrunk_; }
    /// An expert VRAM took over: no longer a visible slot
    void unmark(int layer, int e) {
        if (layer < 0 || layer >= (int) where_.size()) return;
        std::lock_guard<std::mutex> lk(mu_);
        const int s = where_[(size_t) layer][(size_t) e];
        if (s >= 0) slots_[(size_t) s].visible = false;
    }
    size_t visible_count() const {
        size_t n = 0;
        for (size_t i = 0; i < nalloc_; ++i) n += slots_[i].visible ? 1 : 0;
        return n;
    }

private:
    static constexpr int kFree = 0, kLoading = 1, kReady = 2, kFailed = 3;
    // slots a chunk of the arena (the unit the GPU mapping registers; DS4_STORE_CHUNK_SLOTS: a test knob for the registration segment size)
    static size_t chunk_slots() {
        const char* e = std::getenv("DS4_STORE_CHUNK_SLOTS");
        const long v = e ? std::atol(e) : 0;
        return v >= 8 && v <= 4096 ? (size_t) v : 64;
    }
    const size_t kChunk = chunk_slots();
    struct Slot {
        uint8_t* base = nullptr;   // the blob's first byte
        int layer = -1, expert = -1, pin = 0;
        std::atomic<int> src_dev{-1};   // (>= 0: this slot's bytes came from tier device src_dev: a failure marks that device, §9;
                                        //  atomic: wait_left() reads it outside mu_, for a slot preload() does not pin)
        bool prefetch = false, visible = false;
        std::atomic<int> state{kFree}, left{0};
        std::atomic<bool> bad{false}, cancel{false};
        std::atomic<bool> fail{false};   // a job of this slot RAN and failed (bad alone also means "cancelled")
        uint64_t tick = 0;
    };

    static int vis_hash(int layer, int e) { return (int) (((unsigned) layer * 2654435761u + (unsigned) e * 40503u) >> 7) % 100; }
    double score(const Slot& sp) const {
        if (cfg_.policy == kLru) return (double) sp.tick;
        return (double) heat_[(size_t) sp.layer][(size_t) sp.expert] + 1e-9 * (double) sp.tick;
    }
    void touch_heat(int layer, int e) {
        heat_[(size_t) layer][(size_t) e] += 1.0f;
        if (cfg_.policy != kLru && ++hits_in_layer_[(size_t) layer] % (uint64_t) cfg_.decay_every == 0)
            for (float& h : heat_[(size_t) layer]) h *= cfg_.decay;   // (a linear pass over 384 floats every 240 accesses of that layer)
    }
    // DS4_TIER_DIRECT: the checks before a direct read of [a, a + len) of the tier's slot lands in this store slot (mu_ held).  The
    // device's alignments hold for the offset, the length and the memory, the range is inside the slot, and the destination is
    // inside the committed arena.  (a = 0, len = the whole slot: the read as it was.)
    bool direct_fits(const TierDirect& td, const Slot& sp, uint64_t a, uint64_t len) const {
        if (td.fd == os::kBadFd || sp.base == nullptr || td.len != slot_bytes_) return false;
        if (td.align == 0 || td.mem_align == 0 || (td.align & (td.align - 1)) != 0 || (td.mem_align & (td.mem_align - 1)) != 0) return false;
        const uintptr_t dst = (uintptr_t) (sp.base - kSlack), lo = (uintptr_t) arena_, hi = lo + arena_bytes_;
        if (dst < lo || dst > hi || slot_bytes_ > hi - dst) return false;
        if (len == 0 || a > slot_bytes_ || len > slot_bytes_ - a) return false;
        return (td.off + a) % td.align == 0 && len % td.align == 0 && (dst + a) % td.mem_align == 0 && len % td.mem_align == 0;
    }
    // DS4_TIER_DIRECT: the range of the tier's slot that one direct read takes (mu_ held): with DS4_TIER_DIRECT_SPAN on, the aligned span
    // that covers the layer's three matrices (the slot's leading and trailing padding left out; the matrices' gaps stay in); when
    // that span cannot meet the device's alignments or bounds, and always with the switch off, the whole slot.  False: neither fits.
    bool direct_range(const LayerGeo& g, const TierDirect& td, const Slot& sp, uint64_t& a, uint64_t& len) const {
        if (direct_span_ && td.align != 0 && (td.align & (td.align - 1)) == 0) {
            uint64_t lo = UINT64_MAX, hi = 0;
            for (int r = 0; r < 3; ++r) {
                lo = std::min<uint64_t>(lo, kSlack + g.tr[r]);
                hi = std::max<uint64_t>(hi, kSlack + g.tr[r] + g.bytes[r]);
            }
            const uint64_t a0 = lo & ~(td.align - 1), b0 = (hi + td.align - 1) & ~(td.align - 1);
            if (b0 <= slot_bytes_ && b0 > a0 && direct_fits(td, sp, a0, b0 - a0)) {
                a = a0;
                len = b0 - a0;
                return true;
            }
        }
        if (!direct_fits(td, sp, 0, slot_bytes_)) return false;
        a = 0;
        len = slot_bytes_;
        return true;
    }
    // A slot the store holds for a request (ready, or its reads on the way): counted as a hit once ready and not a guess, pinned
    // for the request, and touched.  mu_ held.  Returns s.
    int pin_held(int s) {
        Slot& sp = slots_[(size_t) s];
        try_ready(sp);
        if (sp.state.load() == kReady && !sp.prefetch) st_.hits.fetch_add(1, std::memory_order_relaxed);
        if (sp.prefetch) st_.prefetch_used.fetch_add(1, std::memory_order_relaxed);
        sp.prefetch = false;
        sp.pin++;
        sp.tick = ++clock_;
        return s;
    }
    // a slot for (layer, e) and its reads started; -1 if there is none to be had.  mu_ held.
    int start_fill(int layer, int e, bool demand) {
        int s = take_slot(demand);
        if (s < 0) {
            st_.no_slot.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }
        Slot& sp = slots_[(size_t) s];
        if (sp.layer >= 0 && sp.expert >= 0) {
            where_[(size_t) sp.layer][(size_t) sp.expert] = -1;
            if (sp.prefetch) st_.prefetch_wasted.fetch_add(1, std::memory_order_relaxed);
            st_.evictions.fetch_add(1, std::memory_order_relaxed);
        }
        sp.layer = layer;
        sp.expert = e;
        sp.pin = demand ? 1 : 0;
        sp.prefetch = !demand;
        if (!demand) ++guesses_loading_;
        sp.visible = false;
        sp.tick = ++clock_;
        sp.state.store(kLoading, std::memory_order_release);
        sp.bad.store(false);
        sp.cancel.store(false);
        sp.fail.store(false);
        const LayerGeo& g = geo_[(size_t) layer];
        IoQueue::Job jobs[3];
        // §8.1: a store miss checks the tier next.  A tier hit is served by three memory-copy jobs (ds4_pool.hpp's
        // `mem` form: a copy of len bytes from memory to dst instead of a read) into this store slot, on the tier
        // device's own queue (E3), with the store's priority rule kept (§8.4: a demand copy at the front, a guess at
        // the back, and a guess the layer moved past cancelled as today).  A miss in both bands reads the file, as today.
        int tdev = -1;
        uint64_t tslot = 0;
        uint8_t* tbase = nullptr;
        IoQueue* tq = nullptr;
        if (cfg_.tier_find && cfg_.tier_find(layer, e, tdev, tslot) && cfg_.tier_base && cfg_.tier_queue) {
            tbase = cfg_.tier_base(tdev, tslot);
            tq = cfg_.tier_queue(tdev);
            if (tbase == nullptr || tq == nullptr) tdev = -1;   // (the device closed: fall back to the file read)
        }
        if (tdev >= 0) {
            const uint64_t nb = g.bytes[0] + g.bytes[1] + g.bytes[2];
            st_.tier_hits.fetch_add(1, std::memory_order_relaxed);
            st_.tier_bytes.fetch_add((int64_t) nb, std::memory_order_relaxed);
            if (cfg_.tier_note_start) cfg_.tier_note_start(tdev, nb, demand);
        }
        sp.src_dev = tdev;   // (the slot records its source: finish() marks a failed tier's device)
        // DS4_TIER_DIRECT (§3/§4): a tier hit whose device can take the whole slot now is ONE direct read into this slot, on the
        // device's queue, with the same priority.  Anything the checks refuse is the three mapped copies below, as before.  The
        // count is set before the slot is findable (where_ is published after it).
        TierDirect td;
        uint64_t da = 0, dlen = 0;   // the direct read's range in the slot (direct_range: the whole slot, or the matrices' span)
        const bool direct = tdev >= 0 && cfg_.tier_direct && cfg_.tier_direct(tdev, tslot, td) && direct_range(g, td, sp, da, dlen);
        sp.left.store(direct ? 1 : 3, std::memory_order_release);
        where_[(size_t) layer][(size_t) e] = s;
        if (direct) {
            IoQueue::Job dj;
            dj.dfd = td.fd;
            dj.off = td.off + da;
            dj.len = dlen;
            dj.dst = sp.base - kSlack + da;   // (the store's slot starts kSlack before its blob: the tier's slot has the same layout, E2)
            dj.align = td.align;
            dj.mem_align = td.mem_align;
            dj.payload = g.bytes[0] + g.bytes[1] + g.bytes[2];
            dj.fallback = tbase - kSlack + da;   // (the same range in the tier's mapping: copied if the device refuses the direct read)
            dj.refused = td.refused;
            dj.tier_direct = true;
            dj.left = &sp.left;
            dj.bad = &sp.bad;
            dj.fail = &sp.fail;
            dj.cancel = demand ? nullptr : &sp.cancel;
            dj.tier_copy = true;
            dj.demand = demand;
            st_.tier_direct.fetch_add(1, std::memory_order_relaxed);
            if (hub_ != nullptr) tq->set_hub(hub_);   // (DS4_LANE_WAKE)
            tq->push(&dj, 1, !demand);
            return s;
        }
        for (int r = 0; r < 3; ++r) {
            uint8_t* fin = sp.base + g.tr[r];   // where the matrix belongs
            IoQueue::Job& j = jobs[r];
            j.left = &sp.left;
            j.bad = &sp.bad;
            j.fail = &sp.fail;
            j.cancel = demand ? nullptr : &sp.cancel;
            j.tier_copy = tdev >= 0;
            j.demand = demand;
            if (tdev >= 0) {   // (the tier's bytes are already in the store's layout: a copy, no move-up, keep = null)
                j.mem = tbase + g.tr[r];
                j.dst = fin;
                j.len = g.bytes[r];
                j.off = 0;
            } else {
                const uint64_t off = g.off[r] + (uint64_t) e * g.stride[r];
                const uint64_t lead = off % os::kDirectAlign;
                // the aligned read lands the file's byte `off` at dst + lead: dst is fin - lead rounded down to 4096 (its span
                // spills under 4096 bytes either way into the blob's gaps), the matrix is then moved up to fin
                uint8_t* dst = (uint8_t*) ((uintptr_t) (fin - lead) & ~(uintptr_t) (os::kDirectAlign - 1));
                j.fd = pfds_[(size_t) g.file[r]];
                j.dfd = dfds_[(size_t) g.file[r]];
                j.off = off;
                j.dst = dst;
                j.keep = fin;
                j.len = g.bytes[r];
                st_.bytes_read.fetch_add((int64_t) g.bytes[r], std::memory_order_relaxed);
            }
        }
        if (tdev >= 0) {   // (the tier device's own queue: demand at the front, guess at the back)
            if (hub_ != nullptr) tq->set_hub(hub_);   // (DS4_LANE_WAKE: this device's completions are announced too)
            tq->push(jobs, 3, !demand);
        }
        else io_->push(jobs, 3, !demand);            // (a guess waits behind every demand read)
        return s;
    }
    // Waits until the slot's reads are all done (`left` zero).  Call it without mu_.  wait_slots() has the slot pinned.  preload()
    // does not: once its reads are done another thread may give the slot up and fill it again, and this then waits for that
    // fill (the slots array never moves, and src_dev is atomic); preload() looks at the state again under mu_ afterwards.
    // DS4_TIER_WAKE off (the default): the store queue's 2 ms poll, as in the last release.  On: a tier-backed slot's copies run
    // on its device's queue (§8.2/E3), which the store queue never hears of, so the caller sleeps there until the slot's own
    // count is zero.  The queue is looked up again at every turn (a device that closed in the meantime falls back to the store's
    // queue), and the wait there is bounded: a lost wake-up costs the bound, not a hang.  A slot read from the files and a slot
    // the GPU put in (put_begin) have no tier device and keep the store queue.
    void wait_left(Slot& sp) {
        while (sp.left.load(std::memory_order_acquire) > 0) {
            IoQueue* tq = nullptr;
            if (cfg_.tier_wake) {
                const int dev = sp.src_dev;
                if (dev >= 0 && cfg_.tier_queue) tq = cfg_.tier_queue(dev);
            }
            if (tq != nullptr) tq->wait_until_zero(sp.left);   // (returns when done, or after the 50 ms bound)
            else io_->wait_any(2000);
        }
    }
    // loading -> ready once its reads are done and whole (lock-free; a failed one is cleaned up by finish / take_slot)
    void try_ready(Slot& sp) const {
        int want = kLoading;
        if (sp.left.load(std::memory_order_acquire) == 0 && !sp.bad.load(std::memory_order_acquire))
            sp.state.compare_exchange_strong(want, kReady, std::memory_order_acq_rel);
    }
    void finish(Slot& sp) {   // mu_ held; reads done
        if (sp.state.load() != kLoading) return;
        if (sp.bad.load()) {
            // a copy out of the tier that RAN and failed: its device refills next start (§9).  A guess the layer moved past is
            // dropped unread (bad set, fail not): that is the store's own housekeeping and says nothing about the device.
            if (sp.src_dev >= 0 && sp.fail.load(std::memory_order_acquire) && cfg_.tier_fail) cfg_.tier_fail(sp.src_dev);
            where_[(size_t) sp.layer][(size_t) sp.expert] = -1;
            sp.layer = sp.expert = -1;
            sp.state.store(kFree, std::memory_order_release);
            sp.prefetch = false;
            if (sp.cancel.load()) st_.cancelled.fetch_add(1, std::memory_order_relaxed);
            else st_.fails.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        sp.state.store(kReady, std::memory_order_release);
    }
    bool grow() {
        if (nalloc_ >= nmax_) return false;
        const size_t n = std::min(kChunk, nmax_ - nalloc_);
        const size_t bytes = n * slot_bytes_;
        const uint64_t avail = cfg_.ram_avail ? cfg_.ram_avail() : ~uint64_t{0};
        if (avail != ~uint64_t{0} && avail < bytes + cfg_.floor_bytes) return false;
        void* p = arena_ + nalloc_ * slot_bytes_;
        if (!os::range_commit(p, bytes)) return false;
        chunks_.push_back({p, bytes});
        void* dev = cfg_.map_chunk ? cfg_.map_chunk(p, bytes) : nullptr;
        chunk_dev_.push_back(dev);
        for (size_t i = 0; i < n; ++i) slots_[nalloc_ + i].base = (uint8_t*) p + i * slot_bytes_ + kSlack;
        nalloc_ += n;
        return true;
    }
    // gives back the arena's last chunk (its experts forgotten) if no slot of it is in use.  mu_ held.  False: none, or it is busy.
    bool drop_last_chunk() {
        if (chunks_.empty()) return false;
        const size_t c = chunks_.size() - 1, first = c * kChunk;
        bool busy = false;
        for (size_t i = first; i < nalloc_ && !busy; ++i) {
            Slot& sp = slots_[i];
            if (sp.state.load(std::memory_order_acquire) == kLoading && sp.left.load(std::memory_order_acquire) == 0) {   // (read by now)
                try_ready(sp);
                if (sp.state.load(std::memory_order_acquire) == kLoading) finish(sp);
            }
            busy = sp.pin > 0 || sp.visible || sp.state.load(std::memory_order_acquire) == kLoading;
        }
        if (busy) return false;
        for (size_t i = first; i < nalloc_; ++i) {
            Slot& sp = slots_[i];
            if (sp.layer >= 0 && sp.expert >= 0) {
                where_[(size_t) sp.layer][(size_t) sp.expert] = -1;
                st_.evictions.fetch_add(1, std::memory_order_relaxed);
            }
            sp.layer = sp.expert = -1;
            sp.prefetch = false;
            sp.state.store(kFree, std::memory_order_release);
            sp.base = nullptr;
        }
        if (cfg_.unmap_chunk && chunk_dev_[c] != nullptr) cfg_.unmap_chunk(chunks_[c].first);
        os::range_decommit(chunks_[c].first, chunks_[c].second);
        chunks_.pop_back();
        chunk_dev_.pop_back();
        nalloc_ = first;
        free_hint_ = std::min(free_hint_, first);
        ++shrunk_;
        return true;
    }
    // the slot to give up (or a free one): the policy's lowest among the unpinned, not visible, ready ones.  mu_ held.
    int take_slot(bool demand) {
        for (size_t i = free_hint_; i < nalloc_; ++i)
            if (slots_[i].state.load(std::memory_order_acquire) == kFree) {
                free_hint_ = i + 1;
                return (int) i;
            }
        if (grow()) return take_slot(demand);
        int best = -1;
        double bv = 0;
        for (size_t i = 0; i < nalloc_; ++i) {
            Slot& sp = slots_[i];
            if (sp.state.load(std::memory_order_acquire) == kLoading && sp.left.load(std::memory_order_acquire) == 0) {
                try_ready(sp);
                if (sp.state.load(std::memory_order_acquire) == kLoading) finish(sp);
                if (sp.state.load(std::memory_order_acquire) == kFree) {
                    free_hint_ = std::min(free_hint_, i);
                    return (int) i;
                }
            }
            if (sp.pin > 0 || sp.visible || sp.state.load(std::memory_order_acquire) != kReady) continue;
            const double v = score(sp);
            if (best < 0 || v < bv) {
                best = (int) i;
                bv = v;
            }
        }
        if (best < 0) return -1;
        free_hint_ = std::min<size_t>(free_hint_, (size_t) best);
        return best;
    }

    Cfg cfg_;
    std::vector<LayerGeo> geo_;
    std::vector<os::fd_t> dfds_, pfds_;
    size_t slot_bytes_ = 0, nmax_ = 0, nalloc_ = 0, free_hint_ = 0, arena_bytes_ = 0;
    uint8_t* arena_ = nullptr;
    size_t shrunk_ = 0;
    uint64_t cancel_walks_ = 0;   // (cancel_stale's arena walks: the test's count, under mu_)
    // (mu_) an upper bound on the guesses still loading: start_fill counts a guess up, preload counts its own back down (it is
    // warm, not a guess), cancel_stale recounts them exactly.  Zero: cancel_stale has nothing to cancel and does not walk.
    size_t guesses_loading_ = 0;
    bool direct_span_ = false;   // DS4_TIER_DIRECT_SPAN (read in init): a direct tier read takes the matrices' span, not the slot
    bool pin_first_ = true;      // DS4_STORE_PIN_FIRST (read in init): begin() pins the held experts before it fills misses (0: the old order)
    bool cancel_scan_ = false;   // DS4_STORE_CANCEL_SCAN (read in init): 1 = cancel_stale() walks the arena on every call (the old way)
    std::unique_ptr<Slot[]> slots_;
    std::vector<std::pair<void*, size_t>> chunks_;
    std::vector<void*> chunk_dev_;
    std::vector<std::vector<int>> where_;
    std::vector<std::vector<float>> heat_;
    std::vector<uint64_t> hits_in_layer_;
    uint64_t clock_ = 0;
    std::mutex mu_;
    std::unique_ptr<IoQueue> io_;
    std::shared_ptr<CompletionHub> hub_;   // DS4_LANE_WAKE: the store's completion signal (null: off).  Shared with the queues that hold it.
    Stats st_;
};

/// DS4_LANE_WAKE: the order a CPU lane takes its n entries in.  A round tests ready(i) for each entry still to do: the ready ones go
/// to batch() together, in index order (one call); when none is ready, wait(rest) sleeps until one of the rest may be, and the round
/// is taken again.  So the lane computes whatever is ready and sleeps only when nothing is.  Every entry is in exactly one batch.
/// Bits: the order is only a change of the batches.  That is safe if the entries are independent (each writes its own rows), the
/// same assumption DS4_MISS_SPLIT makes.  Not checked on hardware yet: tools/dsv41/bytecheck.sh must confirm it.
template <class Ready, class Batch, class Wait>
void lane_rounds(size_t n, Ready&& ready, Batch&& batch, Wait&& wait) {
    std::vector<size_t> todo(n);
    for (size_t i = 0; i < n; ++i) todo[i] = i;
    while (!todo.empty()) {
        std::vector<size_t> go, rest;
        for (size_t i : todo) (ready(i) ? go : rest).push_back(i);
        if (!go.empty()) batch(go);
        else wait(rest);
        todo.swap(rest);
    }
}

}  // namespace strata::ds4
