// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_pool.hpp - the DeepSeek engine's CPU workers: run(n, fn) runs fn(0..n-1) on the workers and the caller and
// returns when every job is done.  The design is glmfast::Workers' (src/core/glm_fast_state.hpp): one 64-bit ticket
// (generation << 32 | next index) claimed by CAS, so a worker that wakes late for an older batch can never take a job
// of the current one; idle workers spin for a while, then sleep on a condition variable.
#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "ds4_os.hpp"
#include "ds4_tier_stats.hpp"
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace strata::ds4 {

inline void cpu_relax() {
#if defined(_MSC_VER)
    _mm_pause();
#elif defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#endif
}

class Pool {
public:
    explicit Pool(int n, const std::vector<int>& cpus = {}, int spin_us = 200) : spin_us_(spin_us) {
        for (int i = 0; i < n; ++i) {
            th_.emplace_back([this] { loop(); });
            if (!cpus.empty()) os::pin_thread(th_.back(), cpus[(size_t) i % cpus.size()]);
        }
    }
    ~Pool() {
        quit_.store(true);
        {
            std::lock_guard<std::mutex> lk(mu_);
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    int size() const { return (int) th_.size() + 1; }

    void run(int n, const std::function<void(int)>& fn) {
        start(n, fn);
        help();
        wait();
    }
    /// The batch goes to the workers; the caller may help() and must wait() before fn goes out of scope.  (Two pools
    /// - one a NUMA node - run at once this way: start both, help one, wait for both.)
    void start(int n, const std::function<void(int)>& fn) {
        n_ = n;
        if (n <= 0) return;
        const uint64_t g = (gen_.load(std::memory_order_relaxed) + 1) & 0xffffffffu;
        ticket_.store((g << 32) | 0x7fffffffu, std::memory_order_seq_cst);
        fn_ = &fn;
        total_.store(n, std::memory_order_release);
        done_.store(0, std::memory_order_relaxed);
        ticket_.store(g << 32, std::memory_order_release);
        gen_.store(g, std::memory_order_release);
        if (sleeping_.load(std::memory_order_acquire) > 0) {
            std::lock_guard<std::mutex> lk(mu_);
            cv_.notify_all();
        }
    }
    void help() {
        if (n_ > 0) work(gen_.load(std::memory_order_acquire));
    }
    void wait() {
        if (n_ <= 0) return;
        while (done_.load(std::memory_order_acquire) < n_) cpu_relax();
        n_ = 0;
    }

private:
    void work(uint64_t g) {
        for (;;) {
            uint64_t t = ticket_.load(std::memory_order_acquire);
            if ((t >> 32) != g) return;
            const int n = total_.load(std::memory_order_acquire);
            if ((int) (t & 0xffffffffu) >= n) return;
            if (!ticket_.compare_exchange_weak(t, t + 1, std::memory_order_acq_rel)) continue;
            (*fn_)((int) (t & 0xffffffffu));
            done_.fetch_add(1, std::memory_order_acq_rel);
        }
    }
    void loop() {
        uint64_t seen = gen_.load();
        for (;;) {
            const auto t0 = std::chrono::steady_clock::now();
            uint64_t g;
            for (;;) {
                g = gen_.load(std::memory_order_acquire);
                if (g != seen || quit_.load(std::memory_order_relaxed)) break;
                cpu_relax();
                if (std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(spin_us_)) {
                    std::unique_lock<std::mutex> lk(mu_);
                    sleeping_.fetch_add(1);
                    cv_.wait(lk, [&] { return gen_.load() != seen || quit_.load(); });
                    sleeping_.fetch_sub(1);
                }
            }
            if (quit_.load()) return;
            seen = g;
            work(g);
        }
    }

    std::vector<std::thread> th_;
    int spin_us_;
    int n_ = 0;   // the current batch's size (the caller's side)
    const std::function<void(int)>* fn_ = nullptr;
    alignas(64) std::atomic<uint64_t> ticket_{0};
    alignas(64) std::atomic<uint64_t> gen_{0};
    alignas(64) std::atomic<int> total_{0};
    alignas(64) std::atomic<int> done_{0};
    std::atomic<int> sleeping_{0};
    std::atomic<bool> quit_{false};
    std::mutex mu_;
    std::condition_variable cv_;
};

/// DS4_LANE_WAKE: one completion signal for a whole expert store.  Each IoQueue that carries a store slot's reads (the
/// store's own queue, every tier device's) calls notify() after it counts a read's `left` down, and so does ExpertStore::put_done
/// for a slot the GPU filled.  A waiter tests its own predicate under `mu` and sleeps on `cv` with the same lock, so a
/// completion that lands between its test and its sleep is seen, not lost.  notify() takes no per-slot state: the waiter
/// decides from its counters, so an unrelated completion only makes it look again.
struct CompletionHub {
    std::mutex mu;
    std::condition_variable cv;
    uint64_t epoch = 0;  // prefill can take a snapshot before pumping the staging ring
    void notify() {
        // (a waiter holds mu from its test until it sleeps: this lock waits for that, so the count it just lowered is seen by
        //  the waiter's next test, or the notify below wakes it from its sleep)
        { std::lock_guard<std::mutex> lk(mu); ++epoch; }
        cv.notify_all();
    }
#ifdef DS4_IOQUEUE_TEST_HOOKS
    /// Test builds only: called by a waiter under `mu` when its predicate is false, just before it sleeps.  It must not call notify().
    using WaitHook = void (*)(const CompletionHub*);
    static WaitHook& wait_hook() {
        static WaitHook f = nullptr;
        return f;
    }
#endif
};

/// File reads on threads of their own, in the order they were pushed, the pusher never waiting: the prompt path's
/// streamed experts (each part of an expert a job), unbuffered where the job has a direct descriptor and its range is
/// not all in the page cache already (the disk straight into dst, the cache neither used nor disturbed), buffered
/// otherwise or when the unbuffered read fails - the bytes land at dst + off % os::kDirectAlign either way.  A job's `left` is
/// counted down when it is done (`bad` set if it failed); wait_any() sleeps until some job finished or the timeout, and
/// wait_until_zero() sleeps until the named counter is zero.
// DS4_TIER_LOG: one tier device's runtime copy counters.  The tier's runtime reads are memcpy's from its
// mapping (ds4_store.hpp's tier hit), not pread() calls, so DS4_IO_STATS does not see them.  The histograms
// are fixed-size and updated with relaxed atomics: one job is three matrices, so the overhead is per expert,
// not per row.
struct IoLatencyHist {
    static constexpr uint64_t kBinNs = 50000;   // 50 us a bin: p50/p95 to within 50 us below the cap
    static constexpr size_t kBins = 2048;       // up to 102.4 ms; longer samples make a percentile report max
    std::atomic<uint64_t> bins[kBins] = {};
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> over{0};              // (counted in count; p95 then reports max, not a cap)
    std::atomic<uint64_t> max_ns{0};
    void add(uint64_t ns) {
        uint64_t m = max_ns.load(std::memory_order_relaxed);
        while (m < ns && !max_ns.compare_exchange_weak(m, ns, std::memory_order_relaxed)) {}
        const size_t b = (size_t) (ns / kBinNs);
        if (b >= kBins) over.fetch_add(1, std::memory_order_relaxed);
        else bins[b].fetch_add(1, std::memory_order_relaxed);
        count.fetch_add(1, std::memory_order_relaxed);
    }
    uint64_t percentile(double p) const {
        const uint64_t n = count.load(std::memory_order_relaxed);
        if (n == 0) return 0;
        uint64_t target = (uint64_t) (p * (double) n);
        if (target < 1) target = 1;
        if (target > n) target = n;
        uint64_t cum = 0;
        for (size_t i = 0; i < kBins; ++i) {
            cum += bins[i].load(std::memory_order_relaxed);
            if (cum >= target) return (uint64_t) (i + 1) * kBinNs;
        }
        return max_ns.load(std::memory_order_relaxed);
    }
};
struct AtomicStripeCount {
    std::atomic<uint64_t> jobs{0}, bytes{0};
    void add(uint64_t n) { jobs.fetch_add(1, std::memory_order_relaxed); bytes.fetch_add(n, std::memory_order_relaxed); }
    StripeJobCount snapshot() const { return {jobs.load(std::memory_order_relaxed), bytes.load(std::memory_order_relaxed)}; }
};
struct AtomicStripeIoStats {
    AtomicStripeCount mapped_submitted, direct_submitted;
    AtomicStripeCount mapped_ok, mapped_failed, mapped_cancelled;
    AtomicStripeCount direct_ok, direct_fallback, direct_failed, direct_cancelled;
    StripeIoStats snapshot() const {
        return {mapped_submitted.snapshot(), direct_submitted.snapshot(), mapped_ok.snapshot(), mapped_failed.snapshot(),
                mapped_cancelled.snapshot(), direct_ok.snapshot(), direct_fallback.snapshot(), direct_failed.snapshot(),
                direct_cancelled.snapshot()};
    }
};
struct TierIoStats {
    AtomicStripeIoStats stripe;
    std::atomic<uint64_t> started_experts{0}, started_bytes{0};
    std::atomic<uint64_t> demand_experts{0}, demand_bytes{0};
    std::atomic<uint64_t> prefetch_experts{0}, prefetch_bytes{0};
    std::atomic<uint64_t> completed_jobs{0}, completed_bytes{0};
    std::atomic<uint64_t> queue_ns{0}, service_ns{0}, wait_ns{0};
    std::atomic<uint64_t> demand_jobs{0}, demand_completed_bytes{0}, demand_queue_ns{0}, demand_service_ns{0};
    std::atomic<uint64_t> prefetch_jobs{0}, prefetch_completed_bytes{0}, prefetch_queue_ns{0}, prefetch_service_ns{0};
    std::atomic<uint64_t> cold_jobs{0}, cold_bytes{0};
    std::atomic<uint64_t> cold_queue_ns{0}, cold_service_ns{0};
    std::atomic<uint64_t> queue_depth_max{0};
    IoLatencyHist queue_hist, service_hist, cold_service_hist;

    void note_start(uint64_t bytes, bool demand) {
        started_experts.fetch_add(1, std::memory_order_relaxed);
        started_bytes.fetch_add(bytes, std::memory_order_relaxed);
        if (demand) {
            demand_experts.fetch_add(1, std::memory_order_relaxed);
            demand_bytes.fetch_add(bytes, std::memory_order_relaxed);
        } else {
            prefetch_experts.fetch_add(1, std::memory_order_relaxed);
            prefetch_bytes.fetch_add(bytes, std::memory_order_relaxed);
        }
    }
    void note_copy(uint64_t bytes, uint64_t q_ns, uint64_t s_ns, bool demand, bool cold) {
        completed_jobs.fetch_add(1, std::memory_order_relaxed);
        completed_bytes.fetch_add(bytes, std::memory_order_relaxed);
        queue_ns.fetch_add(q_ns, std::memory_order_relaxed);
        service_ns.fetch_add(s_ns, std::memory_order_relaxed);
        queue_hist.add(q_ns);
        service_hist.add(s_ns);
        if (demand) {
            demand_jobs.fetch_add(1, std::memory_order_relaxed);
            demand_completed_bytes.fetch_add(bytes, std::memory_order_relaxed);
            demand_queue_ns.fetch_add(q_ns, std::memory_order_relaxed);
            demand_service_ns.fetch_add(s_ns, std::memory_order_relaxed);
        } else {
            prefetch_jobs.fetch_add(1, std::memory_order_relaxed);
            prefetch_completed_bytes.fetch_add(bytes, std::memory_order_relaxed);
            prefetch_queue_ns.fetch_add(q_ns, std::memory_order_relaxed);
            prefetch_service_ns.fetch_add(s_ns, std::memory_order_relaxed);
        }
        if (cold) {
            cold_jobs.fetch_add(1, std::memory_order_relaxed);
            cold_bytes.fetch_add(bytes, std::memory_order_relaxed);
            cold_queue_ns.fetch_add(q_ns, std::memory_order_relaxed);
            cold_service_ns.fetch_add(s_ns, std::memory_order_relaxed);
            cold_service_hist.add(s_ns);
        }
    }
    // DS4_TIER_DIRECT: whole-slot jobs.  Counted apart from the mapped copies above: one direct job is a whole slot (padding
    // included), not comparable with the three matrix copies of a mapped expert.  `delivered` = read directly; `fallback` =
    // refused, and copied the mapped way; neither = failed.
    std::atomic<uint64_t> direct_jobs{0}, direct_ok{0}, direct_fallbacks{0}, direct_failed{0};
    std::atomic<uint64_t> direct_slot_bytes{0}, direct_slot_ok_bytes{0}, direct_payload_ok_bytes{0};
    std::atomic<uint64_t> direct_queue_ns{0}, direct_service_ns{0};
    IoLatencyHist direct_service_hist;
    void note_direct(uint64_t payload, uint64_t slot_bytes, bool delivered, bool fallback, uint64_t q_ns, uint64_t s_ns) {
        direct_jobs.fetch_add(1, std::memory_order_relaxed);
        direct_slot_bytes.fetch_add(slot_bytes, std::memory_order_relaxed);
        direct_queue_ns.fetch_add(q_ns, std::memory_order_relaxed);
        direct_service_ns.fetch_add(s_ns, std::memory_order_relaxed);
        direct_service_hist.add(s_ns);
        if (delivered) {
            direct_ok.fetch_add(1, std::memory_order_relaxed);
            direct_slot_ok_bytes.fetch_add(slot_bytes, std::memory_order_relaxed);
            direct_payload_ok_bytes.fetch_add(payload, std::memory_order_relaxed);
        } else if (fallback) {
            direct_fallbacks.fetch_add(1, std::memory_order_relaxed);
        } else {
            direct_failed.fetch_add(1, std::memory_order_relaxed);
        }
    }
    void note_wait(uint64_t ns) { wait_ns.fetch_add(ns, std::memory_order_relaxed); }
    void note_depth(uint64_t depth) {
        uint64_t cur = queue_depth_max.load(std::memory_order_relaxed);
        while (cur < depth && !queue_depth_max.compare_exchange_weak(cur, depth, std::memory_order_relaxed)) {}
    }
};

class IoQueue {
public:
    struct Job {
        os::fd_t fd = os::kBadFd, dfd = os::kBadFd;
        uint64_t off = 0, len = 0;
        uint8_t* dst = nullptr;
        std::atomic<int>* left = nullptr;
        std::atomic<bool>* bad = nullptr;
        const uint8_t* mem = nullptr;   // non-null: a copy of len bytes from memory to dst instead of a read
        const void* map = nullptr;      // the range in the file's mapping: all in the page cache, it is read buffered
        uint8_t* keep = nullptr;        // non-null (a read): the len bytes read copied here as well
        bool exact = false;             // the len bytes land exactly at dst (a ring slot: no staging slack around them)
        std::atomic<bool>* cancel = nullptr;   // set before the job starts: it is dropped unread (bad is set, left counted down)
        std::atomic<bool>* fail = nullptr;     // non-null: set when the job RAN and did not succeed (bad is set too).  A cancelled
                                               // job sets bad and never this: "dropped on purpose" and "went wrong" differ (the tier marks a device only for the latter)
        std::chrono::steady_clock::time_point pushed{};   // (DS4_IO_STATS: when it was queued)
        bool tier_copy = false;   // DS4_TIER_LOG: a runtime copy from a tier device's mapping
        bool demand = false;      // ... and it is a demand copy, not a lookahead guess
        bool tier_stripe = false; // striped runtime job, counted separately when tier tracing is enabled
        // DS4_TIER_DIRECT: a tier slot read whole through `dfd` (off, len: the device's offset alignment `align`; dst: its
        // memory alignment `mem_align`), straight into the store's slot, one job a slot.  A refusal copies `fallback` (the same
        // slot in the tier's mapping) and sets `refused`, so the device is not asked again.  `payload` is the matrices' bytes
        // (for the stats only: len includes the padding).  Not an `exact` job and not a `mem` copy.
        bool tier_direct = false;
        uint64_t align = 0, mem_align = 0, payload = 0;
        const uint8_t* fallback = nullptr;
        std::atomic<bool>* refused = nullptr;
        bool sync_fd = false;          // exact buffered fallback uses a synchronous file handle
        std::shared_ptr<CompletionHub> expert_done;  // prefill: notify only on the last job, across all queues
    };
    explicit IoQueue(int n, TierIoStats* trace = nullptr) : trace_(trace) {
        for (int i = 0; i < n; ++i) th_.emplace_back([this] { loop(); });
    }
    ~IoQueue() {
        if (st_n_.load() > 0 && std::getenv("DS4_IO_STATS") != nullptr)
            std::fprintf(stderr, "io stats: %llu jobs, mean queue wait %.2f ms, read %.2f ms, copy %.2f ms, %.2f MB a job\n",
                         (unsigned long long) st_n_.load(), st_q_.load() / 1000.0 / (double) st_n_.load(),
                         st_r_.load() / 1000.0 / (double) st_n_.load(), st_m_.load() / 1000.0 / (double) st_n_.load(),
                         st_b_.load() / 1e6 / (double) st_n_.load());
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    /// low: a read that can wait (a guess) - the jobs of the first queue are all taken before any of this one's
    void push(const Job* jobs, int n, bool low = false) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            bool traced = false;
            for (int i = 0; i < n; ++i) {
                (low ? ql_ : q_).push_back(jobs[i]);
                (low ? ql_ : q_).back().pushed = std::chrono::steady_clock::now();
                traced = traced || (jobs[i].tier_copy && !jobs[i].expert_done);
                if (trace_ && jobs[i].tier_stripe && !jobs[i].expert_done)
                    (jobs[i].tier_direct ? trace_->stripe.direct_submitted : trace_->stripe.mapped_submitted).add(jobs[i].len);
            }
            if (trace_ && traced) trace_->note_depth((uint64_t) (q_.size() + ql_.size()));
        }
        if (n == 1) cv_.notify_one();
        else cv_.notify_all();
    }
    // Turn the unread part of a speculative prefill pick into demand once routing confirms it.
    void promote(const std::atomic<int>& left) {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto it = ql_.begin(); it != ql_.end();) {
            if (it->left != &left) { ++it; continue; }
            it->demand = true;
            q_.push_back(std::move(*it));
            it = ql_.erase(it);
        }
    }
    void wait_any(int us) {
        std::unique_lock<std::mutex> lk(done_mu_);
        const uint64_t seen = done_;
        done_cv_.wait_for(lk, std::chrono::microseconds(us), [&] { return done_ != seen; });
    }
    /// DS4_LANE_WAKE: the completion signal this queue announces each finished job to (null: none, the default).  The store sets
    /// it before it pushes a slot's reads here, so every job of that slot is announced.  Taken under mu_: a job popped before the
    /// change keeps the signal it was popped with.
    void set_hub(std::shared_ptr<CompletionHub> h) {
        std::lock_guard<std::mutex> lk(mu_);
        hub_ = std::move(h);
    }
#ifdef DS4_IOQUEUE_TEST_HOOKS
    /// Test builds only (the test target defines DS4_IOQUEUE_TEST_HOOKS; production has no hook, no static and no pointer
    /// load): called at the start of each wait_until_zero(), with the queue waited on and the counter, while the
    /// completion mutex is held.  No completion can be announced between the call and the sleep, so a test that acts here
    /// knows the waiter is about to sleep.  It must not call into this queue.
    // A test may gate a particular job or report a read error without touching the source or destination.
    using ReadHook = bool (*)(const Job&);
    static ReadHook& read_hook() {
        static ReadHook f = nullptr;
        return f;
    }
    using SlotWaitHook = void (*)(const IoQueue*, const std::atomic<int>*);
    static SlotWaitHook& slot_wait_hook() {
        static SlotWaitHook f = nullptr;
        return f;
    }
#endif
    /// Blocks until `left` is zero or below (returns true) or `timeout_us` has passed (returns false).  Only the counter
    /// ends the wait: an unrelated job finishing, or one of several reads of the same slot, wakes the waiter to look and it
    /// sleeps again.  The counter is read under the completion mutex, and a job counts it down before it takes that mutex to
    /// announce itself, so a read that finishes just before the waiter sleeps cannot lose its wake-up.  The timeout is a
    /// safety net only: a counter that something else brings to zero (no job of this queue, so no notification) is seen
    /// at most `timeout_us` late.  A slot that is done already returns at once.
    /// The caller keeps `left`'s owner alive and the slot pinned (not recycled) until this returns, and the queue too.
    bool wait_until_zero(const std::atomic<int>& left, int timeout_us = 50'000) {
        std::unique_lock<std::mutex> lk(done_mu_);
#ifdef DS4_IOQUEUE_TEST_HOOKS
        if (SlotWaitHook f = slot_wait_hook()) f(this, &left);
#endif
        return done_cv_.wait_for(lk, std::chrono::microseconds(timeout_us),
                                 [&] { return left.load(std::memory_order_acquire) <= 0; });
    }
    /// Counts a completion that no job of this queue made (a slot the caller filled itself: ExpertStore::put_done), so a
    /// wait_any() sleeper looks now and not at its timeout.
    void notify_done() {
        {
            std::lock_guard<std::mutex> lk(done_mu_);
            ++done_;
        }
        done_cv_.notify_all();
    }
    double busy_ms() const { return busy_us_.load(std::memory_order_relaxed) / 1000.0; }
    uint64_t direct_bytes() const { return direct_.load(std::memory_order_relaxed); }
    uint64_t buffered_bytes() const { return buffered_.load(std::memory_order_relaxed); }

private:
    static void finish_expert(const Job& j) {
        // No slot state is accessed after the decrement. The job owns its signal through notification.
        if (j.left && j.left->fetch_sub(1, std::memory_order_acq_rel) == 1 && j.expert_done)
            j.expert_done->notify();
    }
    void loop() {
#if defined(_WIN32)
        if (std::getenv("DS4_IO_PRIO") != nullptr) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);   // (a read in flight is the lane's critical path)
#endif
        for (;;) {
            Job j;
            std::shared_ptr<CompletionHub> hub;   // (DS4_LANE_WAKE: announced to when set; a null copy costs nothing)
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return quit_ || !q_.empty() || !ql_.empty(); });
                if (quit_ && q_.empty() && ql_.empty()) return;
                std::deque<Job>& qq = q_.empty() ? ql_ : q_;
                j = std::move(qq.front());
                qq.pop_front();
                if (hub_) hub = hub_;
            }
            if (j.cancel != nullptr && j.cancel->load(std::memory_order_acquire)) {   // (dropped unread)
                if (trace_ && j.tier_stripe && !j.expert_done)
                    (j.tier_direct ? trace_->stripe.direct_cancelled : trace_->stripe.mapped_cancelled).add(j.len);
                if (j.bad != nullptr) j.bad->store(true, std::memory_order_release);
                finish_expert(j);
                {
                    std::lock_guard<std::mutex> lk(done_mu_);
                    ++done_;
                }
                done_cv_.notify_all();
                if (hub) hub->notify();
                continue;
            }
            const auto t0 = std::chrono::steady_clock::now();
            const uint64_t lead = j.off % os::kDirectAlign;
            bool ok = false, direct_hit = false, direct_fb = false;
#ifdef DS4_IOQUEUE_TEST_HOOKS
            if (ReadHook f = read_hook(); f && !f(j)) {
                // Follow the same completion and error path as a reported read failure.
            } else
#endif
            if (j.mem != nullptr) {
                const bool traced = trace_ != nullptr && j.tier_copy && !j.expert_done;
                const auto tc0 = traced ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                const bool cold = traced && !os::resident_sampled(j.mem, (size_t) j.len, 4);   // (a cold mapped tier range faults from the drive)
                std::memcpy(j.dst, j.mem, (size_t) j.len);
                if (traced) {
                    using ns = std::chrono::nanoseconds;
                    const auto tc1 = std::chrono::steady_clock::now();
                    trace_->note_copy(j.len,
                                      (uint64_t) std::chrono::duration_cast<ns>(tc0 - j.pushed).count(),
                                      (uint64_t) std::chrono::duration_cast<ns>(tc1 - tc0).count(), j.demand, cold);
                }
                ok = true;
            } else if (j.tier_direct) {
                // DS4_TIER_DIRECT: the tier's whole slot straight into the store's slot (no bounce buffer, no move: the two have
                // the same layout).  Refused: copied from the tier's mapping instead, and the device is not asked again.  Failed:
                // the bytes are not whole, so the job fails and the store marks the device (fail) as for any failed copy.
                const os::DirectRead r = os::pread_exact_direct(j.dfd, j.dst, j.len, j.off, j.align, j.mem_align);
                if (r == os::DirectRead::kOk) {
                    direct_hit = ok = true;
                } else if (r == os::DirectRead::kUnsupported) {
                    if (j.refused != nullptr) j.refused->store(true, std::memory_order_release);
                    if (j.fallback != nullptr) {
                        std::memcpy(j.dst, j.fallback, (size_t) j.len);
                        direct_fb = ok = true;
                    }
                }
            } else if (j.exact) {
                // unbuffered through a bounce buffer (the disk reads aligned spans), then the bytes to where they belong
                thread_local std::vector<uint8_t> bounce;
                const uint64_t span = os::direct_span(j.off, j.len);
                if (j.dfd != os::kBadFd && bounce.size() < span + os::kDirectAlign)
                    bounce.resize((size_t) span + 2 * os::kDirectAlign);
                uint8_t* bp = (uint8_t*) (((uintptr_t) bounce.data() + os::kDirectAlign - 1) & ~(uintptr_t) (os::kDirectAlign - 1));
                if (j.dfd != os::kBadFd && !(j.map != nullptr && os::resident(j.map, (size_t) j.len)) &&
                    os::pread_direct(j.dfd, bp, j.len, j.off)) {
                    std::memcpy(j.dst, bp + lead, (size_t) j.len);
                    ok = true;
                    direct_.fetch_add(j.len, std::memory_order_relaxed);
                } else if (j.sync_fd ? os::pread_full(j.fd, j.dst, j.len, j.off) : os::pread_par(j.fd, j.dst, j.len, j.off)) {
                    ok = true;
                    buffered_.fetch_add(j.len, std::memory_order_relaxed);
                }
            } else if (j.dfd != os::kBadFd && !(j.map != nullptr && os::resident(j.map, (size_t) j.len)) &&
                       os::pread_direct(j.dfd, j.dst, j.len, j.off)) {   // (the page cache's ones: buffered, below)
                ok = true;
                direct_.fetch_add(j.len, std::memory_order_relaxed);
            } else if (os::pread_par(j.fd, j.dst + lead, j.len, j.off)) {
                ok = true;
                buffered_.fetch_add(j.len, std::memory_order_relaxed);
            }
            const auto t_rd = std::chrono::steady_clock::now();
            if (ok && j.keep != nullptr && (j.mem == nullptr || j.tier_stripe))
                std::memmove(j.keep, (j.exact || j.mem != nullptr) ? j.dst : j.dst + lead, (size_t) j.len);
            if (trace_ && j.tier_stripe && !j.expert_done) {
                if (j.tier_direct) {
                    if (direct_hit) trace_->stripe.direct_ok.add(j.len);
                    else if (direct_fb) trace_->stripe.direct_fallback.add(j.len);
                    else trace_->stripe.direct_failed.add(j.len);
                } else if (ok) trace_->stripe.mapped_ok.add(j.len);
                else trace_->stripe.mapped_failed.add(j.len);
            }
            if (j.tier_direct && trace_ != nullptr) {
                using ns = std::chrono::nanoseconds;
                trace_->note_direct(j.payload, j.len, direct_hit, direct_fb,
                                    (uint64_t) std::chrono::duration_cast<ns>(t0 - j.pushed).count(),
                                    (uint64_t) std::chrono::duration_cast<ns>(t_rd - t0).count());
            }
            {
                static const bool stats = std::getenv("DS4_IO_STATS") != nullptr;
                if (stats && j.mem == nullptr && j.cancel == nullptr && !j.tier_direct) {   // (demand reads only)
                    const auto t_end = std::chrono::steady_clock::now();
                    using us = std::chrono::microseconds;
                    st_n_.fetch_add(1);
                    st_q_.fetch_add((uint64_t) std::chrono::duration_cast<us>(t0 - j.pushed).count());
                    st_r_.fetch_add((uint64_t) std::chrono::duration_cast<us>(t_rd - t0).count());
                    st_m_.fetch_add((uint64_t) std::chrono::duration_cast<us>(t_end - t_rd).count());
                    st_b_.fetch_add(j.len);
                }
            }
            busy_us_.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - t0).count(),
                               std::memory_order_relaxed);
            if (!ok && j.fail != nullptr) j.fail->store(true, std::memory_order_release);   // (before bad and left: whoever sees them done sees why)
            if (!ok && j.bad != nullptr) j.bad->store(true, std::memory_order_release);
            finish_expert(j);
            {
                std::lock_guard<std::mutex> lk(done_mu_);
                ++done_;
            }
            done_cv_.notify_all();
            if (hub) hub->notify();
        }
    }
    std::vector<std::thread> th_;
    std::mutex mu_, done_mu_;
    std::condition_variable cv_, done_cv_;
    std::deque<Job> q_, ql_;
    uint64_t done_ = 0;
    bool quit_ = false;
    std::atomic<uint64_t> busy_us_{0}, direct_{0}, buffered_{0};
    std::atomic<uint64_t> st_n_{0}, st_q_{0}, st_r_{0}, st_m_{0}, st_b_{0};   // DS4_IO_STATS: jobs, queue wait / read / copy us, bytes
    TierIoStats* trace_ = nullptr;   // (DS4_TIER_LOG: the tier queue's counters; null on the store's queue)
    std::shared_ptr<CompletionHub> hub_;   // (DS4_LANE_WAKE: guarded by mu_; null by default)
};

}  // namespace strata::ds4
