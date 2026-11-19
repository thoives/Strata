// The DeepSeek engine's expert tier without a GPU or a model (src/dsv41/ds4_tier.hpp): docs/EXPERT_TIER_DSV41_PLAN.md
// §14. Synthetic geometry and small files: the fill's bytes equal the shard's (acceptance 3), the store's reads of tier
// experts come off the tier and are the same bytes (§8.1/§8.2), a start reuses the placement and copies nothing (5), a
// profile change does not refill (6), while a pack, guard, torn-header, corrupt-table, abandoned-fill or verify=full
// mismatch refills (7), and a failed copy marks its device (§9).
// DS4_TIER_DIRECT (opt-in, Linux): the read policy with injected errno values and short reads, the store and queue on a forced
// direct tier (any filesystem), and a real O_DIRECT read of the tier's slots.  The real read is SKIPPED (reported, not counted)
// where the test directory refuses O_DIRECT reads or the device gives no alignment; set DS4_TIER_TEST_DIR to one that does.
// DS4_TIER_WAKE (opt-in): IoQueue::wait_until_zero on a bare queue (a partial completion, a lost notification, a completion just
// before the sleep), and the store's wait_slots() on a held tier queue - the mapped copies and the direct read - which must
// wait on the device's queue (and on the store's queue with the switch off), end only with its own slot, and fall back to the
// store's queue for a slot the GPU put in (put_begin).  The queue's test hook needs DS4_IOQUEUE_TEST_HOOKS (CMake defines it).
//   ds4_expert_tier_test   exit 0: all passed (a skip is reported in the last line)
#ifndef DS4_IOQUEUE_TEST_HOOKS
#error "build this test with -DDS4_IOQUEUE_TEST_HOOKS: its wait checks observe IoQueue::wait_until_zero through the queue's test hook"
#endif
#include "ds4_tier.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <atomic>
#include <map>
#include <memory>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
namespace os = strata::ds4::os;
using strata::ds4::ExpertStore;
using strata::ds4::TierSpec;
using strata::ds4::ExpertTierSet;
using strata::ds4::TierLayerGeo;
using strata::ds4::TierGeoShape;
using strata::ds4::ds4_geometry_text;

static int fails = 0;
#define CHECK(c)                                                                     \
    do {                                                                             \
        if (!(c)) {                                                                \
            std::printf("FAIL line %d: %s\n", __LINE__, #c);                       \
            ++fails;                                                               \
        }                                                                          \
    } while (0)

// The synthetic pack: 3 layers x 6 experts; an expert's three matrices are 4096 bytes each with 8192-byte gaps between
// them (the store's LayerGeo blob), role r of layer l packed in the one shard at (l * 3 + r) * NE * 4096.
constexpr size_t kTierHeaderBytesForTest = 176;
constexpr int NL = 3, NE = 6, MB4 = 4096;
constexpr uint64_t ROLE_B = (uint64_t) NE * MB4;             // one role's bytes in the file
constexpr uint64_t BLOB = 3ull * MB4 + 2ull * 8192;          // 28672: the blob with its gaps
constexpr uint64_t SLOT = 45056;                             // the store's slot: the blob plus kSlack either side
constexpr uint64_t SHARD_BYTES = 3ull * NL * ROLE_B;         // 229376

static uint8_t shard_byte(uint64_t pos) { return (uint8_t) (pos * 37u + (pos >> 9) * 101u + 7u); }

// The file position of (layer, e)'s matrix r, byte i
static uint64_t src_pos(int l, int e, int r, uint64_t i) { return (uint64_t) (l * 3 + r) * ROLE_B + (uint64_t) e * MB4 + i; }

static bool same_matrix(const uint8_t* p, int l, int e, int r) {
    for (uint64_t i = 0; i < (uint64_t) MB4; ++i)
        if (p[i] != shard_byte(src_pos(l, e, r, i))) return false;
    return true;
}

static std::vector<ExpertStore::LayerGeo> make_geo() {
    std::vector<ExpertStore::LayerGeo> g((size_t) NL);
    for (int l = 0; l < NL; ++l) {
        ExpertStore::LayerGeo& gl = g[(size_t) l];
        gl.n_exp = NE;
        for (int r = 0; r < 3; ++r) {
            gl.file[r] = 0;
            gl.off[r] = (uint64_t) (l * 3 + r) * ROLE_B;
            gl.stride[r] = MB4;
            gl.bytes[r] = MB4;
            gl.tr[r] = (uint64_t) r * (MB4 + 8192);
        }
        gl.blob = BLOB;
    }
    return g;
}

// §7.1's geometry, as plain values: a GGUF layer (MXFP4: its matrices read in place, region_* left empty as a GGUF layer
// leaves them - the input that crashed init_tier before the identity was layout-aware) and an EXL3 layer (regions and pieces).
// The down matrix sits in the pack's second file, so the file list matters.
static TierLayerGeo mx_geo(int il) {
    TierLayerGeo t;
    t.il = il;
    t.layout = TierLayerGeo::kMxFp4;
    t.n_exp = NE;
    t.fmt = 2;   // gg::kFmtMXFP4
    t.blob = BLOB;
    for (int r = 0; r < 3; ++r) {
        t.file[r] = r == 2 ? 1 : 0;
        t.off[r] = (uint64_t) (il * 3 + r) * ROLE_B;
        t.stride[r] = MB4;
        t.bytes[r] = MB4;
        t.tr[r] = (uint64_t) r * (MB4 + 8192);
        t.type[r] = r == 2 ? 14 : 12;
    }
    return t;
}
static TierLayerGeo region_geo(int il) {
    TierLayerGeo t;
    t.il = il;
    t.layout = TierLayerGeo::kRegion;
    t.n_exp = 2;
    t.blob = 1024;
    for (int e = 0; e < 2; ++e) {
        t.region_file.push_back(e);
        t.region_off.push_back(4096ull * (uint64_t) (e + 1));
        t.region_bytes.push_back(512);
    }
    for (int r = 0; r < 3; ++r) {
        t.type[r] = 3;   // EXL3 bits
        for (int q = 0; q < 3; ++q) {
            t.piece_src[r][q] = (uint64_t) (r * 3 + q) * 48;
            t.piece_bytes[r][q] = 48;
            t.piece_dst[r][q] = (uint64_t) (r * 3 + q) * 96;
        }
    }
    return t;
}
static TierGeoShape geo_shape() {
    TierGeoShape s;
    s.dim = 64;
    s.ff = 32;
    s.nl = 3;
    return s;
}

// Flip one byte of a file (a torn header, a corrupt table byte, a data byte past its chunk crc)
static void poke(const std::string& p, size_t off) {
    std::FILE* f = std::fopen(p.c_str(), "r+b");
    std::fseek(f, (long) off, SEEK_SET);
    uint8_t b = 0;
    if (std::fread(&b, 1, 1, f) == 1) {
        std::fseek(f, (long) off, SEEK_SET);
        b ^= 0x5Au;
        std::fwrite(&b, 1, 1, f);
    }
    std::fclose(f);
}

// One scenario's world: its own dir, shard, tier devices, store and device specs.
struct World {
    fs::path dir;
    std::string shard, devA, devB;
    std::vector<os::fd_t> dfds, pfds;
    ExpertStore st;
    std::vector<TierSpec> specs;

    World(const fs::path& base, const std::string& name, bool verify_full) : dir(base / name) {
        fs::create_directories(dir);
        shard = (dir / "shard.gguf").string();
        devA = (dir / "tierA.img").string();
        devB = (dir / "tierB.img").string();
        std::vector<uint8_t> data(SHARD_BYTES);
        for (uint64_t p = 0; p < SHARD_BYTES; ++p) data[(size_t) p] = shard_byte(p);
        std::FILE* f = std::fopen(shard.c_str(), "wb");
        std::fwrite(data.data(), 1, data.size(), f);
        std::fclose(f);
        for (const std::string& d : {devA, devB}) {   // (resize_file needs the file to exist: create it empty, then extend)
            std::FILE* f = std::fopen(d.c_str(), "wb");
            std::fclose(f);
        }
        fs::resize_file(devA, 16ull << 20);   // (bigger than any placement of this test needs)
        fs::resize_file(devB, 12ull << 20);
        dfds = {os::open_direct(shard)};
        pfds = {os::open_ro(shard)};
        ExpertStore::Cfg cfg;
        cfg.budget_bytes = 10ull * SLOT;   // (10 slots: enough for the checks, and no more)
        cfg.threads = 2;
        cfg.floor_bytes = 1ull << 20;
        CHECK(st.init(cfg, make_geo(), dfds, pfds));
        CHECK(st.slot_bytes() == SLOT);
        specs = {{devA, 16, 0, verify_full}, {devB, 12, 0, verify_full}};
    }
    ~World() {
        for (os::fd_t fd : dfds) os::close_fd(fd);
        for (os::fd_t fd : pfds) os::close_fd(fd);
        fs::remove_all(dir);
    }
};

// The store's tier hooks, wired the way ds4_model.cu's init_tier does (§8.1/§9): the fill takes the device queues
// through them (tier_fill_read), and find answers only once the set is readable, so wiring early is safe.
static void tier_hooks(World& w, const std::unique_ptr<ExpertTierSet>& t) {
    w.st.set_tier([&](int l, int e, int& dev, uint64_t& s) { return t->find(l, e, dev, s); },
                  [&](int dev, uint64_t s) { return t->slot_base(dev, s); },
                  [&](int dev) { return t->queue(dev); },
                  [&](int dev) { t->note_refill(dev); });
}

// The fill, wired the way ds4_model.cu does it (the store's read-job builder landing in a tier slot, §9).
static bool do_fill(World& w, ExpertTierSet* t, ExpertTierSet::Fill& f) {
    std::string err;
    return t->fill(
        [&](int l, int e, uint8_t* sb, int dev, std::atomic<int>& left, std::atomic<bool>& bad, uint64_t& bytes) {
            return w.st.tier_fill_read(l, e, sb, dev, left, bad, bytes);
        },
        f, err);
}

// Every placed expert's bytes: tier slot == the file's (acceptance 3).  `homes` records where it is for the reuse checks.
static int check_bytes(World& w, ExpertTierSet* t, const std::vector<std::pair<int32_t, int32_t>>& ranked,
                       std::map<std::pair<int, int>, std::pair<int, uint64_t>>& homes) {
    int bad = 0;
    for (const auto& x : ranked) {
        int dev = -1;
        uint64_t k = 0;
        if (!t->find(x.first, x.second, dev, k)) { ++bad; continue; }
        homes[{x.first, x.second}] = {dev, k};
        uint8_t* sb = t->slot_base(dev, k);
        if (sb == nullptr) { ++bad; continue; }
        for (int r = 0; r < 3; ++r)
            if (!same_matrix(sb + (size_t) r * (MB4 + 8192), x.first, x.second, r)) ++bad;
    }
    return bad;
}

// A hook set that also names the drain (the model's init_tier does the same): the store's destructor runs it first.  `t`
// must outlive the store - the tier is declared before the world that owns the store, like Model::Impl's members.
static void tier_hooks_drain(World& w, ExpertTierSet* t, std::atomic<int>* fail_calls) {
    w.st.set_tier([t](int l, int e, int& dev, uint64_t& s) { return t->find(l, e, dev, s); },
                  [t](int dev, uint64_t s) { return t->slot_base(dev, s); },
                  [t](int dev) { return t->queue(dev); },
                  [t, fail_calls](int dev) { if (fail_calls != nullptr) fail_calls->fetch_add(1); t->note_refill(dev); },
                  [t]() { t->drain(); });
}

static void set_env(const char* k, const char* v) {
#ifdef _WIN32
    _putenv_s(k, v != nullptr ? v : "");
#else
    if (v != nullptr) setenv(k, v, 1); else unsetenv(k);
#endif
}

// A copy long enough to hold a one-thread queue while the test acts (a 256 MiB memcpy).
struct BigCopy {
    std::vector<uint8_t> src, dst;
    std::atomic<int> left{1};
    explicit BigCopy(size_t n = 256ull << 20) : src(n, 7), dst(n, 0) {}
    void push(strata::ds4::IoQueue* q) {
        strata::ds4::IoQueue::Job j;
        j.mem = src.data();
        j.dst = dst.data();
        j.len = src.size();
        j.left = &left;
        q->push(&j, 1, false);
    }
};

static int skips = 0;   // (direct-I/O checks that could not run here: reported, never counted as passed)

// ---- IoQueue::wait_until_zero (ds4_pool.hpp): the wait on one counter ------------------------------------------------------------
using strata::ds4::IoQueue;
using test_clock = std::chrono::steady_clock;

static bool wait_flag(const std::atomic<bool>& f, int ms) {   // true when f was set within ms
    const auto t0 = test_clock::now();
    while (!f.load() && test_clock::now() - t0 < std::chrono::milliseconds(ms)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return f.load();
}
static bool wait_below(const std::atomic<int>& v, int bound, int ms) {   // true when v <= bound within ms
    const auto t0 = test_clock::now();
    while (v.load() > bound && test_clock::now() - t0 < std::chrono::milliseconds(ms)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return v.load() <= bound;
}

// The queue's test hook (DS4_IOQUEUE_TEST_HOOKS): runs at the start of every wait_until_zero(), under the queue's completion mutex,
// so no completion can be announced between it and the sleep.  It records the calls (which queue, which counter), and `act` may
// act at that point.  `early` counts calls that began sooner after the one before than the safety-net timeout (50 ms) allows:
// the wait before it ended without its counter being zero and without the timeout (an unrelated completion woke it for good).
// A correct wait cannot raise it, however slowly the test runs.
struct SlotWaitObs {
    std::mutex mu;
    std::condition_variable cv;
    int calls = 0, early = 0, wrong_queue = 0;
    const IoQueue* expect = nullptr;   // when set: every call must be on this queue (wrong_queue counts the others)
    const IoQueue* queue = nullptr;
    const std::atomic<int>* left = nullptr;
    test_clock::time_point last{};
    std::function<void(const std::atomic<int>*)> act;
    bool await_calls(int n, int ms) {
        std::unique_lock<std::mutex> lk(mu);
        return cv.wait_for(lk, std::chrono::milliseconds(ms), [&] { return calls >= n; });
    }
    int calls_now() {
        std::lock_guard<std::mutex> lk(mu);
        return calls;
    }
};
static SlotWaitObs* g_slot_wait_obs = nullptr;

static void observe_slot_wait(const IoQueue* q, const std::atomic<int>* left) {
    SlotWaitObs* o = g_slot_wait_obs;
    if (o == nullptr) return;
    std::function<void(const std::atomic<int>*)> act;
    {
        std::lock_guard<std::mutex> lk(o->mu);
        const auto now = test_clock::now();
        if (o->calls > 0 && now - o->last < std::chrono::milliseconds(40)) ++o->early;
        o->last = now;
        ++o->calls;
        if (o->expect != nullptr && q != o->expect) ++o->wrong_queue;
        o->queue = q;
        o->left = left;
        act = o->act;
    }
    o->cv.notify_all();
    if (act) act(left);
}

// Installs `obs` as the queue's hook for a scope.
struct SlotWaitScope {
    IoQueue::SlotWaitHook saved_hook;
    SlotWaitObs* saved_obs;
    explicit SlotWaitScope(SlotWaitObs& o) : saved_hook(IoQueue::slot_wait_hook()), saved_obs(g_slot_wait_obs) {
        g_slot_wait_obs = &o;
        IoQueue::slot_wait_hook() = observe_slot_wait;
    }
    ~SlotWaitScope() {
        IoQueue::slot_wait_hook() = saved_hook;
        g_slot_wait_obs = saved_obs;
    }
};

// The wait on a bare queue: no store, no tier, no files.  A mem job is a plain copy that counts its `left` down.
static void queue_wait_checks() {
    const std::vector<uint8_t> src(4096, 3);
    auto copy_job = [&](uint8_t* dst, std::atomic<int>* left) {
        IoQueue::Job j;
        j.mem = src.data();
        j.dst = dst;
        j.len = src.size();
        j.left = left;
        return j;
    };
    auto us_since = [](test_clock::time_point t0) {
        return (long long) std::chrono::duration_cast<std::chrono::microseconds>(test_clock::now() - t0).count();
    };

    // A partial completion does not end the wait: two of the slot's three reads are done, the waiter still sleeps; the third
    // ends it.  (The waiter holds the completion mutex until it sleeps, so the two reads are announced to a sleeper.)
    {
        std::vector<uint8_t> dst(3 * 4096, 0);
        std::atomic<int> left{3};
        IoQueue q(1);
        SlotWaitObs obs;
        SlotWaitScope scope(obs);
        std::atomic<bool> returned{false};
        std::atomic<int> got{-1};
        std::thread waiter([&] {
            got = q.wait_until_zero(left, 5'000'000) ? 1 : 0;
            returned = true;
        });
        CHECK(obs.await_calls(1, 2000));
        IoQueue::Job jobs[3] = {copy_job(dst.data(), &left), copy_job(dst.data() + 4096, &left), copy_job(dst.data() + 8192, &left)};
        q.push(jobs, 2, false);
        CHECK(wait_below(left, 1, 5000));   // (both done and announced)
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        CHECK(!returned.load());            // one read is still out: the wait goes on
        q.push(jobs + 2, 1, false);
        CHECK(wait_flag(returned, 3000));
        waiter.join();
        CHECK(got.load() == 1 && left.load() == 0);   // (done, not timed out)
    }

    // A lost notification costs the timeout, not a hang: the counter is zeroed with no job and no signal while the waiter sleeps.
    {
        std::vector<uint8_t> dummy(4096, 0);
        std::atomic<int> left{1};
        IoQueue q(1);
        SlotWaitObs obs;
        SlotWaitScope scope(obs);
        std::atomic<bool> returned{false};
        std::atomic<int> got{-1};
        std::atomic<long long> elapsed_us{0};
        std::thread waiter([&] {
            const auto t0 = test_clock::now();
            got = q.wait_until_zero(left, 20'000) ? 1 : 0;
            elapsed_us = us_since(t0);
            returned = true;
        });
        CHECK(obs.await_calls(1, 2000));
        left.store(0, std::memory_order_release);   // (no queue signal)
        const bool woke = wait_flag(returned, 2000);
        CHECK(woke);   // (a wait without a bound sleeps for ever here)
        if (!woke) {   // (let a broken wait end, so the thread joins: any completion of the queue announces)
            IoQueue::Job nudge = copy_job(dummy.data(), nullptr);
            q.push(&nudge, 1, false);
        }
        waiter.join();
        CHECK(got.load() == 1);   // (the predicate is read once more at the timeout: the zeroed counter is seen)
        CHECK(elapsed_us.load() < 1'000'000);
    }

    // With nothing done the wait returns false at its timeout (not before, not much after).  On a thread with a watchdog: a wait
    // with no bound would never return, and the test must fail, not hang.
    {
        std::vector<uint8_t> dummy(4096, 0);
        std::atomic<int> left{1};
        IoQueue q(1);
        std::atomic<bool> returned{false};
        std::atomic<int> got{-1};
        std::atomic<long long> elapsed_us{0};
        std::thread waiter([&] {
            const auto t0 = test_clock::now();
            got = q.wait_until_zero(left, 20'000) ? 1 : 0;
            elapsed_us = us_since(t0);
            returned = true;
        });
        const bool woke = wait_flag(returned, 2000);
        CHECK(woke);
        if (!woke) {   // (let a broken wait end: the count to zero, then a completion of the queue announces it)
            left.store(0, std::memory_order_release);
            IoQueue::Job nudge = copy_job(dummy.data(), nullptr);
            q.push(&nudge, 1, false);
        }
        waiter.join();
        CHECK(got.load() == 0);
        CHECK(elapsed_us.load() >= 19'000 && elapsed_us.load() < 2'000'000);
    }

    // notify_done(): a completion that no job made wakes a wait_any() sleeper; its timeout here is twenty seconds.  The notice is
    // repeated while the test waits, so a sleeper that was not asleep yet when the first came cannot fail it.
    {
        std::vector<uint8_t> dummy(4096, 0);
        IoQueue q(1);
        std::atomic<bool> returned{false};
        std::thread sleeper([&] {
            q.wait_any(20'000'000);
            returned = true;
        });
        const auto t0 = test_clock::now();
        while (!returned.load() && test_clock::now() - t0 < std::chrono::seconds(2)) {
            q.notify_done();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        CHECK(returned.load());
        if (!returned.load()) {   // (a notice that does nothing: any completion of the queue ends the sleep, so the thread joins)
            IoQueue::Job nudge = copy_job(dummy.data(), nullptr);
            q.push(&nudge, 1, false);
        }
        sleeper.join();
    }

    // A slot that is done already returns at once, with a long timeout; so does a count that reaches zero (no signal) after the
    // waiter took the mutex and before it sleeps: a wait that slept first and looked later would take the five seconds.
    {
        std::atomic<int> left{0};
        IoQueue q(1);
        auto t0 = test_clock::now();
        CHECK(q.wait_until_zero(left, 5'000'000));
        CHECK(us_since(t0) < 1'000'000);

        left.store(1);
        SlotWaitObs obs;
        obs.act = [&](const std::atomic<int>*) { left.store(0, std::memory_order_release); };
        SlotWaitScope scope(obs);
        t0 = test_clock::now();
        CHECK(q.wait_until_zero(left, 5'000'000));
        CHECK(us_since(t0) < 1'000'000);
        CHECK(obs.calls_now() == 1);
    }
}
#if !defined(_WIN32)
// ---- DS4_TIER_DIRECT (opt-in, Linux): one whole-slot direct read into a store slot (ds4_tier.hpp, ds4_store.hpp) --------------
// Three kinds of check.  (1) The read policy on its own (os::pread_exact_direct, with injected errno values and short reads):
// any filesystem.  (2) The store and the tier's queue on a FORCED direct tier (set_direct_test_force: an ordinary buffered
// descriptor, aligned as the slots are, so the retry, refusal and failure paths run on any filesystem).  (3) A real O_DIRECT
// read of the tier's slots, which SKIPS (and is not counted as a pass) where the test directory refuses O_DIRECT reads; set
// DS4_TIER_TEST_DIR to a directory that takes them.
enum Inject { kPass, kEintr, kShort4k, kShortOdd, kEinval, kEio, kZero, kBlock };
static std::mutex g_inj_mu;
static std::vector<int> g_inj;   // the next injected outcomes of the direct path's reads, front first (empty: pass-through)
static int g_inj_calls = 0;      // direct reads that reached the injected read

// A deterministic gate for an injected read: the worker blocks in inject_pread until the test lets it finish.
struct ReadGate {
    std::mutex mu;
    std::condition_variable cv;
    bool open = false;
    void close() {
        std::lock_guard<std::mutex> lk(mu);
        open = false;
    }
    void release() {
        {
            std::lock_guard<std::mutex> lk(mu);
            open = true;
        }
        cv.notify_all();
    }
    void await() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return open; });
    }
};
static ReadGate g_read_gate;

static ssize_t inject_pread(int fd, void* buf, size_t n, off_t off) {
    int code = kPass;
    {
        std::lock_guard<std::mutex> lk(g_inj_mu);
        ++g_inj_calls;
        if (!g_inj.empty()) {
            code = g_inj.front();
            g_inj.erase(g_inj.begin());
        }
    }
    switch (code) {
        case kEintr: errno = EINTR; return -1;
        case kShort4k: return ::pread(fd, buf, n > 4096 ? 4096 : n, off);   // a short read at an aligned point
        case kShortOdd: return ::pread(fd, buf, n > 512 ? 512 : n, off);    // a short read that leaves an unaligned remainder
        case kEinval: errno = EINVAL; return -1;
        case kEio: errno = EIO; return -1;
        case kZero: return 0;
        case kBlock:
            g_read_gate.await();
            return ::pread(fd, buf, n, off);
        default: return ::pread(fd, buf, n, off);
    }
}
static void inject_plan(const std::vector<int>& plan) {
    std::lock_guard<std::mutex> lk(g_inj_mu);
    g_inj = plan;
}
static int inject_calls() {
    std::lock_guard<std::mutex> lk(g_inj_mu);
    return g_inj_calls;
}

// The only worker of a queue, held in a gated injected read (g_read_gate) of an ordinary file: whatever is queued behind it waits
// until the test opens the gate.  hold() returns once the worker is in the gate.
struct WorkerHold {
    int fd = -1;
    void* buf = nullptr;
    std::atomic<int> left{1};
    explicit WorkerHold(const std::string& path) : fd(::open(path.c_str(), O_RDONLY)) {
        CHECK(fd >= 0);
        CHECK(posix_memalign(&buf, 4096, 4096) == 0);
    }
    ~WorkerHold() {
        if (fd >= 0) ::close(fd);
        std::free(buf);
    }
    void hold(IoQueue* q) {
        g_read_gate.close();
        inject_plan({kBlock});
        const int calls0 = inject_calls();
        IoQueue::Job j;
        j.dfd = (os::fd_t) fd;
        j.off = 0;
        j.len = 4096;
        j.dst = (uint8_t*) buf;
        j.align = 4096;
        j.mem_align = 4096;
        j.tier_direct = true;
        j.left = &left;
        q->push(&j, 1, false);
        for (int i = 0; i < 5000 && inject_calls() == calls0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        CHECK(inject_calls() > calls0);
    }
    bool done(int ms) const { return wait_below(left, 0, ms); }   // (after the gate is open)
};

// The read policy alone: the outcome, the bytes, the retries, and what is refused without any read being sent.
static void direct_policy_checks(const fs::path& dir) {
    const std::string p = (dir / "policy.bin").string();
    std::vector<uint8_t> data(3 * MB4);
    for (size_t i = 0; i < data.size(); ++i) data[i] = shard_byte(i);
    {
        std::FILE* f = std::fopen(p.c_str(), "wb");
        std::fwrite(data.data(), 1, data.size(), f);
        std::fclose(f);
    }
    const int fd = ::open(p.c_str(), O_RDONLY);
    CHECK(fd >= 0);
    void* raw = nullptr;
    CHECK(posix_memalign(&raw, 4096, 2 * MB4) == 0);
    uint8_t* buf = (uint8_t*) raw;
    const os::DirectPreadFn saved = os::direct_pread_hook();
    os::direct_pread_hook() = inject_pread;
    // one read of n bytes at off under `plan`: the outcome must be `want`, and for kOk the bytes must be the file's
    auto run = [&](const std::vector<int>& plan, uint64_t off, uint64_t n, os::DirectRead want) {
        inject_plan(plan);
        std::memset(buf, 0, 2 * MB4);
        const os::DirectRead r = os::pread_exact_direct((os::fd_t) fd, buf, n, off, 4096, 4096);
        CHECK(r == want);
        if (r == os::DirectRead::kOk) CHECK(std::memcmp(buf, data.data() + off, (size_t) n) == 0);
    };
    run({}, MB4, MB4, os::DirectRead::kOk);                             // (pass-through)
    run({kEintr}, 0, 2 * MB4, os::DirectRead::kOk);                     // EINTR: retried
    run({kShort4k}, 0, 2 * MB4, os::DirectRead::kOk);                   // a short aligned read: continued
    run({kShortOdd}, 0, 2 * MB4, os::DirectRead::kFailed);              // a short unaligned read: cannot continue
    run({kZero}, 0, 2 * MB4, os::DirectRead::kFailed);                  // the device ends before the slot does
    run({kEio}, 0, 2 * MB4, os::DirectRead::kFailed);                   // a genuine I/O error: a device failure
    run({kEinval}, 0, 2 * MB4, os::DirectRead::kUnsupported);           // refused: the device does not do direct reads
    run({kShort4k, kEinval}, 0, 2 * MB4, os::DirectRead::kUnsupported); // refused after a partial read: still refused
    // misaligned requests are refused before any read is sent
    int before = inject_calls();
    run({}, 512, MB4, os::DirectRead::kUnsupported);                    // offset
    run({}, 0, MB4 + 512, os::DirectRead::kUnsupported);                // length
    CHECK(inject_calls() == before);
    {
        const os::DirectRead r = os::pread_exact_direct((os::fd_t) fd, buf + 512, MB4, 0, 4096, 4096);   // memory
        CHECK(r == os::DirectRead::kUnsupported);
        CHECK(inject_calls() == before);
    }
    os::direct_pread_hook() = saved;
    CHECK(::close(fd) == 0);
    std::free(raw);
    fs::remove(p);
}

#if defined(__linux__)
// Linux: whether the mapping that holds `addr` carries the kernel flag `flag` (/proc/self/smaps VmFlags: "dc" = not copied on fork).
static bool vma_has_flag(uintptr_t addr, const std::string& flag) {
    std::FILE* f = std::fopen("/proc/self/smaps", "r");
    if (f == nullptr) return false;
    char line[512];
    bool inside = false, found = false;
    while (std::fgets(line, sizeof line, f) != nullptr) {
        unsigned long long lo = 0, hi = 0;
        if (std::sscanf(line, "%llx-%llx ", &lo, &hi) == 2) {   // a mapping's header line
            inside = addr >= (uintptr_t) lo && addr < (uintptr_t) hi;
            continue;
        }
        if (inside && std::strncmp(line, "VmFlags:", 8) == 0) {
            std::string s(line + 8);
            size_t p = 0;
            while (p < s.size()) {
                const size_t q = s.find_first_of(" \n", p);
                const std::string tok = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
                if (tok == flag) found = true;
                if (q == std::string::npos) break;
                p = q + 1;
            }
            break;
        }
    }
    std::fclose(f);
    return found;
}
#endif

// A tier's test world: the World (its shard and devices in `base`), the store wired the way init_tier wires it (the drain,
// and the direct hook when `direct` is set), and the tier.  The tier is closed (or dropped) by the caller, after unhook().
struct Forced {
    std::atomic<int> fails{0};   // the device failures the store reported (its tier_fail hook)
    std::unique_ptr<ExpertTierSet> t;
    std::unique_ptr<World> w;
};

static void unhook(Forced& f) {   // the store lets go of the tier before the tier closes (no hook may reach a closed tier)
    f.w->st.set_tier(nullptr, nullptr, nullptr, nullptr);
    f.w->st.set_tier_direct(nullptr);
    f.w->st.set_tier_trace(nullptr, nullptr);
}

// (as ds4_model.cu's init_tier: the drain, the direct hook, and the trace hooks when DS4_TIER_LOG is set)
static void wire(Forced& f) {
    tier_hooks_drain(*f.w, f.t.get(), &f.fails);
    f.w->st.set_tier_direct([&f](int dev, uint64_t s, ExpertStore::TierDirect& o) { return f.t->direct_slot(dev, s, o); });
    if (f.t->tracing())
        f.w->st.set_tier_trace([&f](int dev, uint64_t bytes, bool demand) { f.t->note_copy_start(dev, bytes, demand); },
                               [&f](int dev, uint64_t ns) { f.t->note_copy_wait(dev, ns); });
}

// Opens the tier over a new world (its devices forced direct), decides, and fills it (unless `fill` is false: the caller fills).
static bool open_forced(const fs::path& base, const std::string& name, const std::vector<std::pair<int32_t, int32_t>>& ranked,
                        const std::string& pack, const std::string& prof, Forced& f, bool fill = true) {
    f.t = std::make_unique<ExpertTierSet>();
    f.w = std::make_unique<World>(base, name, false);
    f.t->set_direct_test_force(true);
    std::string err;
    if (!f.t->open(f.w->specs, err)) { std::printf("FAIL open %s: %s\n", name.c_str(), err.c_str()); ++fails; return false; }
    wire(f);
    bool reused = true;
    std::string why;
    std::vector<std::string> notes;
    CHECK(f.t->decide(SLOT, 2, 4, NL, NE, pack, prof, false, ranked, reused, why, notes));
    CHECK(!reused);
    if (fill) {
        ExpertTierSet::Fill fl{};
        CHECK(do_fill(*f.w, f.t.get(), fl));
    }
    return true;
}

// The three matrices of (l, e) in its store slot, which must be ready: the file's bytes.
static bool store_ok(World& w, int l, int e) {
    const int s = w.st.find(l, e);
    if (s < 0) return false;
    for (int r = 0; r < 3; ++r)
        if (!same_matrix(w.st.mat(s, r), l, e, r)) return false;
    return true;
}

// Ensures (l, e) through the store (a tier copy, or the file when the tier cannot serve it), checks its bytes, and releases it.
static bool fetch_ok(World& w, int l, int e) {
    const int es[1] = {e};
    std::vector<char> ok;
    const bool got = w.st.ensure(l, es, 1, ok) && ok[0] == 1;
    const bool same = got && store_ok(w, l, e);
    w.st.release(l, es, 1, ok);
    return same;
}

static bool odirect_reads_ok(const fs::path& dir) {   // a file here that takes an O_DIRECT read of one aligned block
#if !defined(__linux__)
    (void) dir;
    return false;   // (O_DIRECT is Linux's flag here: the real-device check is skipped)
#else
    std::error_code ec;
    fs::create_directories(dir, ec);   // (a directory that cannot be made is a skip, not an exception)
    if (ec) return false;
    const std::string p = (dir / "odirect_probe.bin").string();
    {
        std::FILE* f = std::fopen(p.c_str(), "wb");
        if (f == nullptr) return false;
        std::vector<uint8_t> z(MB4, 0);
        std::fwrite(z.data(), 1, z.size(), f);
        std::fclose(f);
    }
    bool ok = false;
    const int fd = ::open(p.c_str(), O_RDONLY | O_DIRECT);
    if (fd >= 0) {
        void* m = nullptr;
        if (posix_memalign(&m, 4096, MB4) == 0) {
            ok = ::pread(fd, m, MB4, 0) == (ssize_t) MB4;
            std::free(m);
        }
        ::close(fd);
    }
    fs::remove(p);
    return ok;
#endif
}

static void direct_checks(const fs::path& base, const std::vector<std::pair<int32_t, int32_t>>& ranked,
                          const std::string& pack, const std::string& prof) {
    const int L = 2, E = 5;   // the expert the existing checks place on the tier
    fs::create_directories(base);
    set_env("DS4_TIER_LOG", "1");       // (the per-device counters these checks read)
    set_env("DS4_TIER_THREADS", "1");   // (one worker a device: the injected reads are consumed in a known order)
    direct_policy_checks(base);
    // the queues' direct reads go through the injected read from here on (restored at the end of this function)
    const os::DirectPreadFn saved_read = os::direct_pread_hook();
    os::direct_pread_hook() = inject_pread;

    // ---- the mode: unset and 0 are the mapped path (no direct descriptor, no mode in the stats); other values are refused ----
    const char* off_values[] = {nullptr, "0"};
    for (size_t i = 0; i < 2; ++i) {
        set_env("DS4_TIER_DIRECT", off_values[i]);
        auto t = std::make_unique<ExpertTierSet>();
        auto w = std::make_unique<World>(base, "direct_off_" + std::to_string(i), false);
        std::string err;
        CHECK(t->open(w->specs, err));
        CHECK(!t->direct_mode() && !t->direct_active());
        for (const ExpertTierSet::DeviceStats& d : t->stats()) CHECK(d.direct_mode.empty());
        ExpertStore::TierDirect o;
        CHECK(!t->direct_slot(0, 0, o) && !t->direct_slot(1, 0, o));
        t->close();
        w.reset();
    }
    const char* bad_values[] = {"2", "yes", "", "01"};
    for (size_t i = 0; i < 4; ++i) {
        set_env("DS4_TIER_DIRECT", bad_values[i]);
        auto t = std::make_unique<ExpertTierSet>();
        auto w = std::make_unique<World>(base, "direct_bad_" + std::to_string(i), false);
        std::string err;
        CHECK(!t->open(w->specs, err));
        CHECK(err.find("DS4_TIER_DIRECT") != std::string::npos);
        CHECK(t->device_count() == 0);   // (refused before any device is touched)
        w.reset();
    }
    set_env("DS4_TIER_DIRECT", "1");

    // ---- a real O_DIRECT read of the tier's slots: skipped where the directory refuses O_DIRECT reads or a device refuses it ----
    {
        const fs::path odir = std::getenv("DS4_TIER_TEST_DIR") ? fs::path(std::getenv("DS4_TIER_TEST_DIR")) : base;
        if (!odirect_reads_ok(odir)) {
            std::printf("SKIP real O_DIRECT tier reads: %s does not take O_DIRECT reads (set DS4_TIER_TEST_DIR)\n", odir.c_str());
            ++skips;
        } else {
            Forced f;
            f.t = std::make_unique<ExpertTierSet>();
            f.w = std::make_unique<World>(odir, "direct_real", false);
            std::string err;
            CHECK(f.t->open(f.w->specs, err));
            if (!f.t->direct_active()) {
                std::printf("SKIP real O_DIRECT tier reads: no device opened a direct descriptor here (see the DS4_TIER_DIRECT lines)\n");
                ++skips;
            } else {
                wire(f);
                bool reused = true;
                std::string why;
                std::vector<std::string> notes;
                CHECK(f.t->decide(SLOT, 2, 4, NL, NE, pack, prof, false, ranked, reused, why, notes) && !reused);
                ExpertTierSet::Fill fl{};
                CHECK(do_fill(*f.w, f.t.get(), fl));
                std::map<std::pair<int, int>, std::pair<int, uint64_t>> homes;
                CHECK(check_bytes(*f.w, f.t.get(), ranked, homes) == 0);   // the placed bytes are the file's (the fill's own path)
                int got = 0;
                for (const auto& x : ranked) got += fetch_ok(*f.w, x.first, x.second) ? 1 : 0;
                CHECK(got == (int) ranked.size());
                const int64_t submitted = f.w->st.stats().tier_direct.load();
                CHECK(submitted == (int64_t) ranked.size());   // every expert was a tier miss, and every one went direct
                uint64_t ok_total = 0;
                for (const ExpertTierSet::DeviceStats& d : f.t->stats()) {
                    CHECK(d.direct_mode == "direct");
                    CHECK(d.direct_failed == 0 && d.direct_fallbacks == 0);
                    CHECK(d.direct_slot_ok_bytes == d.direct_ok * SLOT);
                    CHECK(d.direct_payload_ok_bytes == d.direct_ok * 3 * MB4);
                    CHECK(d.completed_jobs == 0);   // (no mapped copy ran: the copies' counters stay the mapped path's)
                    ok_total += d.direct_ok;
                }
                CHECK(ok_total == (uint64_t) submitted);
                unhook(f);
                f.t->close();
            }
        }
    }

    // ---- the store and the queue on a forced direct tier (any filesystem): the injected outcomes of one direct read ----
    set_env("DS4_TIER_DIRECT", "1");
    struct Fault { const char* name; std::vector<int> plan; bool want_ok; int want_fails; };
    // (plan, outcome) : kOk recovers in place; a device failure refills at the next start and the expert is read from the file
    const std::vector<Fault> faults = {
        {"EINTR then a short aligned read: recovered", {kEintr, kShort4k}, true, 0},
        {"a short unaligned read: device failed", {kShortOdd}, false, 1},
        {"a genuine I/O error: device failed", {kEio}, false, 1},
        {"the device ends early: device failed", {kZero}, false, 1},
    };
    for (const Fault& fc : faults) {
        Forced f;
        CHECK(open_forced(base, std::string("direct_fault_") + std::to_string(&fc - faults.data()), ranked, pack, prof, f));
        int d = -1;
        uint64_t k = 0;
        CHECK(f.t->find(L, E, d, k));
        inject_plan(fc.plan);
        const bool got = fetch_ok(*f.w, L, E);
        CHECK(got == fc.want_ok);
        CHECK(f.fails.load() == fc.want_fails);
        int d2 = -1;
        uint64_t k2 = 0;
        CHECK(f.t->find(L, E, d2, k2) == (fc.want_fails == 0));   // a failed device is not served until it refills
        if (!fc.want_ok) CHECK(fetch_ok(*f.w, L, E));              // the expert comes from the file now (the caller's fallback)
        if (fc.want_fails == 1) {
            const std::vector<ExpertTierSet::DeviceStats> ds = f.t->stats();
            CHECK(ds[(size_t) d].direct_failed == 1 && ds[(size_t) d].direct_ok == 0 && ds[(size_t) d].direct_fallbacks == 0);
        }
        unhook(f);
        f.t->close();
    }
    CHECK(inject_calls() > 0);   // (the injected reads did run: the outcomes above are the direct path's, not a mapped copy's)
    inject_plan({});

    // A refused unbuffered read (EINVAL): the slot is copied the mapped way, the device stops asking, and the device is not failed.
    {
        Forced f;
        CHECK(open_forced(base, "direct_refused", ranked, pack, prof, f));
        int d = -1;
        uint64_t k = 0;
        CHECK(f.t->find(L, E, d, k));
        inject_plan({kShort4k, kEinval});   // (a partial read, then the refusal: the whole slot is copied, not the rest of it)
        CHECK(fetch_ok(*f.w, L, E));
        CHECK(f.fails.load() == 0);
        CHECK(f.t->find(L, E, d, k));       // (the device still serves its experts: mapped now)
        ExpertStore::TierDirect o;
        CHECK(!f.t->direct_slot(d, k, o));  // (the refusal is recorded: no more direct reads from this device)
        {
            const std::vector<ExpertTierSet::DeviceStats> ds = f.t->stats();
            CHECK(ds[(size_t) d].direct_mode == "fallback" && ds[(size_t) d].direct_fallbacks == 1 && ds[(size_t) d].direct_ok == 0);
        }
        // another expert of the same device: a mapped copy, not a direct read
        int other = -1;
        for (const auto& x : ranked) {
            int dx = -1;
            uint64_t kx = 0;
            if (x != std::make_pair((int32_t) L, (int32_t) E) && f.t->find(x.first, x.second, dx, kx) && dx == d) {
                other = 1;
                const int64_t before = f.w->st.stats().tier_direct.load();
                CHECK(fetch_ok(*f.w, x.first, x.second));
                CHECK(f.w->st.stats().tier_direct.load() == before);
                break;
            }
        }
        CHECK(other == 1);   // (the uniform deal puts more than one expert on each device)
        unhook(f);
        f.t->close();
    }
    inject_plan({});

    // A clean direct read: one job a slot (padding included), the matrix bytes counted apart, and no mapped copy ran.
    {
        Forced f;
        CHECK(open_forced(base, "direct_clean", ranked, pack, prof, f));
        int d = -1;
        uint64_t k = 0;
        CHECK(f.t->find(L, E, d, k));
        const uint64_t before_direct = (uint64_t) f.w->st.stats().tier_direct.load();
        CHECK(fetch_ok(*f.w, L, E));
        CHECK(f.w->st.stats().tier_direct.load() == (int64_t) before_direct + 1);
        const std::vector<ExpertTierSet::DeviceStats> ds = f.t->stats();
        const ExpertTierSet::DeviceStats& a = ds[(size_t) d];
        CHECK(a.direct_mode == "direct");
        CHECK(a.direct_jobs == 1 && a.direct_ok == 1 && a.direct_fallbacks == 0 && a.direct_failed == 0);
        CHECK(a.direct_slot_bytes == SLOT && a.direct_slot_ok_bytes == SLOT);
        CHECK(a.direct_payload_ok_bytes == 3 * MB4);                       // the matrices, not the slot
        CHECK(a.started_bytes == 3 * MB4 && a.demand_experts == 1);        // the logical payload, as the mapped path counts it
        CHECK(a.completed_jobs == 0 && a.completed_bytes == 0);            // (no mapped copy: nothing counted as one)
        CHECK(a.direct_service_ns > 0);
        unhook(f);
        f.t->close();
    }

    // ---- DS4_TIER_WAKE: the CPU lane waits on the tier device's queue, and only its own slot's reads end the wait ----
    // Each scenario holds a queue's worker in a gated injected read, so what the waiter does is decided by the test, not by timing:
    // the waiter is on the queue the test names, an unrelated job finishing does not release it, and its own slot's last read does.
    // A waiter that was released by anything else shows as `early` (it began another wait sooner than the safety-net timeout allows)
    // or as `returned` before the gate opens.
    auto nonzero_wait = [&](Forced& f, int d, uint64_t before) {   // tier_note_wait recorded the wait (>= the settle below)
        const std::vector<ExpertTierSet::DeviceStats> ds = f.t->stats();
        return ds[(size_t) d].wait_ns >= before + 25'000'000ull;
    };
    {   // the whole-slot direct read (DS4_TIER_DIRECT), an unrelated completion, and a read that ends as the waiter goes to sleep
        set_env("DS4_TIER_THREADS", "2");   // (one worker is held in the gated read, the other runs an unrelated job)
        set_env("DS4_TIER_WAKE", "1");
        Forced f;
        CHECK(open_forced(base, "direct_wake", ranked, pack, prof, f));
        int d = -1;
        uint64_t k = 0;
        CHECK(f.t->find(L, E, d, k));
        IoQueue* tq = f.t->queue(d);
        CHECK(tq != nullptr);
        SlotWaitObs obs;
        obs.expect = tq;
        SlotWaitScope scope(obs);
        const uint64_t wait0 = f.t->stats()[(size_t) d].wait_ns;

        g_read_gate.close();
        inject_plan({kBlock});
        const int es[1] = {E};
        std::vector<char> ok;
        std::vector<int> sl;
        f.w->st.begin(L, es, 1, ok, sl);   // (the whole-slot read is on the tier queue, its worker held in it)
        CHECK(sl[0] >= 0 && !f.w->st.loaded(sl[0]));
        std::atomic<bool> returned{false};
        std::thread waiter([&] {
            f.w->st.wait_slots(sl, ok, 0, 1);
            returned = true;
        });
        CHECK(obs.await_calls(1, 2000));   // (the caller waits on this device's queue)
        CHECK(obs.queue == tq && obs.left != nullptr && obs.left->load() > 0);

        // An unrelated job on the same queue finishes while the slot's read is still held: it must not end the wait.
        std::atomic<int> other_left{1};
        std::vector<uint8_t> osrc(4096, 7), odst(4096, 0);
        IoQueue::Job other;
        other.mem = osrc.data();
        other.dst = odst.data();
        other.len = osrc.size();
        other.left = &other_left;
        tq->push(&other, 1, false);
        CHECK(wait_below(other_left, 0, 5000));
        std::this_thread::sleep_for(std::chrono::milliseconds(30));   // (the waiter, if the completion woke it, has run by now)
        CHECK(!returned.load());
        CHECK(obs.early == 0);

        g_read_gate.release();   // the slot's own read finishes
        waiter.join();
        CHECK(returned.load());
        CHECK(ok[0] == 1 && store_ok(*f.w, L, E));
        CHECK(obs.early == 0 && obs.wrong_queue == 0);
        CHECK(nonzero_wait(f, d, wait0));

        // A read that is done, and not yet announced, as the waiter goes to sleep: the count says so, and the wait must not
        // wait for the announcement (the hook runs under the completion mutex, which that announcement needs).
        int l2 = -1, e2 = -1, d2 = -1;
        uint64_t k2 = 0;
        for (const auto& x : ranked) {
            if (x.first == L && x.second == E) continue;
            if (f.t->find(x.first, x.second, d2, k2)) {
                l2 = x.first;
                e2 = x.second;
                break;
            }
        }
        CHECK(l2 >= 0 && e2 >= 0 && d2 >= 0);
        g_read_gate.close();
        inject_plan({kBlock});
        std::vector<char> ok2;
        std::vector<int> sl2;
        f.w->st.begin(l2, &e2, 1, ok2, sl2);
        CHECK(sl2[0] >= 0 && !f.w->st.loaded(sl2[0]));
        obs.act = [&](const std::atomic<int>* left) {
            g_read_gate.release();
            const auto t0 = test_clock::now();
            while (left->load(std::memory_order_acquire) != 0 && test_clock::now() - t0 < std::chrono::seconds(5))
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        };
        obs.expect = f.t->queue(d2);
        std::atomic<bool> returned2{false};
        std::thread waiter2([&] {
            f.w->st.wait_slots(sl2, ok2, 0, 1);
            returned2 = true;
        });
        CHECK(wait_flag(returned2, 2000));
        waiter2.join();
        CHECK(ok2[0] == 1 && store_ok(*f.w, l2, e2));
        CHECK(obs.wrong_queue == 0);
        obs.act = nullptr;

        inject_plan({});
        set_env("DS4_TIER_WAKE", nullptr);
        set_env("DS4_TIER_THREADS", "1");
        unhook(f);
        f.t->close();
    }

    // The default (mapped) path on the real store: the three copies of a tier hit are queued behind the held worker.  With the
    // switch on the waiter sleeps on the tier queue; with it off it never touches that queue (the store queue's poll, as before).
    for (int wake = 1; wake >= 0; --wake) {
        set_env("DS4_TIER_THREADS", "1");
        set_env("DS4_TIER_WAKE", wake ? "1" : nullptr);
        Forced f;
        CHECK(open_forced(base, wake ? "wake_mapped_on" : "wake_mapped_off", ranked, pack, prof, f));
        f.w->st.set_tier_direct(nullptr);   // (every hit is the three mapped copies, as without DS4_TIER_DIRECT)
        int d = -1;
        uint64_t k = 0;
        CHECK(f.t->find(L, E, d, k));
        IoQueue* tq = f.t->queue(d);
        CHECK(tq != nullptr);
        SlotWaitObs obs;
        obs.expect = tq;
        SlotWaitScope scope(obs);
        const uint64_t wait0 = f.t->stats()[(size_t) d].wait_ns;

        WorkerHold hold(f.w->shard);
        hold.hold(tq);   // (the queue's only worker is in the gate)

        const int es[1] = {E};
        std::vector<char> ok;
        std::vector<int> sl;
        f.w->st.begin(L, es, 1, ok, sl);   // (three copy jobs behind the held read)
        CHECK(sl[0] >= 0 && !f.w->st.loaded(sl[0]));
        std::atomic<bool> returned{false};
        std::thread waiter([&] {
            f.w->st.wait_slots(sl, ok, 0, 1);
            returned = true;
        });
        if (wake) CHECK(obs.await_calls(1, 2000));   // (on the tier queue)
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        CHECK(!returned.load());   // (the copies cannot run: the one worker is held)
        CHECK(obs.early == 0 && obs.wrong_queue == 0);
        if (wake) CHECK(obs.queue == tq && obs.calls_now() >= 1);
        else CHECK(obs.calls_now() == 0);   // (off: the store's queue; the tier queue was never waited on)
        g_read_gate.release();
        waiter.join();
        CHECK(returned.load());
        CHECK(ok[0] == 1 && store_ok(*f.w, L, E));
        CHECK(hold.done(5000));
        CHECK(obs.early == 0 && obs.wrong_queue == 0);
        CHECK(nonzero_wait(f, d, wait0));   // (the accounting is the same on or off)
        {
            const ExpertTierSet::DeviceStats a = f.t->stats()[(size_t) d];
            CHECK(a.completed_jobs == 3 && a.demand_jobs == 3);   // (three mapped copies, no direct slot read of the expert's)
        }
        inject_plan({});
        set_env("DS4_TIER_WAKE", nullptr);
        unhook(f);
        f.t->close();
    }

    // preload() waits for each slot it started, on its own: the tier queue with the switch on, the store queue (never the tier's)
    // with it off.  Three experts of one device, their copies behind the held worker; preload returns when all are ready.
    for (int wake = 1; wake >= 0; --wake) {
        set_env("DS4_TIER_THREADS", "1");
        set_env("DS4_TIER_WAKE", wake ? "1" : nullptr);
        Forced f;
        CHECK(open_forced(base, wake ? "wake_preload_on" : "wake_preload_off", ranked, pack, prof, f));
        f.w->st.set_tier_direct(nullptr);
        int d = -1;
        uint64_t k = 0;
        CHECK(f.t->find(L, E, d, k));
        IoQueue* tq = f.t->queue(d);
        CHECK(tq != nullptr);
        std::vector<std::pair<int, int>> list;
        for (const auto& x : ranked) {
            int dd = -1;
            uint64_t kk = 0;
            if (f.t->find(x.first, x.second, dd, kk) && dd == d) list.push_back({x.first, x.second});
            if (list.size() == 3) break;
        }
        CHECK(list.size() == 3);
        SlotWaitObs obs;
        obs.expect = tq;
        SlotWaitScope scope(obs);
        WorkerHold hold(f.w->shard);
        hold.hold(tq);
        std::atomic<bool> returned{false};
        size_t got = 0;
        std::thread t([&] {
            got = f.w->st.preload(list, 0);
            returned = true;
        });
        if (wake) CHECK(obs.await_calls(1, 2000));
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        CHECK(!returned.load());   // (the copies cannot run: the one worker is held)
        if (wake) CHECK(obs.queue == tq);
        else CHECK(obs.calls_now() == 0);
        g_read_gate.release();
        t.join();
        CHECK(returned.load() && got == 3);
        for (const auto& x : list) CHECK(store_ok(*f.w, x.first, x.second));   // (found, so ready, with the file's bytes)
        CHECK(f.w->st.stats().tier_hits.load() == 3 && f.w->st.stats().bytes_read.load() == 0);
        CHECK(obs.wrong_queue == 0);   // (not `early`: preload waits slot after slot, so its waits legitimately follow each other)
        CHECK(hold.done(5000));
        inject_plan({});
        set_env("DS4_TIER_WAKE", nullptr);
        unhook(f);
        f.t->close();
    }

    // A slot the GPU put in (put_begin) has no tier device: its waiter keeps the store's queue, and put_done ends the wait.
    {
        set_env("DS4_TIER_THREADS", "1");
        set_env("DS4_TIER_WAKE", "1");
        Forced f;
        CHECK(open_forced(base, "wake_put", ranked, pack, prof, f));
        int d = -1;
        uint64_t k = 0;
        CHECK(f.t->find(L, E, d, k));   // (the tier holds this expert: a lookup by the expert would pick its queue)
        SlotWaitObs obs;
        SlotWaitScope scope(obs);
        uint8_t* blob = f.w->st.put_begin(L, E);
        CHECK(blob != nullptr);
        const int es[1] = {E};
        std::vector<char> ok;
        std::vector<int> sl;
        f.w->st.begin(L, es, 1, ok, sl);   // (the slot, loading: left 1)
        CHECK(sl[0] >= 0 && !f.w->st.loaded(sl[0]));
        std::atomic<bool> returned{false};
        std::thread waiter([&] {
            f.w->st.wait_slots(sl, ok, 0, 1);
            returned = true;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        CHECK(!returned.load());
        for (int r = 0; r < 3; ++r)   // (the caller fills the blob itself: the expert's bytes)
            for (uint64_t i = 0; i < (uint64_t) MB4; ++i) blob[(size_t) r * (MB4 + 8192) + i] = shard_byte(src_pos(L, E, r, i));
        f.w->st.put_done(L, E, true);
        CHECK(wait_flag(returned, 2000));   // (within the store queue's poll: a hang is a failure, not a stuck test)
        waiter.join();
        CHECK(ok[0] == 1 && store_ok(*f.w, L, E));
        CHECK(obs.calls_now() == 0);   // (never on a tier queue)
        set_env("DS4_TIER_WAKE", nullptr);
        unhook(f);
        f.t->close();
    }

    // The arena is not inherited by a fork once a device is direct (MADV_DONTFORK: kernel flag "dc" on its mapping).
#if defined(__linux__)
    {
        Forced f;
        CHECK(open_forced(base, "direct_fork", ranked, pack, prof, f));
        CHECK(f.t->direct_active());
        const uintptr_t arena = (uintptr_t) (f.w->st.blob(0) - ExpertStore::kSlack);
        CHECK(!vma_has_flag(arena, "dc"));   // (before: an ordinary private mapping)
        CHECK(f.w->st.no_fork_arena());
        CHECK(vma_has_flag(arena, "dc"));
        unhook(f);
        f.t->close();
    }
#endif

    // Canaries around the destination: the whole-slot read writes its own slot and nothing beside it.
    {
        Forced f;
        CHECK(open_forced(base, "direct_canary", ranked, pack, prof, f));
        for (int s = 0; s < (int) f.w->st.committed_slots(); ++s) {
            uint8_t* b = const_cast<uint8_t*>(f.w->st.blob(s)) - ExpertStore::kSlack;
            std::memset(b, 0xCC, SLOT);
        }
        const int es[1] = {E};
        std::vector<char> ok;
        CHECK(f.w->st.ensure(L, es, 1, ok) && ok[0] == 1);
        const int s = f.w->st.find(L, E);
        CHECK(s >= 0 && store_ok(*f.w, L, E));
        auto canary = [&](int slot) {
            const uint8_t* b = f.w->st.blob(slot) - ExpertStore::kSlack;
            for (uint64_t i = 0; i < SLOT; ++i)
                if (b[i] != 0xCC) return false;
            return true;
        };
        if (s > 0) CHECK(canary(s - 1));
        if (s + 1 < (int) f.w->st.committed_slots()) CHECK(canary(s + 1));
        f.w->st.release(L, es, 1, ok);
        unhook(f);
        f.t->close();
    }

    // The slot is not findable while its read is in flight, and a cancelled guess does not fail the device.
    {
        Forced f;
        CHECK(open_forced(base, "direct_inflight", ranked, pack, prof, f));
        int d = -1;
        uint64_t k = 0;
        CHECK(f.t->find(L, E, d, k));
        BigCopy big;   // holds the device's one worker while the guess waits behind it
        big.push(f.t->queue(d));
        f.w->st.prefetch(L, E);
        CHECK(f.w->st.find(L, E) < 0);          // (in flight: not a slot anyone can read)
        CHECK(f.w->st.shrink_chunks(1) == 0);   // (a chunk with a read in flight is not given back)
        f.w->st.cancel_stale(L);                // the layer moved past: the queued guess is dropped unread
        const int es[1] = {E};
        std::vector<char> ok;
        CHECK(!f.w->st.ensure(L, es, 1, ok) || ok[0] == 0);
        while (big.left.load() > 0) f.t->queue(d)->wait_any(1000);
        CHECK(f.fails.load() == 0);             // a cancelled copy says nothing about the device (§9)
        CHECK(f.t->find(L, E, d, k));
        ExpertStore::TierDirect o;
        CHECK(f.t->direct_slot(d, k, o));
        CHECK(f.w->st.stats().cancelled.load() >= 1);
        CHECK(fetch_ok(*f.w, L, E));            // then read directly as a demand read
        unhook(f);
        f.t->close();
    }

    // Shutdown with direct jobs queued behind a long copy: the queue finishes them before the arena goes.
    {
        Forced f;
        CHECK(open_forced(base, "direct_shutdown", ranked, pack, prof, f));
        int d = -1;
        uint64_t k = 0;
        CHECK(f.t->find(L, E, d, k));
        std::vector<std::pair<int, int>> same_dev;
        for (const auto& x : ranked) {
            int dx = -1;
            uint64_t kx = 0;
            if (f.t->find(x.first, x.second, dx, kx) && dx == d && same_dev.size() < 2) same_dev.push_back({x.first, x.second});
        }
        CHECK(same_dev.size() == 2);
        BigCopy big;
        big.push(f.t->queue(d));
        for (const auto& x : same_dev) f.w->st.prefetch(x.first, x.second);
        f.w.reset();   // the store goes: its destructor drains the tier's queues first, then releases the arena
        CHECK(big.left.load() == 0);
        {
            const std::vector<ExpertTierSet::DeviceStats> ds = f.t->stats();
            CHECK(ds[(size_t) d].direct_ok == same_dev.size());   // both queued direct reads finished, into live memory
            CHECK(ds[(size_t) d].direct_failed == 0);
        }
        f.t->close();
    }

    // Concurrency: speculative and demand reads on both devices, two workers each.
    set_env("DS4_TIER_THREADS", "2");
    {
        Forced f;
        CHECK(open_forced(base, "direct_concurrent", ranked, pack, prof, f));
        int round_ok = 0;
        for (int round = 0; round < 2; ++round) {   // (round 2 brings six experts the store has not seen: fresh tier reads, evictions)
            const size_t base_i = (size_t) round * 6;
            for (size_t i = base_i; i < base_i + 6; ++i) f.w->st.prefetch(ranked[i].first, ranked[i].second);
            for (size_t i = base_i; i < base_i + 6; ++i) round_ok += fetch_ok(*f.w, ranked[i].first, ranked[i].second) ? 1 : 0;
        }
        CHECK(round_ok == 12);
        CHECK(f.fails.load() == 0);
        uint64_t ok_total = 0, failed = 0;
        for (const ExpertTierSet::DeviceStats& d : f.t->stats()) {
            ok_total += d.direct_ok;
            failed += d.direct_failed + d.direct_fallbacks;
        }
        CHECK(failed == 0);
        CHECK(ok_total == (uint64_t) f.w->st.stats().tier_direct.load());
        unhook(f);
        f.t->close();
    }
    set_env("DS4_TIER_THREADS", "1");

    // The tier's slots, by device: the last slot of each device is the last whole slot, and nothing past it is offered.
    {
        Forced f;
        CHECK(open_forced(base, "direct_bounds", ranked, pack, prof, f));
        for (size_t dv = 0; dv < f.t->device_count(); ++dv) {
            const uint64_t size = f.t->spec_at(dv).mib << 20;
            const uint64_t n = (size - f.t->data_off()) / SLOT;
            ExpertStore::TierDirect o;
            CHECK(n > 0 && f.t->direct_slot((int) dv, n - 1, o));
            CHECK(o.off == f.t->data_off() + (n - 1) * SLOT && o.len == SLOT && o.off + o.len <= size);
            CHECK(!f.t->direct_slot((int) dv, n, o));
            CHECK(f.t->slot_base((int) dv, 1) - f.t->slot_base((int) dv, 0) == (ptrdiff_t) SLOT);
        }
        ExpertStore::TierDirect none;
        CHECK(!f.t->direct_slot(-1, 0, none));   // (no such device)
        unhook(f);
        f.t->close();
    }

    // A value that is not a slot of this device is refused, and a closed set serves no direct read.
    {
        Forced f;
        CHECK(open_forced(base, "direct_closed", ranked, pack, prof, f));
        ExpertStore::TierDirect o;
        CHECK(!f.t->direct_slot(7, 0, o));
        unhook(f);
        f.t->close();
        CHECK(!f.t->direct_slot(0, 0, o));
    }

    // Fill and read visibility: no slot of a tier that is still filling answers, and the first answer comes after the fill.
    {
        Forced f;
        CHECK(open_forced(base, "direct_visible", ranked, pack, prof, f, false));
        bool early = false;
        ExpertTierSet::Fill fl{};
        std::string err;
        CHECK(f.t->fill(
            [&](int l, int e, uint8_t* sb, int dev, std::atomic<int>& left, std::atomic<bool>& bad, uint64_t& bytes) {
                int d2 = -1;
                uint64_t k2 = 0;
                if (f.t->find(l, e, d2, k2)) early = true;   // (must not happen: the fill has not published)
                return f.w->st.tier_fill_read(l, e, sb, dev, left, bad, bytes);
            },
            fl, err));
        CHECK(!early);
        CHECK(f.t->readable());
        int d = -1;
        uint64_t k = 0;
        CHECK(f.t->find(L, E, d, k));
        CHECK(fetch_ok(*f.w, L, E));
        unhook(f);
        f.t->close();
    }

    // Persistence: a tier filled with the mode off is reused with it on, and the other way round, with no refill.
    {
        Forced f;
        // (1) filled with the mode off (the mapped path's fill is the same bytes either way)
        set_env("DS4_TIER_DIRECT", nullptr);
        f.t = std::make_unique<ExpertTierSet>();
        f.w = std::make_unique<World>(base, "direct_persist", false);
        std::string err;
        CHECK(f.t->open(f.w->specs, err));
        wire(f);
        bool reused = true;
        std::string why;
        std::vector<std::string> notes;
        CHECK(f.t->decide(SLOT, 2, 4, NL, NE, pack, prof, false, ranked, reused, why, notes) && !reused);
        ExpertTierSet::Fill fl{};
        CHECK(do_fill(*f.w, f.t.get(), fl));
        unhook(f);
        f.t->close();
        // (2) reused with the mode on (forced direct): nothing copied, and the reads go direct
        set_env("DS4_TIER_DIRECT", "1");
        f.t = std::make_unique<ExpertTierSet>();
        f.t->set_direct_test_force(true);
        CHECK(f.t->open(f.w->specs, err));
        wire(f);
        CHECK(f.t->decide(SLOT, 2, 4, NL, NE, pack, prof, false, ranked, reused, why, notes) && reused);
        CHECK(fetch_ok(*f.w, L, E));
        {
            int d = -1;
            uint64_t k = 0;
            CHECK(f.t->find(L, E, d, k));
            CHECK(f.t->stats()[(size_t) d].direct_ok == 1);
        }
        unhook(f);
        f.t->close();
        // (3) reused with the mode off again: mapped copies, the same bytes
        set_env("DS4_TIER_DIRECT", nullptr);
        f.t = std::make_unique<ExpertTierSet>();
        CHECK(f.t->open(f.w->specs, err));
        wire(f);
        const int64_t direct_before = f.w->st.stats().tier_direct.load();
        CHECK(f.t->decide(SLOT, 2, 4, NL, NE, pack, prof, false, ranked, reused, why, notes) && reused);
        CHECK(fetch_ok(*f.w, L, E));
        CHECK(f.w->st.stats().tier_direct.load() == direct_before);
        unhook(f);
        f.t->close();
        set_env("DS4_TIER_DIRECT", "1");
    }

    os::direct_pread_hook() = saved_read;
    set_env("DS4_TIER_DIRECT", nullptr);
    set_env("DS4_TIER_LOG", nullptr);
    set_env("DS4_TIER_THREADS", nullptr);
}
#endif

int main() {
    const fs::path base = fs::temp_directory_path() /
                          ("ds4_tier_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const std::string packA = "sha:" + strata::core::tier_sha256_hex((const uint8_t*) "packA", 5);
    const std::string packB = "sha:" + strata::core::tier_sha256_hex((const uint8_t*) "packB", 5);
    const std::string profA = "sha:" + strata::core::tier_sha256_hex((const uint8_t*) "profileA", 8);
    const std::string profB = "sha:" + strata::core::tier_sha256_hex((const uint8_t*) "profileB", 8);
    // The planner's list: every expert in index order (the profile's heat would sort these first), the guards (vram 2 +
    // store 4) cutting the first six: (1,0) .. (2,5) remain, and the uniform deal alternates devices (§4.2).
    std::vector<std::pair<int32_t, int32_t>> ranked;
    for (int l = 0; l < NL; ++l)
        for (int e = 0; e < NE; ++e) ranked.push_back({l, e});
    ranked.erase(ranked.begin(), ranked.begin() + 6);
    const std::vector<std::pair<int32_t, int32_t>> ranked_shuffled(ranked.rbegin(), ranked.rend());
    std::vector<std::string> notes;
    std::string why;

    // ---- the core case: fill, bytes, tier-served store reads, reuse across "starts" ----
    {
        World w(base, "core", false);
        auto t = std::make_unique<ExpertTierSet>();
        std::string err;
        CHECK(t->open(w.specs, err));
        tier_hooks(w, t);
        bool reused = false;
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));   // (first start: the devices hold nothing yet)
        CHECK(!reused);
        ExpertTierSet::Fill f{};
        CHECK(do_fill(w, t.get(), f));
        CHECK(f.placed == (int64_t) ranked.size());
        CHECK(f.per_dev.size() == 2 && f.per_dev[0] == 6 && f.per_dev[1] == 6);   // (§9's per-device progress: the deal alternates)
        CHECK(t->readable() && t->experts_held() == (int64_t) ranked.size());
        std::map<std::pair<int, int>, std::pair<int, uint64_t>> homes;
        CHECK(check_bytes(w, t.get(), ranked, homes) == 0);
        // §8.1: the store's own hooks over the same set (the set_tier above serves them too)
        const std::vector<std::pair<int, int>> one = {{2, 5}};   // (a macro cannot take braced commas: hoisted)
        CHECK(w.st.preload(one, 0) == 1);
        const int s2 = w.st.find(2, 5);
        CHECK(s2 >= 0);
        for (int r = 0; r < 3; ++r) CHECK(same_matrix(w.st.mat(s2, r), 2, 5, r));   // (acceptance 2 shape: served bytes identical)
        CHECK(w.st.stats().tier_hits.load() == 1);
        CHECK(w.st.stats().bytes_read.load() == 0);   // (§8.2: a tier copy adds no file bytes)
        // Acceptance 5: the next start reuses the placement and copies nothing; §7.2: a changed profile sha and a
        // reshuffled ranked list change nothing.
        t->close();
        CHECK(t->open(w.specs, err));
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profB, false, ranked_shuffled, reused, why, notes));
        CHECK(reused);
        {
            int dev = -1;
            uint64_t k = 0;
            int same = 0;
            for (const auto& x : ranked) {
                if (!t->find(x.first, x.second, dev, k)) continue;
                const std::pair<int, uint64_t> now{dev, k};
                if (homes[{x.first, x.second}] == now) ++same;
            }
            CHECK(same == (int) ranked.size());
        }
        // §7.4: a start whose guard differs refills (the planned ranks would be wrong); a fill restores fill_state 2
        CHECK(t->decide(SLOT, 2, 5, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(!reused && why.find("guards") != std::string::npos);
        CHECK(do_fill(w, t.get(), f));
        CHECK(f.placed == (int64_t) ranked.size());
        // A torn header A: its twin B is read (§7.4 step 1); the guard stored above (2/5) still matches
        poke(w.devA, 40);
        CHECK(t->open(w.specs, err));
        CHECK(t->decide(SLOT, 2, 5, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(reused);
        // Both twins torn: no header answers; the whole tier refills (criterion 7's corrupt-header half)
        poke(w.devA, 40 + (64u << 10));
        CHECK(t->open(w.specs, err));
        CHECK(t->decide(SLOT, 2, 5, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(!reused && why.find("header_crc") != std::string::npos);
        // (the refill wrote fill_state = 1: a start after it refills too - the abandoned/torn fill rule, §7.4 step 4)
        CHECK(t->open(w.specs, err));
        CHECK(t->decide(SLOT, 2, 5, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(!reused && why.find("fill_state") != std::string::npos);
    }

    // ---- pack identity and the placement table (§7.1, §7.4: either mismatch refills) ----
    {
        World w(base, "pack", false);
        auto t = std::make_unique<ExpertTierSet>();
        std::string err;
        CHECK(t->open(w.specs, err));
        tier_hooks(w, t);
        bool reused = true;
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(!reused);
        ExpertTierSet::Fill f{};
        CHECK(do_fill(w, t.get(), f));
        // A different pack: refill (criterion 7, pack half)
        CHECK(t->open(w.specs, err));
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packB, profA, false, ranked, reused, why, notes));
        CHECK(!reused && why.find("different pack") != std::string::npos);
        CHECK(do_fill(w, t.get(), f));   // (fill_state 2 again, packB stored)
        // A byte of the placement table no longer matches its crc: refill
        poke(w.devB, 2u * 64u * 1024u + 16u);   // (the table sits at 2 x the header area on every device, §6.3)
        CHECK(t->open(w.specs, err));
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packB, profA, false, ranked, reused, why, notes));
        CHECK(!reused && why.find("placement_crc") != std::string::npos);
        // An abandoned fill: decide planned it, the start died before the copies: fill_state 1 answers
        CHECK(do_fill(w, t.get(), f));
        CHECK(t->open(w.specs, err));
        CHECK(t->decide(SLOT, 2, 3, NL, NE, packB, profA, false, ranked, reused, why, notes));   // (a guard change plans a new deal: headers now say 1)
        CHECK(!reused);
        CHECK(t->open(w.specs, err));
        CHECK(t->decide(SLOT, 2, 3, NL, NE, packB, profA, false, ranked, reused, why, notes));
        CHECK(!reused && why.find("fill_state") != std::string::npos);
    }

    // ---- verify=full catches a data byte that no longer matches its chunk crc (§6.4 of the shared plan) ----
    {
        World w(base, "full", true);
        auto t = std::make_unique<ExpertTierSet>();
        std::string err;
        CHECK(t->open(w.specs, err));
        tier_hooks(w, t);
        bool reused = true;
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(!reused);
        ExpertTierSet::Fill f{};
        CHECK(do_fill(w, t.get(), f));
        int dev = -1;
        uint64_t k = 0;
        CHECK(t->find(1, 0, dev, k));
        CHECK(dev == 0);   // (§4.2: rank 0 of the uniform deal lands on device 0)
        const uint64_t off = t->data_off() + k * SLOT + 8192 + 123;   // (a byte inside that expert's blob)
        CHECK(t->open(w.specs, err));
        poke(w.devA, (size_t) off);
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(!reused && why.find("does not match its crc") != std::string::npos);
    }

    // ---- verify=header (the default): the same corruption is read on trust: the reuse still answers (§11) ----
    {
        World w(base, "hdr", false);
        auto t = std::make_unique<ExpertTierSet>();
        std::string err;
        CHECK(t->open(w.specs, err));
        tier_hooks(w, t);
        bool reused = true;
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(!reused);
        ExpertTierSet::Fill f{};
        CHECK(do_fill(w, t.get(), f));
        int dev = -1;
        uint64_t k = 0;
        CHECK(t->find(1, 0, dev, k));
        const uint64_t off = t->data_off() + k * SLOT + 8192 + 123;
        CHECK(t->open(w.specs, err));
        poke(w.devA, (size_t) off);
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(reused);   // (headers only: a read-time mismatch is what note_refill marks - the case below)
    }

    // ---- a failed copy marks its device: its experts fall back to the files, and the tier refills next start (§9) ----
    {
        World w(base, "mark", false);
        auto t = std::make_unique<ExpertTierSet>();
        std::string err;
        CHECK(t->open(w.specs, err));
        tier_hooks(w, t);
        bool reused = true;
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        ExpertTierSet::Fill f{};
        CHECK(do_fill(w, t.get(), f));
        CHECK(t->open(w.specs, err));
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(reused);
        int dev = -1;
        uint64_t k = 0;
        CHECK(t->find(1, 0, dev, k));
        CHECK(dev == 0);
        t->note_refill(0);   // (what the store's tier_fail hook calls when a copy out of a device fails)
        CHECK(!t->find(1, 0, dev, k));   // (the marked device's experts fall back to the files)
        int dev2 = -1;
        uint64_t k2 = 0;
        CHECK(t->find(2, 5, dev2, k2));
        CHECK(dev2 == 1);   // (the other device still answers)
        t->close();   // (the mark flips both of its headers to fill_state 1: the next start writes the tier again)
        CHECK(t->open(w.specs, err));
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(!reused && why.find("fill_state") != std::string::npos);
    }

    // ---- the queue tells a job that was DROPPED from one that FAILED (a cancelled guess says nothing about a device) ----
    {
        strata::ds4::IoQueue q(1);
        std::atomic<int> l1{1}, l2{1};
        std::atomic<bool> bad1{false}, fail1{false}, bad2{false}, fail2{false}, cancel{true};
        std::vector<uint8_t> src(4096, 1), dst(4096, 0);
        strata::ds4::IoQueue::Job dropped;   // cancelled before it runs: bad, never fail
        dropped.mem = src.data(); dropped.dst = dst.data(); dropped.len = 4096;
        dropped.left = &l1; dropped.bad = &bad1; dropped.fail = &fail1; dropped.cancel = &cancel;
        q.push(&dropped, 1, false);
        strata::ds4::IoQueue::Job broken;    // a read of a descriptor that is not there: ran and failed -> bad and fail
        broken.fd = os::kBadFd; broken.dfd = os::kBadFd; broken.off = 0; broken.len = 4096; broken.dst = dst.data();
        broken.left = &l2; broken.bad = &bad2; broken.fail = &fail2;
        q.push(&broken, 1, false);
        while (l1.load() > 0 || l2.load() > 0) q.wait_any(1000);
        CHECK(bad1.load() && !fail1.load());
        CHECK(bad2.load() && fail2.load());
    }

    // ---- a cancelled tier prefetch does not mark the device (only a failed copy does) ----
    {
        set_env("DS4_TIER_THREADS", "1");   // (one worker per device queue: a long copy ahead holds the guess in the queue)
        auto t = std::make_unique<ExpertTierSet>();   // (declared before the world: destroyed after the store, as in Model::Impl)
        auto w = std::make_unique<World>(base, "cancel", false);
        std::string err;
        CHECK(t->open(w->specs, err));
        std::atomic<int> fail_calls{0};
        tier_hooks_drain(*w, t.get(), &fail_calls);
        bool reused = true;
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        ExpertTierSet::Fill f{};
        CHECK(do_fill(*w, t.get(), f));
        int dev = -1;
        uint64_t k = 0;
        CHECK(t->find(1, 0, dev, k));
        BigCopy big;
        big.push(t->queue(dev));     // the queue is busy for a while ...
        w->st.prefetch(1, 0);        // ... the guess waits behind it ...
        w->st.cancel_stale(2);       // ... and the token is past its layer: dropped unread
        while (big.left.load() > 0) t->queue(dev)->wait_any(1000);
        const int es[1] = {0};
        std::vector<char> ok;
        std::vector<int> sl;
        w->st.begin(1, es, 1, ok, sl);        // (the demand finds the slot loading: waits for the drop, then finishes it)
        const bool all = w->st.wait_slots(sl, ok, 0, 1);
        CHECK(!all);                           // (its read was dropped: the caller reads this one another way)
        CHECK(w->st.stats().cancelled.load() == 1);
        CHECK(w->st.stats().fails.load() == 0);
        CHECK(fail_calls.load() == 0);         // (nothing went wrong on the device)
        CHECK(t->find(1, 0, dev, k));          // (and it is still the tier's)
        t->close();
        CHECK(t->open(w->specs, err));
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(reused);                         // (no mark was written: the next start reuses it)
        w->st.set_tier(nullptr, nullptr, nullptr, nullptr);   // (the world outlives this set's close: no hook may reach it)
        t->close();
        w.reset();
        set_env("DS4_TIER_THREADS", nullptr);
    }

    // ---- a failed fill read marks every device and leaves nothing running (the next start writes the tier again) ----
    {
        auto t = std::make_unique<ExpertTierSet>();
        auto w = std::make_unique<World>(base, "failfill", false);
        std::string err;
        CHECK(t->open(w->specs, err));
        tier_hooks_drain(*w, t.get(), nullptr);
        bool reused = true;
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        ExpertTierSet::Fill f{};
        fs::resize_file(w->shard, 0);   // (the model file is gone under the store: every fill read comes back short and fails)
        std::string ferr;
        const bool ok = t->fill([&](int l, int e, uint8_t* sb, int dev, std::atomic<int>& left, std::atomic<bool>& bad, uint64_t& bytes) {
                                    return w->st.tier_fill_read(l, e, sb, dev, left, bad, bytes);
                                }, f, ferr);
        CHECK(!ok);
        CHECK(!t->readable());
        t->close();                      // (the failed fill marked the devices: fill_state 1 on both headers)
        CHECK(t->open(w->specs, err));
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(!reused && why.find("fill_state") != std::string::npos);
        w->st.set_tier(nullptr, nullptr, nullptr, nullptr);
        t->close();
        w.reset();
    }

    // ---- shutdown with queued and running copies: the tier's queues finish them before the arena goes ----
    for (int round = 0; round < 12; ++round) {
        auto t = std::make_unique<ExpertTierSet>();
        auto w = std::make_unique<World>(base, "shutdown" + std::to_string(round % 3), false);
        std::string err;
        CHECK(t->open(w->specs, err));
        tier_hooks_drain(*w, t.get(), nullptr);
        bool reused = true;
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        ExpertTierSet::Fill f{};
        CHECK(do_fill(*w, t.get(), f));
        BigCopy big0(64ull << 20), big1(64ull << 20);   // (one destination each: two queues never write the same bytes)
        big0.push(t->queue(0));
        big1.push(t->queue(1));
        for (const auto& x : ranked) w->st.prefetch(x.first, x.second);   // (guesses queued behind the copies, on both devices)
        const int es[2] = {0, 1};
        std::vector<char> ok;
        std::vector<int> sl;
        w->st.begin(1, es, 2, ok, sl);   // (and demand copies, pinned, not waited for)
        w->st.cancel_stale(1);
        w.reset();                       // the store goes while the queues hold work: its drain hook runs them to the end first
        CHECK(big0.left.load() == 0 && big1.left.load() == 0);   // (the long copies finished before the store returned)
        CHECK(t->queue(0) == nullptr && t->queue(1) == nullptr);   // (drained: the tier hands out no queue any more)
        t->close();
    }

    // ---- devices in another order, and devices carrying different tables, refill (§6.3) ----
    {
        World w(base, "order", false);
        w.specs = {{w.devA, 12, 0, false}, {w.devB, 12, 0, false}};
        auto t = std::make_unique<ExpertTierSet>();
        std::string err;
        CHECK(t->open(w.specs, err));
        tier_hooks(w, t);
        bool reused = true;
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        ExpertTierSet::Fill f{};
        CHECK(do_fill(w, t.get(), f));
        CHECK(t->open(w.specs, err));
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(reused);
        // the same two files, named in the other order: the headers say device 1 and 0, the placement would point at the wrong disks
        const std::vector<TierSpec> swapped = {w.specs[1], w.specs[0]};
        CHECK(t->open(swapped, err));
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(!reused && why.find("number") != std::string::npos);
        CHECK(!t->readable());
        // a refill in the new order is a valid tier of its own
        CHECK(do_fill(w, t.get(), f));
        CHECK(t->open(swapped, err));
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(reused);
        // device B now carries a table that is self-consistent (its crc is right) but is not device A's
        t->close();
        {
            std::FILE* fb = std::fopen(w.devB.c_str(), "r+b");
            uint8_t hb[kTierHeaderBytesForTest];
            CHECK(std::fread(hb, 1, sizeof hb, fb) == sizeof hb);
            strata::ds4::TierHdr h{};
            std::string hw;
            CHECK(strata::ds4::ds4_tier::th_decode(hb, h, hw));
            const int eb = strata::core::tier_placement_entry_bytes(NL, NE, 2);
            std::vector<uint8_t> tbl((size_t) (h.n_held * (uint64_t) eb));
            std::fseek(fb, (long) (2u * 64u * 1024u), SEEK_SET);
            CHECK(std::fread(tbl.data(), 1, tbl.size(), fb) == tbl.size());
            tbl[eb + 0] ^= 0x01;   // (entry 1's layer/expert word: a different, still decodable, entry)
            h.placement_crc = strata::core::tier_crc32c(tbl.data(), tbl.size());
            std::fseek(fb, (long) (2u * 64u * 1024u), SEEK_SET);
            std::fwrite(tbl.data(), 1, tbl.size(), fb);
            uint8_t nb[kTierHeaderBytesForTest];
            strata::ds4::ds4_tier::th_encode(h, nb);
            std::fseek(fb, 0, SEEK_SET);
            std::fwrite(nb, 1, sizeof nb, fb);
            std::fseek(fb, (long) (64u * 1024u), SEEK_SET);
            std::fwrite(nb, 1, sizeof nb, fb);
            std::fclose(fb);
        }
        CHECK(t->open(swapped, err));
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(!reused && why.find("same placement table") != std::string::npos);
        t->close();
    }


    // ---- §7.1: the pack's identity looks at the shards where the loader opens them (model_dir/name), from any working
    // directory; a replaced shard or another model folder changes it; a shard that cannot be read is an error, not a zero ----
    {
        const fs::path root = base / "ident";
        fs::create_directories(root / "modelA");
        fs::create_directories(root / "modelB");
        fs::create_directories(root / "elsewhere");
        auto put = [](const fs::path& p, const std::string& body) { std::FILE* f = std::fopen(p.string().c_str(), "wb"); std::fwrite(body.data(), 1, body.size(), f); std::fclose(f); };
        put(root / "modelA" / "s-00001.gguf", "AAAAAAAA");
        put(root / "modelA" / "s-00002.gguf", "AAAA");
        put(root / "modelB" / "s-00001.gguf", "BBBBBBBBBBBB");
        put(root / "modelB" / "s-00002.gguf", "BB");
        const std::vector<std::string> names = {"s-00002.gguf", "s-00001.gguf"};   // (the pack's order: the id sorts them)
        const fs::path cwd0 = fs::current_path();
        fs::current_path(root / "elsewhere");   // (the old code stat()ed the bare names here: 0 0 for every shard)
        std::string ia, ib, ia2, err;
        CHECK(strata::ds4::ds4_pack_identity((root / "modelA").string(), names, "g1", ia, err));
        CHECK(strata::ds4::ds4_pack_identity((root / "modelB").string(), names, "g1", ib, err));
        CHECK(!ia.empty() && ia != ib);   // (two model folders with the same shard names are two packs)
        CHECK(strata::ds4::ds4_pack_identity((root / "modelA").string(), names, "g1", ia2, err) && ia2 == ia);   // (stable)
        // a shard replaced by one of the same size: its time changes the id
        put(root / "modelA" / "s-00002.gguf", "aaaa");
        fs::last_write_time(root / "modelA" / "s-00002.gguf", fs::file_time_type::clock::now() + std::chrono::seconds(3));
        CHECK(strata::ds4::ds4_pack_identity((root / "modelA").string(), names, "g1", ia2, err) && ia2 != ia);
        // a missing shard, and an empty one: no identity
        CHECK(!strata::ds4::ds4_pack_identity((root / "modelA").string(), {"s-00001.gguf", "s-00003.gguf"}, "g1", ia2, err) && !err.empty());
        put(root / "modelA" / "s-00001.gguf", "");
        CHECK(!strata::ds4::ds4_pack_identity((root / "modelA").string(), names, "", ia2, err) && !err.empty());   // no geometry: no identity
        // the metadata-only case: the same files (names, sizes, times) with different expert offsets is another pack, so the tier refills
        std::string ig1, ig2;
        CHECK(strata::ds4::ds4_pack_identity((root / "modelB").string(), names, "offsets-A", ig1, err));
        CHECK(strata::ds4::ds4_pack_identity((root / "modelB").string(), names, "offsets-B", ig2, err) && ig1 != ig2);
        fs::current_path(cwd0);
    }

#if !defined(_WIN32)
    // ---- §6.2: two names of one device, a model file as a device, and foreign data are refused before anything is written ----
    {
        World w(base, "safety", false);
        auto t = std::make_unique<ExpertTierSet>();
        std::string err;
        fs::create_symlink(w.devA, w.dir / "tierA.link");
        CHECK(!t->open({w.specs[0], TierSpec{(w.dir / "tierA.link").string(), 16, 0, false}}, err));
        CHECK(err.find("same device") != std::string::npos);
        fs::create_hard_link(w.shard, w.dir / "shard.hard");
        // (the shard is 224 KiB, under mib=1: grown through the link - which IS the shard - so the size check passes and the
        // overlap check is what refuses it; it is put back to its size right after)
        fs::resize_file(w.dir / "shard.hard", 2ull << 20);
        CHECK(!t->open({TierSpec{(w.dir / "shard.hard").string(), 1, 0, false}}, err, {w.shard}));
        CHECK(err.find("model's file") != std::string::npos);
        fs::resize_file(w.shard, SHARD_BYTES);
        {   // someone's data where the header goes
            std::FILE* f = std::fopen(w.devB.c_str(), "r+b");
            std::fseek(f, 100, SEEK_SET);
            std::fwrite("data", 1, 4, f);
            std::fclose(f);
        }
        CHECK(!t->open(w.specs, err));
        CHECK(err.find("not a Strata expert tier") != std::string::npos);
        CHECK(t->device_count() == 0);   // (nothing half-open)
    }
#endif

    // ---- §9/§6.4: a flush that fails leaves the tier unreadable and marked; the next start refills ----
    for (int fail_at : {1, 2, 3, 4}) {   // device 1's flushes in fill: 1 data, 2 chunk table, 3 header A, 4 header B
        World w(base, "flush" + std::to_string(fail_at), false);
        auto t = std::make_unique<ExpertTierSet>();
        std::string err;
        CHECK(t->open(w.specs, err));
        tier_hooks(w, t);
        bool reused = true;
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        int n = 0;
        t->set_flush_fault([&](size_t d) { return d == 1 && ++n == fail_at; });
        ExpertTierSet::Fill f{};
        CHECK(!do_fill(w, t.get(), f));
        CHECK(!t->readable());
        int dev = -1;
        uint64_t k = 0;
        CHECK(!t->find(1, 0, dev, k));
        t->set_flush_fault(nullptr);
        t->close();   // (the marked devices get fill_state 1 here)
        CHECK(t->open(w.specs, err));
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(!reused);
        CHECK(do_fill(w, t.get(), f));   // (a clean fill afterwards is a tier again)
        CHECK(t->open(w.specs, err));
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(reused);
        t->close();
    }
    {   // the header decide writes cannot be flushed: decide says so instead of going on
        World w(base, "flushdecide", false);
        auto t = std::make_unique<ExpertTierSet>();
        std::string err;
        CHECK(t->open(w.specs, err));
        t->set_flush_fault([](size_t d) { return d == 0; });
        bool reused = true;
        CHECK(!t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        CHECK(why.find("header") != std::string::npos);
        t->set_flush_fault(nullptr);
        t->close();
    }

    // ---- startup: the store's background preload reads through the tier hooks while the fill runs and publishes.  A lookup
    // that answers must see the filled bytes (release/acquire on readable_; run this test under -fsanitize=thread too) ----
    {
        World w(base, "race", false);
        auto t = std::make_unique<ExpertTierSet>();
        std::string err;
        CHECK(t->open(w.specs, err));
        tier_hooks(w, t);
        bool reused = true;
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        std::atomic<bool> stop{false};
        std::atomic<int> seen{0}, wrong{0};
        std::thread bg([&] {   // (the model's store_bg thread: preload pieces, and the lookups a decode would make)
            std::vector<std::pair<int, int>> all;
            for (int l = 0; l < NL; ++l)
                for (int e = 0; e < NE; ++e) all.push_back({l, e});
            while (!stop.load()) {
                for (const auto& x : all) {
                    int dev = -1;
                    uint64_t k = 0;
                    if (!t->find(x.first, x.second, dev, k)) continue;
                    seen.fetch_add(1);
                    const uint8_t* sb = t->slot_base(dev, k);
                    for (int r = 0; r < 3; ++r)
                        if (sb == nullptr || !same_matrix(sb + (size_t) r * (MB4 + 8192), x.first, x.second, r)) wrong.fetch_add(1);
                }
                const std::vector<std::pair<int, int>> piece = {all[(size_t) (seen.load() % (int) all.size())]};
                (void) w.st.preload(piece, 0);
            }
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        ExpertTierSet::Fill f{};
        CHECK(do_fill(w, t.get(), f));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        stop.store(true);
        bg.join();
        CHECK(seen.load() > 0);    // (the lookups answered once the fill was published)
        CHECK(wrong.load() == 0);  // (and every byte they saw was the file's)
        t->close();
    }

    // ---- §7.1 (geometry): the identity's text from either layout. A GGUF layer with empty region_* is described without a crash;
    // identical geometry gives identical text; a change to a byte's source, its reading, its slot place or the slot layout changes
    // it; the file list's order does not; a layer that cannot be described in full is refused, with no text ----
    {
        const std::vector<std::string> files = {"shard-00001.gguf", "shard-00002.gguf"};
        const TierGeoShape shape = geo_shape();
        std::string text, text2, text3, rtext, gerr;
        const std::vector<TierLayerGeo> mx = {mx_geo(0), mx_geo(1), mx_geo(2)};
        CHECK(mx[1].region_file.empty() && mx[1].region_off.empty() && mx[1].region_bytes.empty());   // (a GGUF layer: no regions)
        CHECK(ds4_geometry_text(mx, files, shape, text, gerr) && !text.empty());
        CHECK(text.rfind("strata-ds4-geometry 1\n", 0) == 0);   // (the format's version leads the text)
        CHECK(ds4_geometry_text(mx, files, shape, text2, gerr) && text2 == text);   // (identical geometry: identical text)
        // one change at a time: each must change the text (a changed mutation that is refused would not be a change, so it counts as no)
        auto mut = [&](auto f) {
            std::vector<TierLayerGeo> m = mx;
            f(m);
            std::string t, e;
            return ds4_geometry_text(m, files, shape, t, e) && t != text;
        };
        CHECK(mut([](auto& m) { m[1].off[2] += 1; }));          // a source offset
        CHECK(mut([](auto& m) { m[0].stride[1] = 8192; }));     // a stride (still >= the matrix's bytes)
        CHECK(mut([](auto& m) { m[2].bytes[0] = 2048; }));      // a matrix's bytes
        CHECK(mut([](auto& m) { m[0].type[2] = 13; }));         // a type: the same bytes, read as another type
        CHECK(mut([](auto& m) { m[1].tr[0] = 4096; }));         // its place in the slot (no overlap with the others)
        CHECK(mut([](auto& m) { m[1].blob = 40960; }));         // the slot's blob
        CHECK(mut([](auto& m) { m[0].fmt = 3; }));              // the expert format
        CHECK(mut([](auto& m) { m[0].il = 7; }));               // the layer
        CHECK(mut([](auto& m) { m[0].n_exp = 5; }));            // the expert count
        CHECK(mut([](auto& m) { m[2].file[2] = 0; }));          // the file a matrix is read from
        // the file list: the same files listed the other way, with the indices swapped, is the same geometry
        std::vector<TierLayerGeo> sw = mx;
        for (auto& l : sw)
            for (int r = 0; r < 3; ++r) l.file[r] = 1 - l.file[r];
        CHECK(ds4_geometry_text(sw, {files[1], files[0]}, shape, text3, gerr) && text3 == text);
        // a file's name, and the model's shape or slot layout, change it
        CHECK(ds4_geometry_text(mx, {files[0], "shard-00009.gguf"}, shape, text3, gerr) && text3 != text);
        for (int k = 0; k < 4; ++k) {
            TierGeoShape s2 = shape;
            if (k == 0) s2.dim = 65;
            if (k == 1) s2.ff = 33;
            if (k == 2) s2.nl = 4;
            if (k == 3) s2.slot_layout = 2;
            CHECK(ds4_geometry_text(mx, files, s2, text3, gerr) && text3 != text);
        }
        // the EXL3 layout, alone and beside a GGUF layer (a pack of both)
        const std::vector<TierLayerGeo> rg = {region_geo(4)};
        CHECK(ds4_geometry_text(rg, files, shape, rtext, gerr) && !rtext.empty() && rtext != text);
        auto rmut = [&](auto f) {
            std::vector<TierLayerGeo> m = rg;
            f(m);
            std::string t, e;
            return ds4_geometry_text(m, files, shape, t, e) && t != rtext;
        };
        CHECK(rmut([](auto& m) { m[0].region_off[1] += 16; }));     // a region's offset
        CHECK(rmut([](auto& m) { m[0].region_bytes[0] += 8; }));    // a region's bytes
        CHECK(rmut([](auto& m) { m[0].region_file[1] = 0; }));      // a region's file
        CHECK(rmut([](auto& m) { m[0].piece_src[1][2] += 8; }));    // a piece's place in the region
        CHECK(rmut([](auto& m) { m[0].piece_dst[0][1] = 48; }));    // a piece's place in the slot
        CHECK(rmut([](auto& m) { m[0].piece_bytes[2][2] = 40; }));  // a piece's size
        CHECK(rmut([](auto& m) { m[0].type[1] = 4; }));             // a matrix's bits
        CHECK(rmut([](auto& m) { m[0].blob = 2048; }));             // the slot's blob
        const std::vector<TierLayerGeo> both = {mx_geo(0), region_geo(1)};
        CHECK(ds4_geometry_text(both, files, shape, text3, gerr) && text3 != text && text3 != rtext);

        // refusals: an error and no text.  An identity is never made from a geometry the engine cannot read in full.
        auto refuse = [&](const std::vector<TierLayerGeo>& m, const std::vector<std::string>& fl, const TierGeoShape& sh, const char* must) {
            std::string t, e;
            return !ds4_geometry_text(m, fl, sh, t, e) && t.empty() && !e.empty() && e.find(must) != std::string::npos;
        };
        auto with = [](std::vector<TierLayerGeo> m, auto f) {   // (a copy of m with f applied to its layers)
            f(m);
            return m;
        };
        // the input that crashed before: an EXL3 layer whose regions are missing (an empty array, not a field to read)
        CHECK(refuse(with(rg, [](auto& m) { m[0].region_file.clear(); m[0].region_off.clear(); m[0].region_bytes.clear(); }), files, shape,
                     "incomplete"));
        CHECK(refuse(with(rg, [](auto& m) { m[0].region_off.pop_back(); }), files, shape, "incomplete"));
        CHECK(refuse(with(rg, [](auto& m) { m[0].region_file[0] = 5; }), files, shape, "no file in the pack"));
        CHECK(refuse(with(rg, [](auto& m) { m[0].region_bytes[1] = 0; }), files, shape, "no bytes"));
        CHECK(refuse(with(rg, [](auto& m) { m[0].piece_src[0][0] = 500; }), files, shape, "outside its region"));
        CHECK(refuse(with(rg, [](auto& m) { m[0].piece_dst[2][2] = 1000; }), files, shape, "does not fit"));
        CHECK(refuse(with(rg, [](auto& m) { m[0].piece_dst[0][1] = 0; }), files, shape, "two pieces overlap"));
        CHECK(refuse(with(rg, [](auto& m) { m[0].piece_bytes[0][0] = 0; }), files, shape, "no bytes"));
        CHECK(refuse(with(rg, [](auto& m) { m[0].type[1] = 0; }), files, shape, "EXL3 bits"));
        CHECK(refuse(with(mx, [](auto& m) { m[0].file[1] = -1; }), files, shape, "no file in the pack"));
        CHECK(refuse(with(mx, [](auto& m) { m[0].file[1] = 2; }), files, shape, "no file in the pack"));
        CHECK(refuse(with(mx, [](auto& m) { m[0].bytes[0] = 0; }), files, shape, "no bytes"));
        CHECK(refuse(with(mx, [](auto& m) { m[0].type[0] = 0; }), files, shape, "no type"));
        CHECK(refuse(with(mx, [](auto& m) { m[0].tr[2] = 25000; }), files, shape, "does not fit"));
        CHECK(refuse(with(mx, [](auto& m) { m[0].tr[1] = 0; }), files, shape, "overlap in the blob"));
        CHECK(refuse(with(mx, [](auto& m) { m[0].stride[0] = 1024; }), files, shape, "overlaps the next expert"));
        CHECK(refuse(with(mx, [](auto& m) { m[0].blob = 0; }), files, shape, "no size"));
        CHECK(refuse(with(mx, [](auto& m) { m[0].n_exp = 0; }), files, shape, "no routed experts"));
        CHECK(refuse(with(mx, [](auto& m) { m[0].il = -1; }), files, shape, "no index"));
        CHECK(refuse(with(mx, [](auto& m) { m[0].layout = TierLayerGeo::kNone; }), files, shape, "no expert layout was recorded"));
        CHECK(refuse(with(mx, [](auto& m) { m[0].layout = 9; }), files, shape, "not one the tier knows"));
        CHECK(refuse({}, files, shape, "no layer"));
        CHECK(refuse(mx, {}, shape, "no model files"));
        CHECK(refuse(mx, {"shard 1.gguf", "shard-00002.gguf"}, shape, "cannot be written"));
        TierGeoShape nodim = shape;
        nodim.ff = 0;
        CHECK(refuse(mx, files, nodim, "dimensions"));
        // (a rejected geometry leaves no half-written text behind either)
        std::string left = "untouched";
        CHECK(!ds4_geometry_text({}, files, shape, left, gerr) && left.empty());
    }

    // ---- §7.1 with §5 and §9: the identity made from the geometry decides reuse or refill of the placement, as init_tier makes
    // it (the geometry's digest is part of the pack id).  The same geometry reuses; a changed stride or type refills, and a refilled
    // tier is reused by the next start ----
    {
        World w(base, "geo", false);   // (its shards: shard.gguf and, for the down matrix, shard-b.gguf: temp files only)
        const std::vector<std::string> names = {"shard.gguf", "shard-b.gguf"};
        const TierGeoShape shape = geo_shape();
        {
            std::FILE* fb = std::fopen((w.dir / "shard-b.gguf").string().c_str(), "wb");
            std::fputs("down-shard", fb);
            std::fclose(fb);
        }
        auto id_of = [&](const std::vector<TierLayerGeo>& g, std::string& id) {
            std::string text, e;
            if (!ds4_geometry_text(g, names, shape, text, e)) return false;
            return strata::ds4::ds4_pack_identity(w.dir.string(), names, strata::core::tier_sha256_hex((const uint8_t*) text.data(), text.size()), id, e);
        };
        std::vector<TierLayerGeo> lg = {mx_geo(0), mx_geo(1), mx_geo(2)};
        std::vector<TierLayerGeo> lgStride = lg;
        lgStride[1].stride[0] = 8192;   // (a stride: the same files, the bytes read from elsewhere)
        std::vector<TierLayerGeo> lgType = lg;
        lgType[0].type[2] = 13;         // (a type: the same bytes, read as another type)
        std::string idA, idB, idC, err;
        CHECK(id_of(lg, idA) && idA.compare(0, 4, "sha:") == 0);
        CHECK(id_of(lgStride, idB) && idB != idA);
        CHECK(id_of(lgType, idC) && idC != idA && idC != idB);
        // the first start: nothing on the devices, so the tier is filled
        bool reused = false;
        {
            auto t = std::make_unique<ExpertTierSet>();
            CHECK(t->open(w.specs, err));
            tier_hooks(w, t);
            CHECK(t->decide(SLOT, 2, 4, NL, NE, idA, profA, false, ranked, reused, why, notes));
            CHECK(!reused);
            ExpertTierSet::Fill f{};
            CHECK(do_fill(w, t.get(), f));
            CHECK(t->readable() && f.placed == (int64_t) ranked.size());
            t->close();
        }
        // the next start, the same geometry: the placement is reused and nothing is copied
        {
            auto t = std::make_unique<ExpertTierSet>();
            CHECK(t->open(w.specs, err));
            CHECK(t->decide(SLOT, 2, 4, NL, NE, idA, profA, false, ranked, reused, why, notes));
            CHECK(reused);
            t->close();
        }
        // a changed stride: another pack id, so the start refills; the refill completes and the next start reuses it
        {
            auto t = std::make_unique<ExpertTierSet>();
            CHECK(t->open(w.specs, err));
            CHECK(t->decide(SLOT, 2, 4, NL, NE, idB, profA, false, ranked, reused, why, notes));
            CHECK(!reused && why.find("different pack") != std::string::npos);
            tier_hooks(w, t);
            ExpertTierSet::Fill f{};
            CHECK(do_fill(w, t.get(), f));
            t->close();
        }
        {
            auto t = std::make_unique<ExpertTierSet>();
            CHECK(t->open(w.specs, err));
            CHECK(t->decide(SLOT, 2, 4, NL, NE, idB, profA, false, ranked, reused, why, notes));
            CHECK(reused);
            t->close();
        }
        // a changed type alone (the same bytes): refills too
        {
            auto t = std::make_unique<ExpertTierSet>();
            CHECK(t->open(w.specs, err));
            CHECK(t->decide(SLOT, 2, 4, NL, NE, idC, profA, false, ranked, reused, why, notes));
            CHECK(!reused && why.find("different pack") != std::string::npos);
            t->close();
        }
    }

    // ---- DS4_TIER_LOG: the per-device runtime counters (§8.2's line): a demand copy, a guess, and the CPU lane's wait ----
    {
        set_env("DS4_TIER_LOG", "1");
        set_env("DS4_TIER_THREADS", "1");   // (one worker: a long copy ahead deterministically holds the tier copy)
        auto t = std::make_unique<ExpertTierSet>();   // (declared before the world: destroyed after the store, as in Model::Impl)
        auto w = std::make_unique<World>(base, "tierlog", false);
        std::string err;
        CHECK(t->open(w->specs, err));
        CHECK(t->tracing());
        tier_hooks(*w, t);
        w->st.set_tier_trace([&](int dev, uint64_t bytes, bool demand) { t->note_copy_start(dev, bytes, demand); },
                             [&](int dev, uint64_t ns) { t->note_copy_wait(dev, ns); });
        bool reused = true;
        CHECK(t->decide(SLOT, 2, 4, NL, NE, packA, profA, false, ranked, reused, why, notes));
        ExpertTierSet::Fill f{};
        CHECK(do_fill(*w, t.get(), f));
        for (const ExpertTierSet::DeviceStats& d : t->stats())
            CHECK(d.started_experts == 0 && d.completed_jobs == 0);   // (a fill read is a model-file read, not a tier read)
        int dev = -1;
        uint64_t k = 0;
        CHECK(t->find(2, 5, dev, k));
        // A demand read queued behind a long copy on the device's queue: ensure() waits, and the wait is counted.
        BigCopy big;
        big.push(t->queue(dev));
        const int es[1] = {5};
        std::vector<char> ok;
        CHECK(w->st.ensure(2, es, 1, ok) && ok[0] == 1);
        while (big.left.load() > 0) t->queue(dev)->wait_any(1000);
        {
            const std::vector<ExpertTierSet::DeviceStats> ds = t->stats();
            const ExpertTierSet::DeviceStats& a = ds[(size_t) dev];
            CHECK(a.started_experts == 1 && a.demand_experts == 1 && a.prefetch_experts == 0);
            CHECK(a.started_bytes == (uint64_t) 3 * MB4);
            CHECK(a.completed_jobs == 3 && a.demand_jobs == 3);
            CHECK(a.completed_bytes == (uint64_t) 3 * MB4);
            CHECK(a.service_ns > 0 && a.queue_depth_max >= 1);
            CHECK(a.wait_ns > 0);            // (the copy sat behind the long one)
            CHECK(a.cold_jobs <= a.completed_jobs && a.cold_bytes <= a.completed_bytes);
        }
        // A guess is counted as prefetch, not demand.
        w->st.prefetch(2, 4);
        const int es2[1] = {4};
        std::vector<char> ok2;
        CHECK(w->st.ensure(2, es2, 1, ok2) && ok2[0] == 1);
        uint64_t pe = 0, pj = 0;
        for (const ExpertTierSet::DeviceStats& d : t->stats()) {
            pe += d.prefetch_experts;
            pj += d.prefetch_jobs;
        }
        CHECK(pe == 1 && pj == 3);
        w->st.set_tier(nullptr, nullptr, nullptr, nullptr);   // (the world outlives this set's close: no hook may reach it)
        w->st.set_tier_trace(nullptr, nullptr);
        t->close();
        w.reset();
        set_env("DS4_TIER_LOG", nullptr);
        set_env("DS4_TIER_THREADS", nullptr);
    }

    queue_wait_checks();   // IoQueue::wait_until_zero on a bare queue (DS4_TIER_WAKE's wait)
#if !defined(_WIN32)
    direct_checks(base, ranked, packA, profA);   // DS4_TIER_DIRECT (opt-in): see the section above
#endif

    fs::remove_all(base);
    if (fails) {
        std::printf("%d expert tier checks failed\n", fails);
        return 1;
    }
    if (skips) std::printf("expert tier: all checks passed (%d skipped: see SKIP above)\n", skips);
    else std::printf("expert tier: all checks passed\n");
    return 0;
}
