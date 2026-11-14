// The DeepSeek engine's expert tier without a GPU or a model (src/dsv41/ds4_tier.hpp): docs/EXPERT_TIER_DSV41_PLAN.md
// §14. Synthetic geometry and small files: the fill's bytes equal the shard's (acceptance 3), the store's reads of tier
// experts come off the tier and are the same bytes (§8.1/§8.2), a start reuses the placement and copies nothing (5), a
// profile change does not refill (6), while a pack, guard, torn-header, corrupt-table, abandoned-fill or verify=full
// mismatch refills (7), and a failed copy marks its device (§9).
//   ds4_expert_tier_test   exit 0: all passed
#include "ds4_tier.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <atomic>
#include <map>
#include <memory>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#if !defined(_WIN32)
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

    fs::remove_all(base);
    if (fails) {
        std::printf("%d expert tier checks failed\n", fails);
        return 1;
    }
    std::printf("expert tier: all checks passed\n");
    return 0;
}
