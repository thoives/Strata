// tests/core/expert_tier_copy_test.cpp - §12.4 of docs/EXPERT_TIER_PLAN.md: the copy workers.
//
// A batch whose tier-held blobs are copied by the tier's workers and whose rest comes from the file source gives
// the same bytes as the file path alone; worker count 1, 2 and 4 give the same bytes; the shim's (tier_sim.hpp,
// §10) delays show the tier share and the file share of the §7.3 split overlapping by timing order. The tmpfs
// files under /tmp/strata-tier/ make every check here valid for correctness (plan §10); nothing here is a speed
// claim. The engine's own split lives in FileExpertSource::copy_blobs (CUDA build); this test drives the same
// pattern: tier share on the tier's workers, file share on the caller's thread, at the same time.
#include "strata/core/expert_tier.hpp"
#include "strata/core/expert_source.hpp"
#include "tier_sim.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace strata::core;

// The out-of-line ExpertSource members the fakes' vtables still reference (this test does not link
// src/core/expert_source.cpp: no CUDA here).
namespace strata::core {
bool ExpertSource::copy_blob(int64_t, int64_t, uint8_t*) { return false; }
bool ExpertSource::copy_blobs(const int32_t*, const int32_t*, uint8_t* const*, size_t) { return true; }
}

static void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}

namespace fs = std::filesystem;

static constexpr int64_t kLayers = 4, kExperts = 8;
static const uint64_t kBlobs[kLayers] = {256u << 10, 256u << 10, 1u << 20, 1u << 20};

static uint8_t truth_byte(int64_t l, int64_t e, size_t i) {
    return (uint8_t) ((i * 131u + (size_t) (l * kExperts + e) * 37u) ^ (i >> 8));
}

struct FakeFileTier : ExpertSource {
    std::vector<std::vector<uint8_t>> store;
    FakeFileTier() {
        store.resize((size_t) kLayers * kExperts);
        for (int64_t l = 0; l < kLayers; ++l)
            for (int64_t e = 0; e < kExperts; ++e) {
                std::vector<uint8_t>& t = store[(size_t) l * kExperts + e];
                t.resize((size_t) kBlobs[l]);
                for (size_t i = 0; i < t.size(); ++i) t[i] = truth_byte(l, e, i);
            }
    }
    const uint8_t* blob(int64_t layer, int64_t expert) override {
        return store[(size_t) layer * kExperts + expert].data();
    }
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override {
        const std::vector<uint8_t>& t = store[(size_t) layer * kExperts + expert];
        std::memcpy(dst, t.data(), t.size());
        return true;
    }
};

struct Dir {
    fs::path path;
    explicit Dir() {
        // §10: test files under /tmp/strata-tier/, removed after the run.
        path = fs::temp_directory_path() / ("strata-tier/copy-" + std::to_string(rand()));
        fs::create_directories(path);
    }
    ~Dir() { std::error_code ec; fs::remove_all(path, ec); }
};

static fs::path make_device(Dir& d, const std::string& name, uint64_t mib) {
    fs::path p = d.path / name;
    std::ofstream out(p, std::ios::binary);
    out << std::string(1, '\0');
    out.close();
    fs::resize_file(p, mib << 20);
    return p;
}

static std::vector<std::pair<int32_t, int32_t>> ranked_first6() {
    // the six hottest in band order; the other two experts stay file-only (the §12.4 batch mixes both shares)
    std::vector<std::pair<int32_t, int32_t>> r;
    for (int64_t e = 0; e < 6; ++e) r.push_back({(int32_t) (e % kLayers), (int32_t) e});
    return r;
}

static TierIdentity identity_for(const std::vector<uint8_t>& profile_bytes, const std::vector<uint64_t>& blobs) {
    TierIdentity id;
    id.pack_id = "st:9876543210:1730400000";
    id.profile_id = "sha:" + tier_sha256_hex(profile_bytes.data(), profile_bytes.size());
    id.n_layers = kLayers;
    id.n_expert = kExperts;
    id.layer_blob_bytes = blobs;
    return id;
}

// §12.4 case 1: a batch with tier and file blobs gives the same bytes as the file path alone. The tier share
// runs on the tier's own workers and the file share on this thread at the same time, the §7.3 pattern.
static void test_split_batch_same_bytes() {
    Dir dir;
    const std::vector<uint64_t> blobs = {256u << 10, 256u << 10, 1u << 20, 1u << 20};
    std::vector<TierSpec> specs = {{make_device(dir, "a", 4).string(), 4, 0, false},
                                   {make_device(dir, "b", 4).string(), 4, 0, false}};
    FakeFileTier file_tier;
    const auto r = ranked_first6();
    TierIdentity id = identity_for({'S', 'T', 'R', 'P', 9}, blobs);
    ExpertTierSet set;
    std::string err;
    bool reused = true;
    require(set.open(specs, id, err, reused), err);
    require(set.fill(file_tier, r, id, err), err);

    // the whole grid as one batch: 6 pairs the tier holds, 2 only the file tier has.
    std::vector<std::vector<uint8_t>> dst((size_t) kLayers * kExperts);
    std::vector<uint8_t*> dstp;
    std::vector<int32_t> layers, experts;
    for (int64_t l = 0; l < kLayers; ++l)
        for (int64_t e = 0; e < kExperts; ++e) {
            const size_t i = (size_t) l * kExperts + e;
            layers.push_back((int32_t) l);
            experts.push_back((int32_t) e);
            dst[i].assign((size_t) kBlobs[l], 0);
            dstp.push_back(dst[i].data());
        }
    // split as FileExpertSource::copy_blobs does (§7.3): the tier share to the tier, the rest to the file tier.
    std::vector<int32_t> tl, te;
    std::vector<uint8_t*> td;
    std::vector<size_t> fi;
    for (size_t i = 0; i < layers.size(); ++i) {
        if (set.has(layers[i], experts[i])) {
            tl.push_back(layers[i]); te.push_back(experts[i]); td.push_back(dstp[i]);
        } else fi.push_back(i);
    }
    require(!tl.empty() && !fi.empty(), "the batch did not mix a tier share and a file share");
    bool tier_ok = true;
    uint64_t tbytes = 0;
    std::thread tier_worker([&] {
        tier_ok = set.copy_blobs(tl.data(), te.data(), td.data(), tl.size(), tbytes);
    });
    bool file_ok = true;
    for (size_t i : fi)
        if (!file_tier.copy_blob(layers[i], experts[i], dstp[i])) { file_ok = false; break; }
    if (tier_worker.joinable()) tier_worker.join();
    require(tier_ok && file_ok, "a share of the split batch failed");
    require(tbytes == set.bytes_held(), "the tier share's byte count is wrong");
    for (size_t i = 0; i < dst.size(); ++i) {
        const int64_t l = (int64_t) (i / (size_t) kExperts), e = (int64_t) (i % (size_t) kExperts);
        require(std::equal(file_tier.store[i].begin(), file_tier.store[i].end(), dst[i].begin()),
                "the split batch served bytes different from the file path's");
    }
    set.close();
}

// §12.4 case 2: worker count 1, 2 and 4 give the same bytes (uniform deal, layer order per device).
static void test_worker_caps_same_bytes() {
    Dir dir;
    const std::vector<uint64_t> blobs = {256u << 10, 256u << 10, 1u << 20, 1u << 20};
    std::vector<TierSpec> specs = {{make_device(dir, "a", 4).string(), 4, 0, false},
                                   {make_device(dir, "b", 4).string(), 4, 0, false}};
    FakeFileTier file_tier;
    const auto r = ranked_first6();
    TierIdentity id = identity_for({'w'}, blobs);
    ExpertTierSet set;
    std::string err;
    bool reused = true;
    require(set.open(specs, id, err, reused), err);
    require(set.fill(file_tier, r, id, err), err);

    std::vector<int32_t> layers, experts;
    std::vector<std::vector<uint8_t>> dst(r.size());
    std::vector<uint8_t*> dstp;
    for (size_t i = 0; i < r.size(); ++i) {
        layers.push_back(r[i].first);
        experts.push_back(r[i].second);
        dst[i].assign((size_t) kBlobs[r[i].first], 0);
        dstp.push_back(dst[i].data());
    }
    for (const int cap : {1, 2, 4}) {
        set.set_worker_cap(cap);
        for (auto& v : dst) std::fill(v.begin(), v.end(), 0);
        uint64_t bytes = 0;
        require(set.copy_blobs(layers.data(), experts.data(), dstp.data(), r.size(), bytes), "copy_blobs failed");
        require(bytes == set.bytes_held(), "the batch's byte count is wrong");
        for (size_t i = 0; i < r.size(); ++i) {
            const std::vector<uint8_t>& t = file_tier.store[(size_t) r[i].first * kExperts + r[i].second];
            require(dst[i] == t, "a worker cap produced wrong bytes");
        }
    }
    set.close();
}

// §12.4 case 3 (P0-sim, §10 shim): the tier share and the shimmed file share overlap in time, checked by the
// order of the intervals, not by any speed number. This shapes the §7.3 worker pattern; the simulated link says
// nothing about hardware.
static void test_shim_shares_overlap() {
    Dir dir;
    const std::vector<uint64_t> blobs = {256u << 10, 256u << 10, 1u << 20, 1u << 20};
    std::vector<TierSpec> specs = {{make_device(dir, "a", 4).string(), 4, 0, false},
                                   {make_device(dir, "b", 4).string(), 4, 0, false}};
    FakeFileTier file_tier;
    const auto r = ranked_first6();
    TierIdentity id = identity_for({'o'}, blobs);
    ExpertTierSet set;
    std::string err;
    bool reused = true;
    require(set.open(specs, id, err, reused), err);
    require(set.fill(file_tier, r, id, err), err);

    SimSource sim(file_tier, blobs, 1.0, 300);   // a simulated 1 GiB/s link with a 300 us latency (§10 shim)

    // the tier share (the 6 held pairs) on the tier's own workers, dst buffers of its own:
    std::vector<int32_t> tl, te;
    for (int64_t l = 0; l < kLayers; ++l)
        for (int64_t e = 0; e < kExperts; ++e)
            if (set.has(l, e)) { tl.push_back((int32_t) l); te.push_back((int32_t) e); }
    std::vector<std::vector<uint8_t>> tdst(tl.size());
    std::vector<uint8_t*> tdptr;
    for (size_t i = 0; i < tl.size(); ++i) {
        tdst[i].assign((size_t) kBlobs[tl[i]], 0);
        tdptr.push_back(tdst[i].data());
    }
    // the file share, every expert, through the shim, on this thread (one worker: the engine's file path is
    // one stager thread at a time, §7.3):
    std::vector<std::vector<uint8_t>> fdst((size_t) kLayers * kExperts);
    for (int64_t l = 0; l < kLayers; ++l)
        for (int64_t e = 0; e < kExperts; ++e)
            fdst[(size_t) l * kExperts + e].assign((size_t) kBlobs[l], 0);

    uint64_t tbytes = 0;
    const int64_t t_tier0 = sim.now_us();
    std::thread tier_worker([&] { set.copy_blobs(tl.data(), te.data(), tdptr.data(), tl.size(), tbytes); });
    bool file_ok = true;
    for (int64_t l = 0; l < kLayers && file_ok; ++l)
        for (int64_t e = 0; e < kExperts && file_ok; ++e)
            if (!sim.copy_blob(l, e, fdst[(size_t) l * kExperts + e].data())) file_ok = false;
    if (tier_worker.joinable()) tier_worker.join();
    const int64_t t_tier1 = sim.now_us();
    require(file_ok, "the shimmed share failed");

    // Timing order: some shimmed copy is running while the tier share runs (the shares overlap), and the shimmed
    // copies are sequential (the file share's single worker). Interval order only; no speed number is compared.
    const auto iv = sim.intervals();
    require(iv.size() == (size_t) kLayers * kExperts, "the shim did not record every copy");
    bool overlapped = false;
    for (const auto& c : iv)
        if (c.t0_us < t_tier1 && c.t1_us > t_tier0) overlapped = true;
    require(overlapped, "the tier share did not overlap any shimmed file copy (timing order)");
    for (size_t i = 1; i < iv.size(); ++i)
        require(iv[i].t0_us >= iv[i - 1].t1_us, "the file share ran more than one worker: the split is wrong");
    require(tbytes == set.bytes_held(), "the tier share's byte count is wrong");
    set.close();
}

int main() {
    std::srand(20261031);
    try {
        test_split_batch_same_bytes();
        test_worker_caps_same_bytes();
        test_shim_shares_overlap();
        std::cout << "expert_tier_copy_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "expert_tier_copy_test: " << error.what() << '\n';
        return 1;
    }
}
