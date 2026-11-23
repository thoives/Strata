// tests/core/expert_source_striped_test.cpp - the prompt path's reads of striped tier experts (docs/EXPERT_TIER.md, "The layout").
//
// A FileExpertSource with a striped expert tier must serve the tier's experts from the devices, not from experts.bin:
//   - transient() is true for a striped expert the RAM copy does not hold, so the prompt path copies it with copy_blob
//     (which gathers the stripes) instead of taking a pointer;
//   - copy_blob and copy_blobs return the file's own bytes for every expert, held or not, in a mixed batch;
//   - the tier's read counters count the held experts, not the file.
// The whole layout is checked alongside: its held experts keep a stable pointer (has()) and transient() stays false.
// blob_stable() on a striped expert still returns the file's bytes: that is the file fallback the pointer consumers keep
// until their staging leases exist (docs/EXPERT_TIER.md), and this test pins that it is correct, not that it is fast.
//
// Not covered here: the RAM copy and exchange overrides (they need a complement), GGUF mapped in place (blob_parts),
// device failures, and the decode path. Those are named in the test report, not claimed by this file.
#include "strata/core/expert_source.hpp"
#include "strata/core/expert_tier.hpp"
#include "strata/core/expert_tier_dev.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::core;

namespace {

void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}

constexpr int64_t kLayers = 2;
constexpr int64_t kExperts = 8;
constexpr int kDevices = 3;

struct TempDir {
    fs::path path;
    TempDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("strata-expert-source-striped-" + std::to_string(stamp));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

std::string make_device(const TempDir& d, const std::string& name, uint64_t mib) {
    const fs::path p = d.path / name;
    {
        std::ofstream out(p, std::ios::binary);
        out << '\0';
    }
    fs::resize_file(p, mib << 20);
    return p.string();
}

TierIdentity identity_for(const std::vector<uint64_t>& blobs) {
    TierIdentity id;
    id.pack_id = "st:34123456789:1730430000";
    const std::vector<uint8_t> profile = {'S', 'R', 'P', 'F', 1};
    id.profile_id = "sha:" + tier_sha256_hex(profile.data(), profile.size());
    id.n_layers = kLayers;
    id.n_expert = kExperts;
    id.layer_blob_bytes = blobs;
    return id;
}

size_t index_of(int64_t layer, int64_t expert) { return (size_t) (layer * kExperts + expert); }

// Writes a synthetic experts.bin of the canonical layout's size. The bytes are random, so a wrong offset or a mixed-up
// stripe shows as a byte difference.
void write_pack(const fs::path& dir, uint64_t bytes) {
    std::mt19937_64 rng(0x5EED5EEDull);
    std::vector<uint8_t> data((size_t) bytes);
    for (uint8_t& b : data) b = (uint8_t) (rng() >> 56);
    std::ofstream out(dir / "experts.bin", std::ios::binary | std::ios::trunc);
    out.write((const char*) data.data(), (std::streamsize) data.size());
    require((bool) out, "could not write the synthetic experts.bin");
}

// Fills a tier of the given layout over every expert (rank order: layer-major) and returns it open. `mib` per device.
void fill_tier(ExpertTierSet& tier, FileExpertSource& src, const TempDir& dir, const std::vector<uint64_t>& blobs, bool striped,
               uint64_t stripe_bytes, uint64_t mib, const std::string& tag) {
    std::vector<TierSpec> specs;
    for (int d = 0; d < kDevices; ++d) specs.push_back({make_device(dir, tag + std::to_string(d) + ".dev", mib), mib, 0, false});
    tier.set_layout(striped, stripe_bytes);
    std::string err;
    bool reused = false;
    require(tier.open(specs, identity_for(blobs), err, reused), tag + ": open: " + err);
    require(!reused, tag + ": a fresh device was reused");
    std::vector<std::pair<int32_t, int32_t>> ranked;
    for (int64_t l = 0; l < kLayers; ++l)
        for (int64_t e = 0; e < kExperts; ++e) ranked.push_back({(int32_t) l, (int32_t) e});
    require(tier.fill(src, ranked, identity_for(blobs), err), tag + ": fill: " + err);
}

// The prompt path's view of one striped configuration: transient, copy_blob, copy_blobs, and the counters.
void check_striped(FileExpertSource& src, const ExpertTierSet& tier, const std::vector<std::vector<uint8_t>>& truth,
                   const std::string& what) {
    int64_t held = 0;
    for (int64_t l = 0; l < kLayers; ++l)
        for (int64_t e = 0; e < kExperts; ++e) {
            const bool h = tier.held(l, e);
            held += h ? 1 : 0;
            require(src.transient(l, e) == h, what + ": transient(" + std::to_string(l) + "," + std::to_string(e) +
                                                   ") should be " + (h ? "true (copied from the stripes)" : "false"));
            require(tier.has(l, e) == false || !h, what + ": a striped expert must not have a contiguous blob");
        }
    require(held > 0, what + ": no expert was held, so the check is empty");
    require(held < kLayers * kExperts || what.find("partial") == std::string::npos,
            what + ": the partial configuration should not fit every expert");

    // one expert at a time: byte-exact, and the tier's counters rise only for the held one
    for (int64_t l = 0; l < kLayers; ++l)
        for (int64_t e = 0; e < kExperts; ++e) {
            const auto& want = truth[index_of(l, e)];
            std::vector<uint8_t> got(want.size(), 0xAB);
            const uint64_t reads0 = src.tier_reads();
            require(src.copy_blob(l, e, got.data()), what + ": copy_blob refused expert " + std::to_string(l) + "," + std::to_string(e));
            require(got == want, what + ": copy_blob bytes differ from experts.bin");
            const bool h = tier.held(l, e);
            require(src.tier_reads() - reads0 == (h ? 1u : 0u), what + ": copy_blob counted the wrong tier reads");
        }

    // one batch with held and file experts mixed, in a shuffled order: byte-exact per destination
    std::vector<std::pair<int32_t, int32_t>> order;
    for (int64_t l = 0; l < kLayers; ++l)
        for (int64_t e = 0; e < kExperts; ++e) order.push_back({(int32_t) l, (int32_t) e});
    std::mt19937 shuffle_rng(7);
    std::shuffle(order.begin(), order.end(), shuffle_rng);
    std::vector<int32_t> bl, be;
    std::vector<std::vector<uint8_t>> out;
    for (const auto& pr : order) {
        bl.push_back(pr.first);
        be.push_back(pr.second);
    }
    std::vector<uint8_t*> dst;
    for (size_t i = 0; i < bl.size(); ++i) {
        out.emplace_back(truth[index_of(bl[i], be[i])].size(), 0xCD);
        dst.push_back(out.back().data());
    }
    const uint64_t reads0 = src.tier_reads();
    require(src.copy_blobs(bl.data(), be.data(), dst.data(), bl.size()), what + ": copy_blobs refused the batch");
    for (size_t i = 0; i < bl.size(); ++i)
        require(out[i] == truth[index_of(bl[i], be[i])], what + ": copy_blobs bytes differ for expert " + std::to_string(bl[i]) + "," +
                                                              std::to_string(be[i]));
    require(src.tier_reads() - reads0 == (uint64_t) held, what + ": copy_blobs counted the wrong tier reads");

    // a pointer consumer still gets the file's bytes (the documented fallback), never a stale or half-gathered buffer
    for (int64_t l = 0; l < kLayers; ++l)
        for (int64_t e = 0; e < kExperts; ++e) {
            const uint8_t* p = src.blob_stable(l, e);
            require(p != nullptr, what + ": blob_stable returned no pointer");
            require(std::memcmp(p, truth[index_of(l, e)].data(), truth[index_of(l, e)].size()) == 0,
                    what + ": blob_stable bytes differ from experts.bin");
        }
}

// The decode path's reads: blob() takes a staging slot for a striped expert and fills it from the devices. The bytes are checked
// right after each call (a blob() pointer is valid until the next blob() call), a second ask is a slot hit with the same bytes,
// and the counters show tier reads for the held experts and file reads for the rest.
void check_decode(FileExpertSource& src, const ExpertTierSet& tier, const std::vector<std::vector<uint8_t>>& truth, const std::string& what) {
    for (int64_t l = 0; l < kLayers; ++l)
        for (int64_t e = 0; e < kExperts; ++e) {
            const auto& want = truth[index_of(l, e)];
            const uint64_t tier0 = src.tier_read_bytes(), file0 = src.file_read_bytes();
            const uint8_t* p = src.blob(l, e);
            require(p != nullptr, what + ": blob() returned no pointer for " + std::to_string(l) + "," + std::to_string(e));
            require(std::memcmp(p, want.data(), want.size()) == 0, what + ": blob() bytes differ for " + std::to_string(l) + "," + std::to_string(e));
            if (tier.held(l, e)) {
                require(src.tier_read_bytes() - tier0 == want.size(), what + ": a held expert's first decode read was not a tier read");
                require(src.file_read_bytes() == file0, what + ": a held expert's decode read touched the file");
                const uint64_t tier1 = src.tier_read_bytes();
                const uint8_t* again = src.blob(l, e);
                require(again != nullptr && std::memcmp(again, want.data(), want.size()) == 0, what + ": the second decode read differs");
                require(src.tier_read_bytes() == tier1, what + ": the second decode read was not a slot hit");
            } else {
                require(src.file_read_bytes() - file0 == want.size(), what + ": a file expert's decode read was not counted as a file read");
            }
        }
}

void test_striped(uint64_t mib, uint64_t stripe_bytes, const std::string& what) {
    TempDir dir;
    std::string err;
    require(strata::kernels::cpu::expert_layout_load(dir.path.string(), kLayers, kExperts, err), "could not load the canonical layout: " + err);
    const auto& lay = strata::kernels::cpu::expert_layout();
    write_pack(dir.path, lay.total);

    FileExpertSource src;
    require(src.open(dir.path.string(), kLayers, kExperts, err), "open: " + err);
    std::vector<std::vector<uint8_t>> truth(kLayers * kExperts);
    std::vector<uint64_t> blobs((size_t) kLayers);
    for (int64_t l = 0; l < kLayers; ++l) {
        blobs[(size_t) l] = lay.blob_bytes(l);
        for (int64_t e = 0; e < kExperts; ++e) {
            truth[index_of(l, e)].resize((size_t) blobs[(size_t) l]);
            require(src.copy_blob(l, e, truth[index_of(l, e)].data()), "the file's own copy refused an expert");
        }
    }

    ExpertTierSet tier;
    fill_tier(tier, src, dir, blobs, /*striped=*/true, stripe_bytes, mib, "s");
    src.set_tier(&tier);
    check_striped(src, tier, truth, what);
    check_decode(src, tier, truth, what);
    src.set_tier(nullptr);
    tier.close();
}

// The whole layout, the same source: a held expert keeps its pointer and the prompt path does not copy it.
void test_whole_unchanged(uint64_t mib) {
    TempDir dir;
    std::string err;
    require(strata::kernels::cpu::expert_layout_load(dir.path.string(), kLayers, kExperts, err), "could not load the canonical layout: " + err);
    const auto& lay = strata::kernels::cpu::expert_layout();
    write_pack(dir.path, lay.total);

    FileExpertSource src;
    require(src.open(dir.path.string(), kLayers, kExperts, err), "open: " + err);
    std::vector<uint64_t> blobs((size_t) kLayers);
    for (int64_t l = 0; l < kLayers; ++l) blobs[(size_t) l] = lay.blob_bytes(l);

    ExpertTierSet tier;
    fill_tier(tier, src, dir, blobs, /*striped=*/false, 0, mib, "w");
    src.set_tier(&tier);
    int64_t held = 0;
    for (int64_t l = 0; l < kLayers; ++l)
        for (int64_t e = 0; e < kExperts; ++e) {
            if (!tier.held(l, e)) continue;
            ++held;
            require(tier.has(l, e), "whole: a held expert has no contiguous blob");
            require(!src.transient(l, e), "whole: transient() changed for a held expert");
            require(src.blob_stable(l, e) == tier.blob(l, e), "whole: blob_stable no longer returns the tier's pointer");
        }
    require(held > 0, "whole: no expert was held");
    src.set_tier(nullptr);
    tier.close();
}

}  // namespace

int main() {
    try {
        test_striped(32, 4 * 1024, "striped, all held, 4 KiB stripes");
        test_striped(32, 64 * 1024, "striped, all held, 64 KiB stripes");
        test_striped(3, 64 * 1024, "striped, partial fit");   // a few experts on the small devices, the rest from the file
        test_whole_unchanged(32);
        std::cout << "expert_source_striped_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "expert_source_striped_test: " << error.what() << '\n';
        return 1;
    }
}
