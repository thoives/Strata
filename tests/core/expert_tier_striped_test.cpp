// tests/core/expert_tier_striped_test.cpp - the striped layout of the Qwen engine's expert tier (docs/EXPERT_TIER.md, "The
// layout"), on a temp directory, like expert_tier_test.cpp.  Checked here: the bytes gathered back from the stripes are the
// blob's bytes for every stripe size, device count, worker count and blob size (blobs that end in a short stripe included);
// a restart reuses the stripes only in the layout, stripe unit and device order they were written with; a corrupted stripe is
// caught by verify=full and never served; a torn fill refills; a whole-layout tier is unchanged.  No timing claim is made.
#include "strata/core/expert_tier.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/expert_tier_dev.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::core;

// The out-of-line ExpertSource members this test's fake needs (it does not link src/core/expert_source.cpp: no CUDA here).
namespace strata::core {
bool ExpertSource::copy_blob(int64_t, int64_t, uint8_t*) { return false; }
bool ExpertSource::copy_blobs(const int32_t*, const int32_t*, uint8_t* const*, size_t) { return true; }
}

namespace fs = std::filesystem;

static void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}

static constexpr int64_t kLayers = 4, kExperts = 8;
// Two layers of 300 KiB + 123 bytes (not a 4 KiB multiple, a short last stripe for most stripe sizes) and two of 1 MiB + 4097.
static const std::vector<uint64_t> kBlobs = {300u * 1024 + 123, 300u * 1024 + 123, (1u << 20) + 4097, (1u << 20) + 4097};

static uint8_t truth_byte(int64_t l, int64_t e, size_t i, uint32_t salt) {
    return (uint8_t) ((i * 131u + (size_t) (l * kExperts + e) * 37u + salt) ^ (i >> 8));
}

// The truth: every expert's blob, and a copy source that counts its reads (the fill reads each placed blob once).
struct Source : ExpertSource {
    std::vector<uint64_t> bytes;
    std::vector<std::vector<uint8_t>> store;
    int copies = 0;
    explicit Source(std::vector<uint64_t> per_layer, uint32_t salt = 0) : bytes(std::move(per_layer)) {
        store.resize((size_t) (kLayers * kExperts));
        for (int64_t l = 0; l < kLayers; ++l)
            for (int64_t e = 0; e < kExperts; ++e) {
                std::vector<uint8_t>& t = store[(size_t) (l * kExperts + e)];
                t.resize((size_t) bytes[(size_t) l]);
                for (size_t i = 0; i < t.size(); ++i) t[i] = truth_byte(l, e, i, salt);
            }
    }
    const uint8_t* blob(int64_t layer, int64_t expert) override { return store[(size_t) (layer * kExperts + expert)].data(); }
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override {
        ++copies;
        const std::vector<uint8_t>& t = store[(size_t) (layer * kExperts + expert)];
        std::memcpy(dst, t.data(), t.size());
        return true;
    }
};

struct Dir {
    fs::path path;
    Dir() {
        path = fs::temp_directory_path() / ("strata-tier-striped/tier-" + std::to_string(std::rand()));
        fs::create_directories(path);
    }
    ~Dir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

static std::string make_device(const Dir& d, const std::string& name, uint64_t mib) {
    const fs::path p = d.path / name;
    std::ofstream out(p, std::ios::binary);
    out << std::string(1, '\0');
    out.close();
    fs::resize_file(p, mib << 20);
    return p.string();
}

static std::vector<TierSpec> specs_of(const std::vector<std::string>& paths, uint64_t mib) {
    std::vector<TierSpec> s;
    for (const std::string& p : paths) s.push_back({p, mib, 0, false});
    return s;
}

// Every (layer, expert) once, in an order that is not the layer order, so the device deal crosses layers.
static std::vector<std::pair<int32_t, int32_t>> ranked_all() {
    std::vector<std::pair<int32_t, int32_t>> r;
    for (int64_t round = 0; round < kExperts; ++round)
        for (int64_t l = 0; l < kLayers; ++l) r.push_back({(int32_t) l, (int32_t) ((round + l) % kExperts)});
    return r;
}

static TierIdentity identity_for(const std::vector<uint64_t>& blobs) {
    TierIdentity id;
    id.pack_id = "st:34123456789:1730430000";
    const std::vector<uint8_t> profile = {'S', 'T', 'R', 'P', 7};
    id.profile_id = "sha:" + tier_sha256_hex(profile.data(), profile.size());
    id.n_layers = kLayers;
    id.n_expert = kExperts;
    id.layer_blob_bytes = blobs;
    return id;
}

// Every held expert's gathered bytes equal the truth, and held() agrees with what the fill placed.
static void check_gather(const ExpertTierSet& set, const Source& truth, const std::string& what) {
    std::vector<uint8_t> dst;
    int64_t held = 0;
    for (int64_t l = 0; l < kLayers; ++l)
        for (int64_t e = 0; e < kExperts; ++e) {
            if (!set.held(l, e)) continue;
            ++held;
            dst.assign((size_t) truth.bytes[(size_t) l], 0xAB);
            require(set.gather(l, e, dst.data()), what + ": gather refused a held expert");
            require(dst == truth.store[(size_t) (l * kExperts + e)], what + ": gathered bytes differ from the blob");
        }
    require(held == set.experts_held(), what + ": held() disagrees with experts_held()");
}

// The first fill, then a restart that reuses it: stripe unit and device count vary; all bytes are compared with the truth.
static void test_gather_matches_blobs(uint64_t stripe_kib, size_t n_devices, uint64_t mib = 32) {
    Dir dir;
    std::vector<std::string> paths;
    for (size_t d = 0; d < n_devices; ++d) paths.push_back(make_device(dir, "d" + std::to_string(d), mib));
    const std::vector<TierSpec> specs = specs_of(paths, mib);
    const TierIdentity id = identity_for(kBlobs);
    const auto ranked = ranked_all();
    const std::string what = "stripe " + std::to_string(stripe_kib) + " KiB, " + std::to_string(n_devices) + " devices";

    Source truth(kBlobs);
    ExpertTierSet set;
    std::string err;
    bool reused = true;
    set.set_layout(true, stripe_kib * 1024);
    require(set.open(specs, id, err, reused), what + ": open: " + err);
    require(!reused, what + ": fresh devices were reused");
    require(set.fill(truth, ranked, id, err), what + ": fill: " + err);
    require(set.experts_held() == (int64_t) ranked.size(), what + ": not every expert fitted (room exists)");
    require(truth.copies == (int) ranked.size(), what + ": the fill did not read each blob exactly once");
    for (const auto& pr : ranked) {
        require(set.held(pr.first, pr.second), what + ": a placed expert is not held");
        require(!set.has(pr.first, pr.second), what + ": a striped expert must have no contiguous pointer");
        require(set.blob(pr.first, pr.second) == nullptr, what + ": blob() must be null for a striped expert");
    }
    check_gather(set, truth, what + " (fill)");

    // a batch in reverse rank order, split over the devices with 1 and then 4 workers per device: the same bytes
    for (int cap : {1, 4}) {
        set.set_worker_cap(cap);
        std::vector<int32_t> layers, experts;
        std::vector<std::vector<uint8_t>> bufs;
        for (auto it = ranked.rbegin(); it != ranked.rend(); ++it) {
            layers.push_back(it->first);
            experts.push_back(it->second);
            bufs.emplace_back((size_t) truth.bytes[(size_t) it->first], 0xCD);
        }
        std::vector<uint8_t*> dst;
        for (auto& b : bufs) dst.push_back(b.data());
        uint64_t bytes = 0;
        require(set.copy_blobs(layers.data(), experts.data(), dst.data(), layers.size(), bytes), what + ": copy_blobs refused");
        uint64_t want = 0;
        for (size_t i = 0; i < layers.size(); ++i) {
            want += truth.bytes[(size_t) layers[i]];
            require(bufs[i] == truth.store[(size_t) (layers[i] * kExperts + experts[i])],
                    what + ": copy_blobs bytes differ (workers " + std::to_string(cap) + ")");
        }
        require(bytes == want, what + ": copy_blobs byte count wrong");
    }

    // a restart reuses the stripes: nothing is copied, and the gathered bytes are still the truth
    set.close();
    Source again(kBlobs);
    ExpertTierSet set2;
    bool reused2 = false;
    set2.set_layout(true, stripe_kib * 1024);
    require(set2.open(specs, id, err, reused2), what + ": reopen: " + err);
    require(reused2, what + ": a restart did not reuse the striped placement");
    check_gather(set2, truth, what + " (reuse)");
    require(again.copies == 0, what + ": reuse copied blobs");
    set2.close();
}

// Blobs no longer than one stripe (one device each) and odd sizes around the 4 KiB alignment.
static void test_small_and_odd_blobs() {
    const std::vector<uint64_t> blobs = {1, 4095, 4096, 65537};
    Dir dir;
    const std::vector<TierSpec> specs = specs_of({make_device(dir, "a", 8), make_device(dir, "b", 8), make_device(dir, "c", 8)}, 8);
    const TierIdentity id = identity_for(blobs);
    const auto ranked = ranked_all();
    Source truth(blobs, 9);
    ExpertTierSet set;
    std::string err;
    bool reused = true;
    set.set_layout(true, 64 * 1024);
    require(set.open(specs, id, err, reused), "small blobs: open: " + err);
    require(set.fill(truth, ranked, id, err), "small blobs: fill: " + err);
    require(set.experts_held() == (int64_t) ranked.size(), "small blobs: not every expert fitted");
    check_gather(set, truth, "small and odd blobs");
    set.close();
}

// The layout, the stripe unit and the device order are the placement: any of them changing refills the set.
static void test_layout_and_stripe_changes_refill() {
    Dir dir;
    const std::vector<std::string> paths = {make_device(dir, "a", 32), make_device(dir, "b", 32)};
    const std::vector<TierSpec> specs = specs_of(paths, 32);
    const TierIdentity id = identity_for(kBlobs);
    const auto ranked = ranked_all();
    Source truth(kBlobs);
    std::string err;
    bool reused = true;

    // whole first, then striped over it: refilled (never read as stripes)
    ExpertTierSet whole;
    require(whole.open(specs, id, err, reused), err);
    require(whole.fill(truth, ranked, id, err), err);
    whole.close();
    ExpertTierSet striped;
    striped.set_layout(true, 64 * 1024);
    require(striped.open(specs, id, err, reused), err);
    require(!reused, "a whole tier was reused as striped");
    bool said = false;
    for (const auto& n : striped.notes()) said = said || n.find("written in the whole layout") != std::string::npos;
    require(said, "the whole -> striped refill was not noted");
    require(striped.fill(truth, ranked, id, err), err);
    striped.close();

    // striped with another stripe unit: refilled
    ExpertTierSet other_stripe;
    other_stripe.set_layout(true, 256 * 1024);
    require(other_stripe.open(specs, id, err, reused), err);
    require(!reused, "a stripe unit change was reused");
    said = false;
    for (const auto& n : other_stripe.notes()) said = said || n.find("stripe unit") != std::string::npos;
    require(said, "the stripe unit change was not noted");
    require(other_stripe.fill(truth, ranked, id, err), err);
    other_stripe.close();

    // the same stripes with the devices in the other order: refilled (the order is part of the placement, §6.3)
    ExpertTierSet swapped;
    swapped.set_layout(true, 256 * 1024);
    require(swapped.open({specs[1], specs[0]}, id, err, reused), err);
    require(!reused, "a device order change was reused");
    swapped.close();

    // striped -> whole: refilled, and the whole layout reads its own tier again
    ExpertTierSet back_whole;
    require(back_whole.open(specs, id, err, reused), err);
    require(!reused, "a striped tier was reused as whole");
    said = false;
    for (const auto& n : back_whole.notes()) said = said || n.find("striped layout") != std::string::npos;
    require(said, "the striped -> whole refill was not noted");
    require(back_whole.fill(truth, ranked, id, err), err);
    back_whole.close();
    ExpertTierSet whole_again;
    require(whole_again.open(specs, id, err, reused), err);
    require(reused, "a whole tier written by this engine did not reuse");
    whole_again.close();
}

// The whole layout is the default and keeps its behaviour: has() is the pointer, held() agrees, gather() is the blob.
static void test_whole_layout_unchanged() {
    Dir dir;
    const std::vector<TierSpec> specs = specs_of({make_device(dir, "a", 16), make_device(dir, "b", 16)}, 16);
    const TierIdentity id = identity_for(kBlobs);
    const auto ranked = ranked_all();
    Source truth(kBlobs);
    ExpertTierSet set;
    std::string err;
    bool reused = true;
    require(set.open(specs, id, err, reused), err);
    require(!set.striped(), "the whole layout is not the default");
    require(set.fill(truth, ranked, id, err), err);
    for (const auto& pr : ranked) {
        require(set.has(pr.first, pr.second) == set.held(pr.first, pr.second), "whole: has() and held() disagree");
        require(set.has(pr.first, pr.second), "whole: a placed expert has no pointer");
    }
    check_gather(set, truth, "whole layout");
    set.close();
}

// verify=full: a flipped byte in one stripe is found at the next start; every expert with a piece in that chunk is served from the
// files (not held), the device is marked for refill, and nothing wrong is ever gathered.
static void test_verify_full_catches_a_bad_stripe() {
    Dir dir;
    const std::vector<std::string> paths = {make_device(dir, "a", 32), make_device(dir, "b", 32)};
    std::vector<TierSpec> specs = specs_of(paths, 32);
    const TierIdentity id = identity_for(kBlobs);
    const auto ranked = ranked_all();
    Source truth(kBlobs);
    std::string err;
    bool reused = true;
    ExpertTierSet s1;
    s1.set_layout(true, 64 * 1024);
    require(s1.open(specs, id, err, reused), err);
    require(s1.fill(truth, ranked, id, err), err);
    s1.close();

    // Expert ranked[0] is placed with start device 0 (the first placed expert), so device 1 holds its stripes 1, 3, 5, ...: its
    // piece on device 1 begins with blob bytes [64 KiB, 64 KiB + 64). Flip a byte of that piece on disk.
    const std::vector<uint8_t>& blob0 = truth.store[(size_t) (ranked[0].first * kExperts + ranked[0].second)];
    const size_t stripe = 64 * 1024;
    std::vector<uint8_t> probe(blob0.begin() + (std::ptrdiff_t) stripe, blob0.begin() + (std::ptrdiff_t) stripe + 64);
    std::vector<uint8_t> file;
    {
        std::ifstream in(paths[1], std::ios::binary);
        file.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    size_t at = 0;
    bool found = false;
    for (size_t cand = 0; cand + 64 <= file.size(); cand += 4096)
        if (std::equal(probe.begin(), probe.end(), file.begin() + (std::ptrdiff_t) cand)) { at = cand; found = true; break; }
    require(found, "the stripe's bytes were not found on device 1");
    file[at + 10] ^= 0xFF;
    {
        std::ofstream out(paths[1], std::ios::binary | std::ios::trunc);
        out.write((const char*) file.data(), (std::streamsize) file.size());
    }

    specs[0].verify_full = true;
    specs[1].verify_full = true;
    ExpertTierSet s2;
    s2.set_layout(true, 64 * 1024);
    require(s2.open(specs, id, err, reused), err);
    require(reused, "verify=full refilled by itself");
    require(!s2.held(ranked[0].first, ranked[0].second), "an expert with a bad stripe is still held");
    bool noted = false;
    for (const auto& n : s2.notes()) noted = noted || n.find("CRC mismatch") != std::string::npos;
    require(noted, "the bad stripe was not noted");
    int64_t held_left = 0;
    for (const auto& pr : ranked) held_left += s2.held(pr.first, pr.second) ? 1 : 0;
    require(held_left > 0 && held_left < (int64_t) ranked.size(), "a bad chunk should invalidate some experts, not all or none");
    check_gather(s2, truth, "verify=full after a bad stripe");   // every expert still held is exact
    s2.close();

    ExpertTierSet s3;
    s3.set_layout(true, 64 * 1024);
    bool reused3 = true;
    require(s3.open(specs, id, err, reused3), err);
    require(!reused3, "the mark-for-refill did not survive the close");
    s3.close();
}

// A fill whose flush fails on one device leaves every header unfinished: the next start refills the whole set.
static void test_torn_striped_fill_refills() {
    Dir dir;
    const std::vector<TierSpec> specs = specs_of({make_device(dir, "a", 32), make_device(dir, "b", 32)}, 32);
    const TierIdentity id = identity_for(kBlobs);
    const auto ranked = ranked_all();
    Source truth(kBlobs);
    std::string err;
    bool reused = true;
    ExpertTierSet torn;
    torn.set_layout(true, 64 * 1024);
    require(torn.open(specs, id, err, reused), err);
    torn.set_flush_fault([](size_t d) { return d == 1; });
    require(!torn.fill(truth, ranked, id, err), "a fill whose flush failed reported success");
    torn.close();
    ExpertTierSet again;
    again.set_layout(true, 64 * 1024);
    require(again.open(specs, id, err, reused), err);
    require(!reused, "a torn striped fill was reused");
    again.close();
}

// Unequal devices: the smallest device limits the deal (no device is dropped, the rest of the larger one stays unused). The
// experts placed are a prefix of the ranked list, and the report says how much is unused.
static void test_unequal_devices_prefix_and_report() {
    Dir dir;
    const std::vector<TierSpec> specs = {{make_device(dir, "big", 32), 32, 0, false}, {make_device(dir, "small", 4), 4, 0, false}};
    const TierIdentity id = identity_for(kBlobs);
    const auto ranked = ranked_all();
    Source truth(kBlobs);
    ExpertTierSet set;
    std::string err;
    bool reused = true;
    set.set_layout(true, 64 * 1024);
    require(set.open(specs, id, err, reused), err);
    require(set.fill(truth, ranked, id, err), err);
    const int64_t placed = set.experts_held();
    require(placed > 0 && placed < (int64_t) ranked.size(), "the small device should stop the deal partway");
    for (size_t i = 0; i < ranked.size(); ++i)
        require(set.held(ranked[i].first, ranked[i].second) == ((int64_t) i < placed),
                "the placed experts are not a prefix of the ranked list");
    bool unused = false;
    for (const auto& n : set.notes()) unused = unused || n.find("unused") != std::string::npos;
    require(unused, "the unused part of the larger device was not reported");
    check_gather(set, truth, "unequal devices");
    set.close();
}

// The arguments the layout refuses before any device is written: one device, a stripe that is not a 4 KiB multiple.
static void test_bad_layouts_refused() {
    Dir dir;
    const TierIdentity id = identity_for(kBlobs);
    std::string err;
    bool reused = false;
    {
        ExpertTierSet one;
        one.set_layout(true, 64 * 1024);
        require(!one.open(specs_of({make_device(dir, "solo", 8)}, 8), id, err, reused), "striped over one device was accepted");
        require(err.find("2 to 8 devices") != std::string::npos, "the one-device refusal does not say why: " + err);
    }
    {
        ExpertTierSet odd;
        odd.set_layout(true, 4097);
        require(!odd.open(specs_of({make_device(dir, "p", 8), make_device(dir, "q", 8)}, 8), id, err, reused),
                "a stripe that is not a 4 KiB multiple was accepted");
    }
}

// The whole layout's verify=full: a bad chunk drops the experts it touches, and the held counts drop with them (the startup
// log's shares read them). Only the counts change on this default path; the other experts' bytes are checked as before.
static void test_whole_verify_full_counts() {
    Dir dir;
    const std::vector<std::string> paths = {make_device(dir, "a", 32), make_device(dir, "b", 32)};
    std::vector<TierSpec> specs = specs_of(paths, 32);
    const TierIdentity id = identity_for(kBlobs);
    const auto ranked = ranked_all();
    Source truth(kBlobs);
    std::string err;
    bool reused = true;
    ExpertTierSet s1;
    require(s1.open(specs, id, err, reused), err);
    require(s1.fill(truth, ranked, id, err), err);
    require(s1.experts_held() == (int64_t) ranked.size(), "whole: every expert should fit 2 x 32 MiB");
    s1.close();

    // flip one byte of ranked[0]'s blob, wherever the whole layout put it
    const std::vector<uint8_t>& blob0 = truth.store[(size_t) (ranked[0].first * kExperts + ranked[0].second)];
    const std::vector<uint8_t> probe(blob0.begin(), blob0.begin() + 64);
    bool flipped = false;
    for (const std::string& p : paths) {
        std::vector<uint8_t> file;
        {
            std::ifstream in(p, std::ios::binary);
            file.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        for (size_t cand = 0; cand + 64 <= file.size() && !flipped; ++cand)
            if (std::equal(probe.begin(), probe.end(), file.begin() + (std::ptrdiff_t) cand)) {
                file[cand + 10] ^= 0xFF;
                std::ofstream out(p, std::ios::binary | std::ios::trunc);
                out.write((const char*) file.data(), (std::streamsize) file.size());
                flipped = true;
            }
        if (flipped) break;
    }
    require(flipped, "the blob's bytes were not found on either device");

    specs[0].verify_full = true;
    specs[1].verify_full = true;
    ExpertTierSet s2;
    require(s2.open(specs, id, err, reused), err);
    require(reused, "whole: verify=full refilled by itself");
    require(!s2.held(ranked[0].first, ranked[0].second), "whole: an expert with a bad chunk is still held");
    int64_t held = 0;
    uint64_t bytes = 0;
    for (const auto& pr : ranked)
        if (s2.held(pr.first, pr.second)) {
            ++held;
            bytes += kBlobs[(size_t) pr.first];
        }
    require(held < (int64_t) ranked.size(), "whole: the bad chunk should invalidate an expert");
    require(s2.experts_held() == held, "whole: experts_held() still counts the invalidated experts");
    require(s2.bytes_held() == bytes, "whole: bytes_held() still counts the invalidated experts");
    check_gather(s2, truth, "whole verify=full after a bad chunk");
}

int main() {
    std::srand(20261123);
    try {
        test_gather_matches_blobs(4, 2);     // 4 KiB stripes: the smallest unit
        test_gather_matches_blobs(64, 2);
        test_gather_matches_blobs(256, 3);   // three devices, a stripe that is smaller than the 1 MiB blobs' quarter
        test_small_and_odd_blobs();
        test_layout_and_stripe_changes_refill();
        test_whole_layout_unchanged();
        test_verify_full_catches_a_bad_stripe();
        test_whole_verify_full_counts();
        test_torn_striped_fill_refills();
        test_unequal_devices_prefix_and_report();
        test_bad_layouts_refused();
        std::cout << "expert_tier_striped_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "expert_tier_striped_test: " << error.what() << '\n';
        return 1;
    }
}
