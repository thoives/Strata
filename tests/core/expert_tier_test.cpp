// tests/core/expert_tier_test.cpp - §12.3 of docs/EXPERT_TIER_PLAN.md: the whole persistence story on /tmp.
// A file on a tmpfs is RAM-backed, so mapping/header/identity/fill/reuse/torn-fill/verify are all valid here (§10);
// only timing and RAM-budget claims are not, and this test makes none.
#include "strata/core/expert_tier.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/expert_tier_dev.hpp"
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::core;

// The out-of-line ExpertSource members this test's fake source still needs (the fake overrides them; the vtable
// references the base definitions, and this test does not link src/core/expert_source.cpp: no CUDA here).
namespace strata::core {
bool ExpertSource::copy_blob(int64_t, int64_t, uint8_t*) { return false; }
bool ExpertSource::copy_blobs(const int32_t*, const int32_t*, uint8_t* const*, size_t) { return true; }
}

static void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}

namespace fs = std::filesystem;

static constexpr int64_t kLayers = 4, kExperts = 8;
static const uint64_t kBlobs[kLayers] = {256u << 10, 256u << 10, 1u << 20, 1u << 20};   // two blob sizes (§3)

static uint8_t truth_byte(int64_t l, int64_t e, size_t i) {
    return (uint8_t) ((i * 131u + (size_t) (l * kExperts + e) * 37u) ^ (i >> 8));
}

struct FakeFileTier : ExpertSource {
    std::vector<std::vector<uint8_t>> store;
    int copies = 0;
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
        ++copies;
        const std::vector<uint8_t>& t = store[(size_t) layer * kExperts + expert];
        std::memcpy(dst, t.data(), t.size());
        return true;
    }
};

struct Dir {
    fs::path path;
    explicit Dir() {
        path = fs::temp_directory_path() / ("strata-tier/tier-" + std::to_string(rand()));   // §10: /tmp/strata-tier/
        fs::create_directories(path);
    }
    ~Dir() { std::error_code ec; fs::remove_all(path, ec); }
};

// make a device file of `mib` MiB
static fs::path make_device(Dir& d, const std::string& name, uint64_t mib) {
    fs::path p = d.path / name;
    std::ofstream out(p, std::ios::binary);
    out << std::string(1, '\0');
    out.close();
    fs::resize_file(p, mib << 20);
    return p;
}

static std::vector<std::pair<int32_t, int32_t>> ranked_all_hot() {
    // hottest first, RAM kept none here: the §4.1 band order is the caller's; this test hands the whole list over
    std::vector<std::pair<int32_t, int32_t>> r;
    for (int64_t e = 0; e < kExperts; ++e) r.push_back({(int32_t) (e % kLayers), (int32_t) e});
    return r;
}

static TierIdentity identity_for(const std::vector<uint8_t>& profile_bytes, const std::vector<uint64_t>& blobs) {
    TierIdentity id;
    id.pack_id = "st:34123456789:1730430000";
    id.profile_id = "sha:" + tier_sha256_hex(profile_bytes.data(), profile_bytes.size());
    id.n_layers = kLayers;
    id.n_expert = kExperts;
    id.layer_blob_bytes = blobs;
    return id;
}


// §12.3 open, map, fill and byte-compare every held blob against the file tier
static void test_fill_and_bytes() {
    Dir dir;
    const std::vector<uint64_t> blobs = {256u << 10, 256u << 10, 1u << 20, 1u << 20};
    const std::vector<uint8_t> profile = {'S', 'T', 'R', 'P', 1, 2, 3};
    TierIdentity id = identity_for(profile, blobs);
    const std::vector<TierSpec> specs = {{make_device(dir, "a", 8).string(), 8, 0, false},
                                         {make_device(dir, "b", 8).string(), 8, 0, false}};
    FakeFileTier file_tier;
    ExpertTierSet set;
    std::string err;
    bool reused = true;
    require(set.open(specs, id, err, reused), err);
    require(!reused, "fresh devices were " "reused");
    const std::vector<std::pair<int32_t, int32_t>> r = ranked_all_hot();
    require(set.fill(file_tier, r, id, err), err);
    require(set.experts_held() == (int64_t) r.size(), "not every ranked expert was placed (room exists)");
    uint64_t want_bytes = 0;
    for (const auto& pr : r) want_bytes += kBlobs[pr.first];
    require(set.bytes_held() == want_bytes, "bytes_held wrong");
    require(file_tier.copies == (int) r.size(), "the fill did not read exactly the placed blobs once each");
    require(set.fill_seconds() >= 0.0 && set.fill_bytes() == set.bytes_held(), "fill accounting off");

    std::vector<uint8_t> dst(1u << 20);
    for (const auto& pr : r) {
        require(set.has(pr.first, pr.second), "a placed expert is not held");
        const uint8_t* p = set.blob(pr.first, pr.second);
        const std::vector<uint8_t>& t = file_tier.store[(size_t) pr.first * kExperts + pr.second];
        std::memset(dst.data(), 0, t.size());
        std::memcpy(dst.data(), p, t.size());
        require(std::equal(t.begin(), t.end(), dst.begin()), "the tier serves wrong bytes");
    }
    require(!set.has(3, 4), "an expert nobody placed entered the tier");

    // reuse: a second open with the same identity reuses and copies nothing (§6.3). The first process has ended (its devices are
    // closed: a tier is held by one process at a time, §5.4), so the restart opens them again.
    set.close();
    FakeFileTier second;
    ExpertTierSet set2;
    bool reused2 = false;
    require(set2.open(specs, id, err, reused2), err);
    require(reused2, "a restart did not reuse the placement");
    require(second.copies == 0, "reuse copied blobs");
    for (const auto& pr : r) {
        const uint8_t* p = set2.blob(pr.first, pr.second);
        const std::vector<uint8_t>& t = second.store[(size_t) pr.first * kExperts + pr.second];
        require(p != nullptr && std::equal(t.begin(), t.end(), p), "reused tier serves wrong bytes");
    }
    set2.close();
    require(!set2.has(0, 0), "has() true after close()");
}

static void test_identity_changes_refill() {
    Dir dir;
    const std::vector<uint64_t> blobs = {256u << 10, 256u << 10, 1u << 20, 1u << 20};
    std::vector<TierSpec> specs = {{make_device(dir, "a", 8).string(), 8, 0, false},
                                   {make_device(dir, "b", 8).string(), 8, 0, false}};
    FakeFileTier file_tier;
    const std::vector<std::pair<int32_t, int32_t>> r = ranked_all_hot();

    TierIdentity a = identity_for({'x', 'y'}, blobs);
    ExpertTierSet s1;
    std::string err;
    bool reused = true;
    require(s1.open(specs, a, err, reused), err);
    require(s1.fill(file_tier, r, a, err), err);
    s1.close();

    // profile bytes changed -> the profile id changed -> refill
    TierIdentity b = identity_for({'x', 'z'}, blobs);
    ExpertTierSet s2;
    require(s2.open(specs, b, err, reused), err);
    require(!reused, "a changed profile did not force a refill");
    s2.close();

    // the same identity with the device order swapped: device_index no longer matches -> refill (§6.2)
    std::swap(specs[0], specs[1]);
    ExpertTierSet s3;
    require(s3.open(specs, a, err, reused), err);
    require(!reused, "device order is part of the identity (§6.3)");
    s3.close();
    std::swap(specs[0], specs[1]);

    // a capacity change (mib) -> refill
    specs[0].mib = 12;
    fs::resize_file(specs[0].path, 12 << 20);
    ExpertTierSet s4;
    require(s4.open(specs, a, err, reused), err);
    require(!reused, "a capacity change did not force a refill");
    s4.close();
}

static void test_torn_fill_refills() {
    Dir dir;
    const std::vector<uint64_t> blobs = {256u << 10, 256u << 10, 1u << 20, 1u << 20};
    std::vector<TierSpec> specs = {{make_device(dir, "a", 8).string(), 8, 0, false}};
    FakeFileTier file_tier;
    const std::vector<std::pair<int32_t, int32_t>> r = ranked_all_hot();
    TierIdentity id = identity_for({'q'}, blobs);
    ExpertTierSet s1;
    std::string err;
    bool reused = true;
    require(s1.open(specs, id, err, reused), err);
    require(s1.fill(file_tier, r, id, err), err);
    s1.close();

    // simulate a fill interrupted at its very end: fill_state back to 1 (header A, its CRC kept true)
    {
        std::fstream f(specs[0].path, std::ios::binary | std::ios::in | std::ios::out);
        uint8_t buf[160];
        f.read((char*) buf, 160);
        TierHeader h{};
        std::string why;
        require(decode_header(buf, h, why), why);
        h.fill_state = 1;
        encode_header(h, buf);
        f.seekg(0);
        f.write((const char*) buf, 160);
    }
    ExpertTierSet s2;
    require(s2.open(specs, id, err, reused), err);
    require(!reused, "a device found at fill_state == 1 was not refilled (§6.3)");
    s2.close();
}

static void test_verify_full_and_refill_mark() {
    Dir dir;
    const std::vector<uint64_t> blobs = {256u << 10, 256u << 10, 1u << 20, 1u << 20};
    std::vector<TierSpec> specs = {{make_device(dir, "a", 16).string(), 16, 0, false},
                                   {make_device(dir, "b", 16).string(), 16, 0, false}};
    FakeFileTier file_tier;
    const std::vector<std::pair<int32_t, int32_t>> r = ranked_all_hot();
    TierIdentity id = identity_for({'p'}, blobs);
    ExpertTierSet s1;
    std::string err;
    bool reused = true;
    require(s1.open(specs, id, err, reused), err);
    require(s1.fill(file_tier, r, id, err), err);
    s1.close();

    // flip one byte inside one blob, on disk, through a fresh mapping of the file
    const int64_t flip_l = 2, flip_e = 2;
    {
        std::fstream f(specs[0].path, std::ios::binary | std::ios::in | std::ios::out);
        const std::vector<uint8_t>& t = file_tier.store[(size_t) flip_l * kExperts + flip_e];
        // find where this blob's bytes sit: the first 4 KiB-aligned run matching the first 64 bytes
        std::vector<uint8_t> file_data;
        f.read((char*) file_data.data(), 0);
        file_data.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const uint8_t* probe = t.data();
        size_t off = 0;
        bool found = false;
        for (size_t cand = 0; cand + 64 <= file_data.size(); cand += 4096) {
            if (std::equal(probe, probe + 64, file_data.begin() + cand)) { off = cand; found = true; break; }
        }
        require(found, "the blob's bytes were not found on the device");
        file_data[off + 10] ^= 0xFF;
        f.close();
        std::ofstream w(specs[0].path, std::ios::binary);
        w.write((const char*) file_data.data(), (std::streamsize) file_data.size());
    }

    // verify=full at the next start: the chunk's CRC mismatches, the blob falls back to the file tier, the chunk
    // is logged, and the device is marked for refill at the start after that (§6.4).
    std::vector<TierSpec> vspecs = specs;
    for (auto& s : vspecs) s.verify_full = true;
    ExpertTierSet s2;
    require(s2.open(vspecs, id, err, reused), err);
    require(reused, "verify=full does not refill by itself");
    require(!s2.has(flip_l, flip_e), "a blob whose chunk CRC failed is still served from the tier");
    require(s2.blob(flip_l, flip_e) == nullptr, "blob() must be null for the untrustworthy copy");
    bool noted = false;
    for (const auto& note : s2.notes())
        if (note.find("CRC mismatch") != std::string::npos) noted = true;
    require(noted, "the chunk failure was not noted (§8.5: the engine logs notes)");
    s2.close();

    ExpertTierSet s3;
    bool reused3 = true;
    require(s3.open(specs, id, err, reused3), err);
    require(!reused3, "the mark-for-refill (§6.4) did not survive the close");
    s3.close();
}

static void test_device_too_small_and_duplicate() {
    Dir dir;
    std::vector<TierSpec> specs = {{make_device(dir, "a", 1).string(), 8, 0, false}};   // file 1 MiB, mib 8
    FakeFileTier file_tier;
    TierIdentity id = identity_for({'z'}, std::vector<uint64_t>(kLayers, 256u << 10));
    ExpertTierSet s;
    std::string err;
    bool reused = true;
    require(!s.open(specs, id, err, reused), "a too-small device opened");
    require(err.find("mib") != std::string::npos, "the size refusal does not name mib");
    s.close();

    const fs::path a = make_device(dir, "dup1", 8);
    std::vector<TierSpec> dup = {{a.string(), 8, 0, false}, {a.string(), 8, 0, false}};
    ExpertTierSet s2;
    require(!s2.open(dup, id, err, reused), "a duplicate --expert-tier path was accepted");
    s2.close();
}

static void test_sha_is_sha256() {
    // §6.2 uses SHA-256; the pinned vector from the standard keeps the runtime-computed constants honest.
    require(tier_sha256_hex((const uint8_t*) "", 0) ==
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "SHA-256 of the empty string");
    require(tier_sha256_hex((const uint8_t*) "abc", 3).size() == 64, "SHA-256 length");
}

// every expert of the synthetic shape, in layer-major order
static std::vector<std::pair<int32_t, int32_t>> ranked_everything() {
    std::vector<std::pair<int32_t, int32_t>> r;
    for (int32_t l = 0; l < kLayers; ++l)
        for (int32_t e = 0; e < kExperts; ++e) r.push_back({l, e});
    return r;
}

// where blob (l, e)'s first 64 bytes sit on the device file (a 4 KiB aligned position), or -1
static int64_t find_on_device(const std::vector<uint8_t>& file_data, const std::vector<uint8_t>& blob) {
    for (size_t cand = 0; cand + 64 <= file_data.size(); cand += 4096)
        if (std::equal(blob.begin(), blob.begin() + 64, file_data.begin() + (std::ptrdiff_t) cand)) return (int64_t) cand;
    return -1;
}

// §6.4 verify=full keeps EVERY bad chunk, not the last one found: two blobs in two different chunks are both
// served from the file tier for this run, and a blob in a third (good) chunk is still the tier's.
static void test_verify_full_keeps_every_bad_chunk() {
    Dir dir;
    const std::vector<uint64_t> blobs = {256u << 10, 256u << 10, 1u << 20, 1u << 20};
    std::vector<TierSpec> specs = {{make_device(dir, "a", 32).string(), 32, 0, false}};
    FakeFileTier file_tier;
    const std::vector<std::pair<int32_t, int32_t>> r = ranked_everything();
    TierIdentity id = identity_for({'v'}, blobs);
    ExpertTierSet s1;
    std::string err;
    bool reused = true;
    require(s1.open(specs, id, err, reused), err);
    require(s1.fill(file_tier, r, id, err), err);
    require(s1.experts_held() == (int64_t) r.size(), "the 32 MiB device should hold every expert");
    s1.close();

    std::vector<uint8_t> data;
    {
        std::ifstream f(specs[0].path, std::ios::binary);
        data.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }
    // the first placed blob (rank 0), the last one, and one in the middle: three distinct chunks of the 5 the data spans
    const std::vector<std::pair<int32_t, int32_t>> victims = {r.front(), r.back()};
    const std::pair<int32_t, int32_t> middle = {2, 0};
    std::vector<int64_t> offs;
    for (const auto& v : victims) {
        const int64_t off = find_on_device(data, file_tier.store[(size_t) v.first * kExperts + v.second]);
        require(off >= 0, "a victim's bytes were not found on the device");
        offs.push_back(off);
    }
    const int64_t mid_off = find_on_device(data, file_tier.store[(size_t) middle.first * kExperts + middle.second]);
    require(mid_off >= 0, "the control blob was not found on the device");
    require(offs[1] - offs[0] > 2 * (int64_t) kTierChunk, "the two victims should sit in different chunks");
    for (int64_t off : offs) data[(size_t) off + 10] ^= 0xFF;
    {
        std::ofstream w(specs[0].path, std::ios::binary);
        w.write((const char*) data.data(), (std::streamsize) data.size());
    }
    // the control blob must not share a chunk with either victim for the test to mean anything
    const auto chunk_of = [](int64_t off) { return off / (int64_t) kTierChunk; };
    require(chunk_of(mid_off) != chunk_of(offs[0]) && chunk_of(mid_off) != chunk_of(offs[1]) &&
            chunk_of(mid_off + (int64_t) kBlobs[middle.first] - 1) != chunk_of(offs[1]),
            "the control blob shares a chunk with a victim: the layout changed, so pick another control");

    std::vector<TierSpec> vspecs = specs;
    vspecs[0].verify_full = true;
    ExpertTierSet s2;
    require(s2.open(vspecs, id, err, reused), err);
    require(reused, "verify=full does not refill by itself");
    for (const auto& v : victims) require(!s2.has(v.first, v.second), "a blob in a bad chunk is still served from the tier");
    require(s2.has(middle.first, middle.second), "a blob in a good chunk was dropped");
    // each served-from-files blob is the file tier's, through the source's own order (the tier says "not held")
    int bad_notes = 0;
    for (const auto& note : s2.notes())
        if (note.find("CRC mismatch") != std::string::npos) ++bad_notes;
    require(bad_notes >= 1, "the chunk failures were not noted");
    s2.close();
    ExpertTierSet s3;
    require(s3.open(specs, id, err, reused), err);
    require(!reused, "the device with bad chunks was not marked for refill");
    s3.close();
}

// §5.2 / §4.2: the reserve is never written, and a blob never starts inside the aligned gap past what the device may hold
static void test_reserve_is_never_written() {
    Dir dir;
    const std::vector<uint64_t> blobs = {256u << 10, 256u << 10, 1u << 20, 1u << 20};
    std::vector<TierSpec> specs = {{make_device(dir, "a", 8).string(), 8, 2, false}};   // 8 MiB device, 2 MiB reserve
    FakeFileTier file_tier;
    TierIdentity id = identity_for({'r'}, blobs);
    ExpertTierSet s1;
    std::string err;
    bool reused = true;
    require(s1.open(specs, id, err, reused), err);
    require(s1.fill(file_tier, ranked_everything(), id, err), err);
    require(s1.experts_held() > 0 && s1.experts_held() < kLayers * kExperts, "an 8 MiB device cannot hold all 32 experts");
    s1.close();
    std::ifstream f(specs[0].path, std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    require(data.size() == (8u << 20), "the device file changed size");
    const size_t cap = 6u << 20;
    for (size_t i = cap; i < data.size(); ++i)
        if (data[i] != 0) throw std::runtime_error("a byte was written inside the reserve, at offset " + std::to_string(i));
    // and the placement is reused by the next start with the same configuration
    ExpertTierSet s2;
    require(s2.open(specs, id, err, reused), err);
    require(reused, "a reserve-limited fill was not reusable");
    s2.close();
}

// A placement that CRC-checks but lies about the device (here: a blob that would end past its capacity) is refilled,
// not mapped out of range.
static void test_reuse_refuses_a_placement_outside_the_device() {
    Dir dir;
    const std::vector<uint64_t> blobs = {256u << 10, 256u << 10, 1u << 20, 1u << 20};
    std::vector<TierSpec> specs = {{make_device(dir, "a", 8).string(), 8, 0, false}};
    FakeFileTier file_tier;
    TierIdentity id = identity_for({'o'}, blobs);
    ExpertTierSet s1;
    std::string err;
    bool reused = true;
    require(s1.open(specs, id, err, reused), err);
    require(s1.fill(file_tier, ranked_all_hot(), id, err), err);
    s1.close();
    // the same device opened by a start whose blobs are larger than the stored placement's: the sizes are part of the
    // layout (checked through the pack), and a reuse that would run off the device must not happen
    TierIdentity big = id;
    big.layer_blob_bytes = {6u << 20, 6u << 20, 6u << 20, 6u << 20};
    ExpertTierSet s2;
    require(s2.open(specs, big, err, reused), err);
    require(!reused, "a placement that no longer fits the device was reused");
    s2.close();
}

// ---- the pack identity covers every file: the aggregate "total size + newest time" cannot see these changes
static void write_file(const fs::path& p, const std::string& body) {
    std::ofstream o(p, std::ios::binary);
    o << body;
}
static void test_pack_identity_is_per_file() {
    Dir dir;
    const fs::path a = dir.path / "experts-a.bin", b = dir.path / "experts-b.bin", prof = dir.path / "profile.txt";
    write_file(a, "AAAA");
    write_file(b, "BBBB");
    write_file(prof, "p1");
    const auto t_old = fs::file_time_type::clock::now() - std::chrono::hours(48);
    const auto t_new = t_old + std::chrono::hours(24);
    fs::last_write_time(a, t_old);
    fs::last_write_time(b, t_new);
    std::string base, err;
    require(tier_pack_identity(dir.path.string(), {prof.string()}, base, err), err);
    require(base.size() == 68 && base.compare(0, 4, "sha:") == 0, "the pack id is not sha:<64 hex>");
    std::string again;
    require(tier_pack_identity(dir.path.string(), {prof.string()}, again, err) && again == base, "the id is not stable");

    // 1. the excluded profile file may change freely (its own hash identifies it)
    write_file(prof, "p2-longer");
    std::string x;
    require(tier_pack_identity(dir.path.string(), {prof.string()}, x, err) && x == base, "an excluded file changed the pack id");
    require(tier_pack_identity(dir.path.string(), {}, x, err) && x != base, "an included file did not enter the id");

    // 2. the two files swap their times: the total size and the newest time are unchanged, the per-file ones are not
    fs::last_write_time(a, t_new);
    fs::last_write_time(b, t_old);
    require(tier_pack_identity(dir.path.string(), {prof.string()}, x, err) && x != base, "swapped file times were not detected");
    fs::last_write_time(a, t_old);
    fs::last_write_time(b, t_new);
    require(tier_pack_identity(dir.path.string(), {prof.string()}, x, err) && x == base, "restoring the times did not restore the id");

    // 3. a same-size rewrite of the OLDER file with the old time kept off-by-a-second: size and newest time unchanged
    write_file(a, "AAAB");
    fs::last_write_time(a, t_old + std::chrono::seconds(1));
    require(tier_pack_identity(dir.path.string(), {prof.string()}, x, err) && x != base, "a same-size rewrite of one file was not detected");
    fs::last_write_time(a, t_old);

    // 4. a renamed file, and a missing pack
    fs::rename(b, dir.path / "experts-c.bin");
    require(tier_pack_identity(dir.path.string(), {prof.string()}, x, err) && x != base, "a renamed file was not detected");
    require(!tier_pack_identity((dir.path / "nowhere").string(), {}, x, err) && !err.empty(), "a missing pack gave an id");
}


// ---- docs/EXPERT_TIER_PLAN.md §6.4: the completion state is inside the header's checksum, and a header of the old layout
// (version 1, whose crc left fill_state out) is not read: such a tier refills once
static void test_header_crc_covers_fill_state() {
    TierHeader h{};
    std::memcpy(h.magic, kTierMagic, 8);
    h.device_count = 1;
    h.n_layers = kLayers;
    h.n_expert = kExperts;
    h.fill_state = 1;
    uint8_t buf[kTierHeaderBytes];
    encode_header(h, buf);
    TierHeader out{};
    std::string why;
    require(decode_header(buf, out, why) && out.fill_state == 1, "a fresh header does not decode: " + why);
    buf[kTierFillStateOffset] = 2;   // fill_state 1 -> 2 with the crc left as it was: a torn or forged "complete"
    require(!decode_header(buf, out, why) && why.find("header_crc") != std::string::npos, "a flipped fill_state still decoded");

    // the version 1 layout: the same fields, the crc over the first 152 bytes only
    encode_header(h, buf);
    buf[8] = 1; buf[9] = buf[10] = buf[11] = 0;
    const uint32_t old_crc = tier_crc32c(buf, 152);
    std::memcpy(buf + 156, &old_crc, 4);   // (little-endian hosts: the codec's byte order)
    require(!decode_header(buf, out, why) && why.find("layout version 1") != std::string::npos, "a version 1 header was read: " + why);

    // on a device: a version 1 tier is refilled, never reused
    Dir dir;
    const std::vector<uint64_t> blobs(kBlobs, kBlobs + kLayers);
    TierIdentity id = identity_for({'v'}, blobs);
    const std::vector<TierSpec> specs = {{make_device(dir, "v1", 8).string(), 8, 0, false}};
    FakeFileTier file_tier;
    {
        ExpertTierSet set;
        std::string err;
        bool reused = true;
        require(set.open(specs, id, err, reused) && !reused, err);
        require(set.fill(file_tier, ranked_all_hot(), id, err), err);
    }
    for (uint64_t at : {(uint64_t) 0, kTierHeaderArea}) {   // rewrite both headers as version 1 (fill_state 2, the old crc)
        std::fstream f(specs[0].path, std::ios::in | std::ios::out | std::ios::binary);
        uint8_t b[kTierHeaderBytes];
        f.seekg((std::streamoff) at);
        f.read((char*) b, sizeof b);
        b[8] = 1;
        const uint32_t c = tier_crc32c(b, 152);
        std::memcpy(b + 156, &c, 4);
        f.seekp((std::streamoff) at);
        f.write((const char*) b, sizeof b);
    }
    ExpertTierSet again;
    std::string err;
    bool reused = true;
    require(again.open(specs, id, err, reused), err);
    require(!reused, "a version 1 tier was reused");
}

// ---- §5.4: devices are compared by identity, not by path; a model file is never a tier device; foreign data is not overwritten
#ifndef _WIN32
static void test_aliases_and_model_files() {
    Dir dir;
    TierIdentity id = identity_for({'a'}, std::vector<uint64_t>(kLayers, 256u << 10));
    const fs::path a = make_device(dir, "a.tier", 8);
    std::string err;
    bool reused = true;
    // a symlink to a device given beside the device
    fs::create_symlink(a, dir.path / "a.link");
    {
        ExpertTierSet s;
        require(!s.open({{a.string(), 8, 0, false}, {(dir.path / "a.link").string(), 8, 0, false}}, id, err, reused),
                "a device and a symlink to it opened as two devices");
        require(err.find("same device") != std::string::npos, "the alias refusal does not say so: " + err);
    }
    // a hard link to a device
    fs::create_hard_link(a, dir.path / "a.hard");
    {
        ExpertTierSet s;
        require(!s.open({{(dir.path / "a.hard").string(), 8, 0, false}, {a.string(), 8, 0, false}}, id, err, reused),
                "a device and a hard link to it opened as two devices");
        require(err.find("same device") != std::string::npos, "the hard-link refusal does not say so: " + err);
    }
    // the same file through a path with a "." and a ".." in it (the old check compared the strings)
    {
        const std::string dotted = (dir.path / "." / "x" / ".." / "a.tier").string();
        fs::create_directories(dir.path / "x");
        ExpertTierSet s;
        require(!s.open({{a.string(), 8, 0, false}, {dotted, 8, 0, false}}, id, err, reused), "two spellings of one path opened");
    }
    // a model file: by its own path, through a symlink, through a hard link - all refused, and nothing is written to it
    const fs::path model = make_device(dir, "experts.bin", 8);
    { std::fstream f(model, std::ios::in | std::ios::out | std::ios::binary); f.seekp(1000); f << "MODEL"; }
    const auto before = fs::last_write_time(model);
    fs::create_symlink(model, dir.path / "model.link");
    fs::create_hard_link(model, dir.path / "model.hard");
    const std::vector<std::string> protect = {model.string(), (dir.path / "missing.profile").string()};
    for (const fs::path& p : {model, dir.path / "model.link", dir.path / "model.hard"}) {
        ExpertTierSet s;
        require(!s.open({{p.string(), 8, 0, false}}, id, err, reused, protect), "a model file was opened as a tier: " + p.string());
        require(err.find("model's file") != std::string::npos, "the model-file refusal does not say so: " + err);
    }
    {
        std::ifstream f(model, std::ios::binary);
        std::string head(1005, '\0');
        f.read(&head[0], 1005);
        require(head.compare(1000, 5, "MODEL") == 0 && fs::last_write_time(model) == before, "a refused model file was written");
    }
    // a device that is not a model file opens beside the protected list (a protect entry that does not exist is skipped)
    {
        ExpertTierSet s;
        require(s.open({{a.string(), 8, 0, false}}, id, err, reused, protect), "a separate device was refused: " + err);
    }
    // the pack's own files: tier_pack_files lists every regular file, so a tier kept inside the pack is refused
    {
        const fs::path pack = dir.path / "pack";
        fs::create_directories(pack / "sub");
        write_file(pack / "meta.txt", "m");
        const fs::path inside = make_device(dir, "pack/sub/tier.bin", 8);
        const std::vector<std::string> files = tier_pack_files(pack.string());
        require(files.size() == 2, "tier_pack_files did not list the pack's files");
        ExpertTierSet s;
        require(!s.open({{inside.string(), 8, 0, false}}, id, err, reused, files), "a tier inside the pack's folder was accepted");
    }
}
#endif

static void test_foreign_data_is_not_overwritten() {
    Dir dir;
    TierIdentity id = identity_for({'f'}, std::vector<uint64_t>(kLayers, 256u << 10));
    std::string err;
    bool reused = true;
    // someone's data in the first 128 KiB: refused, and left as it was
    const fs::path busy = make_device(dir, "busy.img", 8);
    { std::fstream f(busy, std::ios::in | std::ios::out | std::ios::binary); f.seekp(70000); f << "my photos"; }
    {
        ExpertTierSet s;
        require(!s.open({{busy.string(), 8, 0, false}}, id, err, reused), "a device holding data was opened for a fill");
        require(err.find("not a Strata expert tier") != std::string::npos && err.find("--expert-tier-overwrite") != std::string::npos,
                "the foreign-data refusal does not say what to do: " + err);
    }
    {
        std::ifstream f(busy, std::ios::binary);
        std::string b(70009, '\0');
        f.read(&b[0], 70009);
        require(b.compare(70000, 9, "my photos") == 0, "a refused device was written");
    }
    // data past the header areas but inside the mib: someone's data (a sparse file can hold it), refused and left as it was
    const fs::path late = make_device(dir, "late.img", 8);
    { std::fstream f(late, std::ios::in | std::ios::out | std::ios::binary); f.seekp(200000); f << "x"; }
    {
        ExpertTierSet s;
        require(!s.open({{late.string(), 8, 0, false}}, id, err, reused), "data past the header areas was taken as blank");
        require(err.find("not a Strata expert tier") != std::string::npos, "the refusal does not name the data: " + err);
    }
    // data past the mib of a larger device is never written by a fill of that mib: blank for it, accepted
    const fs::path beyond = make_device(dir, "beyond.img", 16);
    { std::fstream f(beyond, std::ios::in | std::ios::out | std::ios::binary); f.seekp(12ll << 20); f << "outside"; }
    {
        ExpertTierSet s;
        require(s.open({{beyond.string(), 8, 0, false}}, id, err, reused), "data past the mib was refused: " + err);
        s.close();
    }
    // the other engine's tier (STRPTDS4) is an expert tier: accepted, and refilled for this engine
    const fs::path ds4 = make_device(dir, "ds4.img", 8);
    { std::fstream f(ds4, std::ios::in | std::ios::out | std::ios::binary); f.seekp(0); f << "STRPTDS4junkjunk"; }
    {
        ExpertTierSet s;
        require(s.open({{ds4.string(), 8, 0, false}}, id, err, reused) && !reused, "the DeepSeek engine's tier was refused: " + err);
    }
    require(tier_blank_or_tier(nullptr, 0), "an empty view is blank");
}

// ---- §6.4: a flush that fails leaves no header claiming a complete tier: the next start refills
static void test_flush_failure_refills() {
    const std::vector<uint64_t> blobs(kBlobs, kBlobs + kLayers);
    TierIdentity id = identity_for({'w'}, blobs);
    // flush order of a two-device fill, per device: 1, 2 the fill_state 1 header (A, B); 3 the data and the chunk table;
    // 4, 5 the fill_state 2 header (A, B).  Fail each of 3, 4 and 5 once on device 1.
    for (int fail_at : {3, 4, 5}) {
        Dir dir;
        const std::vector<TierSpec> specs = {{make_device(dir, "a", 8).string(), 8, 0, false},
                                             {make_device(dir, "b", 8).string(), 8, 0, false}};
        FakeFileTier file_tier;
        {
            ExpertTierSet set;
            std::string err;
            bool reused = true;
            int n = 0;
            set.set_flush_fault([&](size_t d) { return d == 1 && ++n == fail_at; });
            require(set.open(specs, id, err, reused) && !reused, err);
            require(!set.fill(file_tier, ranked_all_hot(), id, err), "a fill whose flush failed reported success");
            require(!err.empty(), "the failed fill gave no reason");
            set.close();   // (the abandoned fill writes fill_state 1 over both headers here)
        }
        ExpertTierSet again;
        std::string err;
        bool reused = true;
        require(again.open(specs, id, err, reused), err);
        require(!reused, "a tier whose fill failed at flush " + std::to_string(fail_at) + " was reused");
        // and a clean fill afterwards is reused
        require(again.fill(file_tier, ranked_all_hot(), id, err), err);
        again.close();
        ExpertTierSet third;
        require(third.open(specs, id, err, reused) && reused, "a good fill after a failed one was not reused: " + err);
    }
}

// ---- §6.2: model files outside the pack (a GGUF in place) are part of the identity
static void test_pack_identity_extra_files() {
    Dir dir;
    fs::create_directories(dir.path / "pack");
    write_file(dir.path / "pack" / "meta.txt", "m");
    const fs::path g = dir.path / "model-00001-of-00001.gguf";
    write_file(g, "GGUF1111");
    std::string base, x, err;
    require(tier_pack_identity((dir.path / "pack").string(), {}, base, err, {g.string()}), err);
    require(tier_pack_identity((dir.path / "pack").string(), {}, x, err) && x != base, "the outside file did not enter the id");
    write_file(g, "GGUF2222");   // the same size, a new time
    fs::last_write_time(g, fs::file_time_type::clock::now() + std::chrono::seconds(5));
    require(tier_pack_identity((dir.path / "pack").string(), {}, x, err, {g.string()}) && x != base, "a replaced GGUF kept the id");
    require(!tier_pack_identity((dir.path / "pack").string(), {}, x, err, {(dir.path / "gone.gguf").string()}),
            "a missing model file still gave an id");
}

// ---- identity of block devices, without needing one: a partition and its disk are one device for the overlap check
static void test_block_ids() {
    TierFileId a, b, f;
    a.valid = b.valid = f.valid = true;
    a.block = b.block = true;
    a.rdev = b.rdev = 0x0801;
    require(tier_same_file(a, b), "two nodes of one block device are not the same device");
    b.rdev = 0x0802;
    require(!tier_same_file(a, b) || tier_whole_disk(0x0801) == 0x0802 || tier_whole_disk(0x0802) == 0x0801,
            "two different block devices compared equal");
    f.dev = 0x0801;
    f.ino = 12;
    require(tier_holds_file(a, f), "a block device holding a model file's file system was not caught");
    TierFileId reg = f;
    require(!tier_same_file(a, reg), "a block device and a regular file compared equal");
}

// ---- Linux block devices: a regular file by its length, a block device by BLKGETSIZE64, anything else refused
#ifndef _WIN32
static void test_device_kinds() {
    Dir dir;
    const fs::path f = make_device(dir, "reg", 4);
    int fd = ::open(f.c_str(), O_RDWR);
    require(fd >= 0, "cannot open the regular file");
    uint64_t size = 0;
    bool is_block = true;
    std::string why;
    require(tier_fd_size(fd, size, is_block, why) && size == (4u << 20) && !is_block, "a regular file's size: " + why);
    ::close(fd);
    // a character device has no capacity to map
    fd = ::open("/dev/null", O_RDWR);
    if (fd >= 0) {
        require(!tier_fd_size(fd, size, is_block, why) && !why.empty(), "/dev/null was accepted as a tier device");
        ::close(fd);
    }
    // a loop-backed block device, only where this process may create one (root, losetup, /dev/loop-control)
    if (::geteuid() != 0) { std::cout << "expert_tier_test: loop device case skipped (not root)\n"; return; }
    const fs::path img = dir.path / "loop.img";
    write_file(img, "x");
    fs::resize_file(img, 16u << 20);
    const std::string cmd = "losetup --find --show '" + img.string() + "' > '" + (dir.path / "loopdev").string() + "' 2>/dev/null";
    if (std::system(cmd.c_str()) != 0) { std::cout << "expert_tier_test: loop device case skipped (losetup unavailable)\n"; return; }
    std::string dev;
    { std::ifstream l(dir.path / "loopdev"); std::getline(l, dev); }
    struct Detach { std::string d; ~Detach() { std::string c = "losetup -d '" + d + "' 2>/dev/null"; const int rc = std::system(c.c_str()); (void) rc; } } detach{dev};
    require(!dev.empty(), "losetup printed no device");
    std::vector<TierSpec> specs = {{dev, 8, 0, false}};
    FakeFileTier file_tier;
    TierIdentity id = identity_for({'L'}, {256u << 10, 256u << 10, 1u << 20, 1u << 20});
    ExpertTierSet s1;
    std::string err;
    bool reused = true;
    require(s1.open(specs, id, err, reused), "the loop device did not open: " + err);
    require(s1.fill(file_tier, ranked_all_hot(), id, err), err);
    s1.close();
    ExpertTierSet s2;
    require(s2.open(specs, id, err, reused) && reused, "the loop device placement was not reused: " + err);
    s2.close();
    std::vector<TierSpec> too_big = {{dev, 32, 0, false}};
    ExpertTierSet s3;
    require(!s3.open(too_big, id, err, reused) && err.find("mib") != std::string::npos, "a loop device smaller than mib opened");
    // a block device under two names (the node and a symlink to it): one device
    fs::create_symlink(dev, dir.path / "loop.link");
    ExpertTierSet s4;
    require(!s4.open({{dev, 8, 0, false}, {(dir.path / "loop.link").string(), 8, 0, false}}, id, err, reused),
            "a block device and a symlink to it opened as two devices");
    // a block device another opener holds exclusively (as a mount does): O_EXCL refuses it, nothing is written
    ExpertTierSet held;
    require(held.open(specs, id, err, reused), err);
    ExpertTierSet s5;
    require(!s5.open(specs, id, err, reused) && err.find("in use") != std::string::npos, "a block device in use was opened: " + err);
}
#endif

int main() {
    std::srand(20261030);
    try {
        test_sha_is_sha256();
        test_fill_and_bytes();
        test_identity_changes_refill();
        test_torn_fill_refills();
        test_verify_full_and_refill_mark();
        test_device_too_small_and_duplicate();
        test_verify_full_keeps_every_bad_chunk();
        test_reserve_is_never_written();
        test_reuse_refuses_a_placement_outside_the_device();
        test_pack_identity_is_per_file();
        test_header_crc_covers_fill_state();
        test_foreign_data_is_not_overwritten();
        test_flush_failure_refills();
        test_pack_identity_extra_files();
        test_block_ids();
#ifndef _WIN32
        test_aliases_and_model_files();
        test_device_kinds();
#endif
        std::cout << "expert_tier_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "expert_tier_test: " << error.what() << '\n';
        return 1;
    }
}
