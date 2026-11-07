// tests/core/expert_tier_plan_test.cpp - §12.1 of docs/EXPERT_TIER_PLAN.md: every case of the uniform deal.
// Pure CPU: no device, no file, no CUDA.  The planner is the whole of the tier's placement logic, so this test
// runs anywhere (§10: development without hardware).
#include "strata/core/expert_tier_plan.hpp"

#include <cstdint>
#include <set>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::core;

static void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}

// TierDevice is one member; brace lists read as one device with too many values, so the tests build vectors by hand.
static std::vector<TierDevice> devs(std::initializer_list<uint64_t> caps) {
    std::vector<TierDevice> out;
    for (auto c : caps) out.push_back(TierDevice{c});
    return out;
}

// every expert placed, hottest first, offsets ascending and 4 KiB aligned on each device
static void test_one_device_room_for_all() {
    std::vector<uint64_t> blob(4, 4096);   // 1 blob KiB per layer, 4 layers
    std::vector<std::pair<int32_t, int32_t>> r;
    for (int32_t l = 0; l < 4; ++l)
        for (int32_t e = 0; e < 8; ++e) r.push_back({l, e});
    auto pl = plan_expert_tiers(r, blob, devs({64ull * 4096}));
    require(pl.size() == r.size(), "one device, room for all: nothing left over");
    for (size_t i = 0; i < pl.size(); ++i) {
        require(pl[i].device == 0, "one device, room for all: an expert went to no device");
        require(pl[i].offset == i * 4096, "one device, room for all: the rank order of the offsets is wrong");
    }
}

static void test_two_equal_devices_alternate() {
    std::vector<uint64_t> blob(1, 4096);
    std::vector<std::pair<int32_t, int32_t>> r;
    for (int32_t e = 0; e < 8; ++e) r.push_back({0, e});
    auto pl = plan_expert_tiers(r, blob, devs({16ull * 4096, 16ull * 4096}));
    for (size_t i = 0; i < r.size(); ++i) require(pl[i].device == (int32_t) (i % 2), "two equal devices: not the hottest-first alternation");
}

static void test_two_devices_unequal_capacity() {
    // 6 blobs of 4 KiB; device 0 holds 2, device 1 holds 6.  §4.2: alternate until the smaller is full,
    // then the rest go to the larger, and the rotation keeps running with it to the end.
    std::vector<uint64_t> blob(1, 4096);
    std::vector<std::pair<int32_t, int32_t>> r;
    for (int32_t e = 0; e < 6; ++e) r.push_back({0, e});
    auto pl = plan_expert_tiers(r, blob, devs({2ull * 4096, 6ull * 4096}));
    const int32_t want[6] = {0, 1, 0, 1, 1, 1};
    for (size_t i = 0; i < r.size(); ++i) require(pl[i].device == want[i], "unequal devices: wrong device after the small one filled");
    uint64_t used0 = 0, used1 = 0;
    for (const auto& p : pl) (p.device == 0 ? used0 : used1) += 4096;
    require(used0 == 2ull * 4096 && used1 == 4ull * 4096, "unequal devices: a device over or under its capacity");
}

static void test_layers_with_different_blob_sizes() {
    // layer 0 blobs 1 MiB, layer 1 blobs 4 KiB.  Device 0 has 1 MiB + 8 KiB, device 1 has 2 MiB.
    // Ranks: (1,0) 4k, (0,0) 1M, (1,1) 4k, (0,1) 1M, (1,2) 4k, (1,3) 4k.
    std::vector<uint64_t> blob = {1u << 20, 4096};
    std::vector<std::pair<int32_t, int32_t>> r = {{1, 0}, {0, 0}, {1, 1}, {0, 1}, {1, 2}, {1, 3}};
    auto pl = plan_expert_tiers(r, blob, devs({(1u << 20) + 8192, 2u << 20}));
    // the deal: r0->d0, r1->d1, r2->d0, r3->d1 (d1 used 1M+4k fits 2M), r4: d0 used 8k+4k... charged by real size,
    // so d0 took 4k+4k=8k plus the 1 MiB at rank 2?  No: d0 was charged r0 (4k) + r2 (4k) = 8k; r3 was d1's.
    // r4 (4k): d0 has 1 MiB of its capacity left unused - it fits - d0 took r4?  The rotation continues by rank.
    for (const auto& p : pl) require(p.device >= 0, "mixed blob sizes: an expert fell out of both devices");
    uint64_t used[2] = {0, 0};
    for (size_t i = 0; i < pl.size(); ++i)
        used[(size_t) pl[i].device] += blob[(size_t) r[i].first];
    require(used[0] <= (1u << 20) + 8192 && used[1] <= 2u << 20, "mixed blob sizes: a device overfilled");
}

static void test_device_smaller_than_one_blob_dropped() {
    std::vector<uint64_t> blob(1, 4096);
    std::vector<std::pair<int32_t, int32_t>> r = {{0, 0}, {0, 1}};
    auto pl = plan_expert_tiers(r, blob, devs({2048, 16ull * 4096}));   // device 0 cannot take any blob
    require(pl[0].device == 1 && pl[1].device == 1, "a too-small device was not dropped from the rotation");
}

static void test_all_devices_full_stops() {
    std::vector<uint64_t> blob(1, 4096);
    std::vector<std::pair<int32_t, int32_t>> r;
    for (int32_t e = 0; e < 10; ++e) r.push_back({0, e});
    auto pl = plan_expert_tiers(r, blob, devs({4ull * 4096, 4ull * 4096}));   // room for 8 blobs
    int placed = 0;
    for (const auto& p : pl) if (p.device >= 0) ++placed;
    require(placed == 8, "all devices full: not exactly 8 experts placed");
    for (size_t i = 8; i < pl.size(); ++i)
        require(pl[i].device == -1, "all devices full: a colder expert took a place (stop rule broken)");
}

static void test_deterministic() {
    std::vector<uint64_t> blob = {4096, 8192, 4096};
    std::vector<std::pair<int32_t, int32_t>> r;
    for (int32_t l = 0; l < 3; ++l)
        for (int32_t e = 0; e < 11; ++e) r.push_back({l, e});
    std::vector<TierDevice> dv = {{100000}, {250000}, {65536}};
    const auto a = plan_expert_tiers(r, blob, dv);
    const auto b = plan_expert_tiers(r, blob, dv);
    bool same = a.size() == b.size();
    for (size_t i = 0; same && i < a.size(); ++i) same = a[i].device == b[i].device && a[i].offset == b[i].offset;
    require(same, "same input, run twice: not identical output");
    // the 4 KiB alignment holds for every placed offset
    for (const auto& p : a) if (p.device >= 0) require(p.offset % 4096 == 0, "an offset is not 4 KiB aligned");
}

static void test_band_order_contract() {
    // §12.1 band order: the planner is handed only the experts RAM did not keep; it places what it is given and
    // never invents an expert.  (The contract: the caller's list, hottest first, is the whole input.)
    std::vector<uint64_t> blob(2, 4096);
    std::vector<std::pair<int32_t, int32_t>> r = {{1, 7}, {0, 3}};
    auto pl = plan_expert_tiers(r, blob, devs({4096, 4096}));
    require(pl[0].device == 0 && pl[1].device == 1, "band order: the given order is not honoured");
}

static void test_device_order_reversed_differs() {
    std::vector<uint64_t> blob(1, 4096);
    std::vector<std::pair<int32_t, int32_t>> r = {{0, 0}, {0, 1}, {0, 2}};
    const auto a = plan_expert_tiers(r, blob, devs({8192, 4096}));
    const auto b = plan_expert_tiers(r, blob, devs({4096, 8192}));
    bool differs = false;
    for (size_t i = 0; i < a.size(); ++i) differs |= a[i].device != b[i].device;
    require(differs, "reversing the device order must give a different placement (§6.3 forces the refill on it)");
}

static void test_empty() {
    auto pl = plan_expert_tiers({}, {4096}, devs({4096}));
    require(pl.empty(), "empty ranked list: placements exist");
    auto pl2 = plan_expert_tiers({{0, 0}}, {4096}, {});
    require(pl2.size() == 1 && pl2[0].device == -1, "no devices: the expert is not left to the file tier");
}

// §4.2: the 4 KiB alignment is part of the fit.  A blob that starts at the next boundary must END inside the capacity.
static void test_alignment_is_counted_before_the_fit() {
    // blobs of 5000 bytes: the second starts at 8192, so two need 8192 + 5000 = 13192 bytes, not 10000
    std::vector<uint64_t> blob(1, 5000);
    std::vector<std::pair<int32_t, int32_t>> r = {{0, 0}, {0, 1}, {0, 2}};
    auto pl = plan_expert_tiers(r, blob, devs({13191}));   // one byte short of two aligned blobs
    require(pl[0].device == 0 && pl[0].offset == 0, "alignment: the first blob has room");
    require(pl[1].device == -1, "alignment: a blob that ends past the capacity after aligning was placed");
    require(pl[2].device == -1, "alignment: the stop rule did not hold after the refusal");
    pl = plan_expert_tiers(r, blob, devs({13192}));         // exactly two aligned blobs
    require(pl[0].device == 0 && pl[1].device == 0 && pl[1].offset == 8192, "exact capacity: two blobs fit to the byte");
    require(pl[2].device == -1, "exact capacity: a third blob was placed past the end");
    // a capacity of exactly one blob, and one byte under it
    pl = plan_expert_tiers(r, blob, devs({5000}));
    require(pl[0].device == 0 && pl[1].device == -1, "capacity == blob: exactly one expert");
    pl = plan_expert_tiers(r, blob, devs({4999}));
    require(pl[0].device == -1, "capacity < blob: a blob was placed");
}

static void test_reserve_and_variable_blob_boundaries() {
    // capacity as the caller gives it, already less the reserve: 3 MiB - 1 byte must not take a third 1 MiB blob after a
    // 1 MiB + 1 byte one pushed the cursor off the 4 KiB grid
    const uint64_t MiB = 1ull << 20;
    std::vector<uint64_t> blob = {MiB + 1, MiB, 4096 + 1};
    std::vector<std::pair<int32_t, int32_t>> r = {{0, 0}, {1, 0}, {1, 1}, {2, 0}};
    // layer 0 ends at MiB+1; layer 1 starts at MiB+4096 and ends at 2*MiB+4096; the next layer 1 starts at 2*MiB+4096
    auto pl = plan_expert_tiers(r, blob, devs({3 * MiB + 4095}));
    require(pl[0].offset == 0 && pl[1].offset == MiB + 4096, "variable blobs: wrong offsets");
    // the third blob starts at 2 MiB + 4096 and is 1 MiB: it ends at 3 MiB + 4096, one byte past this capacity
    require(pl[2].device == -1, "reserve boundary: a blob ending one byte past the capacity was placed");
    require(pl[3].device == -1, "reserve boundary: the stop rule did not hold (a colder expert took a hotter one's place)");
    // one byte more capacity and the same blob fits exactly
    pl = plan_expert_tiers(r, blob, devs({3 * MiB + 4096}));
    require(pl[2].device == 0 && pl[2].offset == 2 * MiB + 4096, "exact capacity with unequal blobs: the third blob fits");
    require(pl[3].device == -1, "exact capacity: nothing is left for the fourth");
}

// every placement fits its device: offset + blob <= capacity, offsets are 4 KiB aligned and never overlap
static void test_fuzz_never_overfills() {
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    auto rnd = [&](uint64_t n) { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return (seed >> 33) % n; };
    for (int iter = 0; iter < 2000; ++iter) {
        const size_t layers = 1 + rnd(4);
        std::vector<uint64_t> blob(layers);
        for (auto& b : blob) b = 1 + rnd(20000);
        std::vector<TierDevice> d(1 + rnd(3));
        for (auto& x : d) x.capacity_bytes = rnd(120000);
        std::vector<std::pair<int32_t, int32_t>> r;
        for (int i = 0; i < 40; ++i) r.push_back({(int32_t) rnd(layers), i});
        const auto pl = plan_expert_tiers(r, blob, d);
        std::vector<std::vector<std::pair<uint64_t, uint64_t>>> spans(d.size());
        for (size_t i = 0; i < pl.size(); ++i) {
            if (pl[i].device < 0) continue;
            const uint64_t b = blob[(size_t) r[i].first];
            require(pl[i].offset % 4096 == 0, "fuzz: unaligned offset");
            require(pl[i].offset + b <= d[(size_t) pl[i].device].capacity_bytes, "fuzz: a blob ends past its device's capacity");
            for (const auto& sp : spans[(size_t) pl[i].device])
                require(pl[i].offset >= sp.second || pl[i].offset + b <= sp.first, "fuzz: blobs overlap");
            spans[(size_t) pl[i].device].push_back({pl[i].offset, pl[i].offset + b});
        }
    }
}

// ---- tier_ranked_rest: the planner's input list, with and without a profile
static bool has_pair(const std::vector<std::pair<int32_t, int32_t>>& v, int32_t l, int32_t e) {
    for (const auto& p : v) if (p.first == l && p.second == e) return true;
    return false;
}
static void test_ranked_rest_without_a_profile() {
    const auto none = [](int64_t, int64_t) { return false; };
    auto r = tier_ranked_rest({}, 3, 4, none);
    require(r.size() == 12, "no profile: every expert is in the list");
    for (size_t i = 0; i < r.size(); ++i) require(r[i].first == (int32_t) (i / 4) && r[i].second == (int32_t) (i % 4), "no profile: layer-major order");
    // GPU-held (layer 1, expert 2) and RAM-resident (layer 0, expert 0) are left out
    const auto held = [](int64_t l, int64_t e) { return (l == 1 && e == 2) || (l == 0 && e == 0); };
    r = tier_ranked_rest({}, 3, 4, held);
    require(r.size() == 10 && !has_pair(r, 1, 2) && !has_pair(r, 0, 0), "no profile: a held expert entered the tier's list");
    require(tier_ranked_rest({}, 0, 4, none).empty() && tier_ranked_rest({}, 3, 0, none).empty(), "empty shape");
}
static void test_ranked_rest_with_a_profile() {
    const auto held = [](int64_t l, int64_t e) { return (l == 1 && e == 2) || (l == 0 && e == 0); };
    // hottest first; a duplicate; an out-of-range pair; a GPU-held one; a RAM-resident one
    const std::vector<std::pair<int32_t, int32_t>> prof = {{2, 3}, {1, 2}, {0, 1}, {2, 3}, {9, 9}, {0, 0}, {-1, 0}, {1, 0}, {0, 1}};
    const auto r = tier_ranked_rest(prof, 3, 4, held);
    require(r.size() == 10, "profile: 12 experts less the 2 held ones");
    require(r[0] == std::make_pair(2, 3) && r[1] == std::make_pair(0, 1) && r[2] == std::make_pair(1, 0), "profile: the ranked ones come first, in rank order");
    std::set<std::pair<int32_t, int32_t>> uniq(r.begin(), r.end());
    require(uniq.size() == r.size(), "profile: an expert is listed twice");
    require(!has_pair(r, 1, 2) && !has_pair(r, 0, 0), "profile: a held expert entered the list");
    // the unranked remainder follows in layer-major order and contains every other expert
    size_t pos = 3;
    for (int32_t l = 0; l < 3; ++l)
        for (int32_t e = 0; e < 4; ++e) {
            if (held(l, e) || (l == 2 && e == 3) || (l == 0 && e == 1) || (l == 1 && e == 0)) continue;
            require(pos < r.size() && r[pos] == std::make_pair(l, e), "profile: the unranked remainder is out of order");
            ++pos;
        }
    require(pos == r.size(), "profile: the remainder missed experts");
    // a profile that names only held experts still leaves the unranked ones to the tier
    const auto r2 = tier_ranked_rest({{1, 2}, {0, 0}}, 3, 4, held);
    require(r2.size() == 10, "profile of held experts only: the tier lost the rest");
}

int main() {
    try {
        test_empty();
        test_one_device_room_for_all();
        test_two_equal_devices_alternate();
        test_two_devices_unequal_capacity();
        test_layers_with_different_blob_sizes();
        test_device_smaller_than_one_blob_dropped();
        test_all_devices_full_stops();
        test_deterministic();
        test_band_order_contract();
        test_device_order_reversed_differs();
        test_alignment_is_counted_before_the_fit();
        test_reserve_and_variable_blob_boundaries();
        test_fuzz_never_overfills();
        test_ranked_rest_without_a_profile();
        test_ranked_rest_with_a_profile();
        std::cout << "expert_tier_plan_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "expert_tier_plan_test: " << error.what() << '\n';
        return 1;
    }
}
