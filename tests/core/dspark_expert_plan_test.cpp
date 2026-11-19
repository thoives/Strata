// tests/core/dspark_expert_plan_test.cpp - the DSpark hybrid placement's host policy (docs/DSPARK_HYBRID_PLAN.md §3): the
// per-layer slot quotas and the adaptive draft expert cache.  Pure CPU: no device, no file, no CUDA.  Synthetic routes only;
// nothing here is a speed or memory measurement.
#include "strata/core/dspark_expert_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::core;

static void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}

static constexpr uint64_t KIB4 = 4096;

// A config with decay 1 (scores are plain counts) unless a test says otherwise: the margin and tie cases read as arithmetic.
static DraftCacheConfig counts(double min_score = 2.0, double margin = 1.25) {
    DraftCacheConfig c;
    c.decay_per_round = 1.0;
    c.min_score = min_score;
    c.margin = margin;
    c.retry_cooldown_rounds = 16;
    return c;
}

// --- quotas ----------------------------------------------------------------------------------------------------------

static void test_zero_budget_gives_nothing() {
    auto p = plan_draft_quotas(0, {{KIB4, 8}, {KIB4, 8}});
    for (const auto& q : p.layers) require(q.usable == 0 && q.spare == 0 && !q.fully_resident, "zero budget: a layer got slots");
    require(p.leftover_bytes == 0, "zero budget: leftover must be 0");
}

static void test_empty_layers_take_nothing_and_the_rest_goes_on() {
    auto p = plan_draft_quotas(6 * KIB4, {{KIB4, 0}, {0, 8}, {KIB4, 100}});
    require(p.layers[0].usable == 0 && p.layers[1].usable == 0, "empty layers: a layer with no experts or no bytes got slots");
    require(p.layers[2].usable == 5 && p.layers[2].spare == 1, "empty layers: the budget did not reach the layer that has experts");
    require(p.leftover_bytes == 0, "empty layers: leftover");
}

static void test_partial_layer_gets_usable_plus_one_spare() {
    auto p = plan_draft_quotas(5 * KIB4, {{KIB4, 100}});
    const auto& q = p.layers[0];
    require(q.usable == 4 && q.spare == 1 && !q.fully_resident, "partial: expected 4 usable and 1 spare");
    require(q.bytes() == 5 * KIB4 && p.leftover_bytes == 0, "partial: bytes or leftover");
}

static void test_a_share_too_small_for_a_spare_gets_nothing() {
    auto p = plan_draft_quotas(KIB4, {{KIB4, 100}});   // one slot: a usable slot needs its spare beside it
    require(p.layers[0].usable == 0 && p.layers[0].spare == 0, "one slot: must give no adaptive slots");
    require(p.leftover_bytes == KIB4, "one slot: the whole budget must be left for the verifier");
    auto two = plan_draft_quotas(2 * KIB4, {{KIB4, 100}});
    require(two.layers[0].usable == 1 && two.layers[0].spare == 1, "two slots: one usable and its spare");
}

static void test_fully_resident_layer_returns_its_share() {
    // layer 0 has 2 experts (8 KiB), layer 1 has 100 (400 KiB); 10 slots of budget: equal shares of 20 KiB each.  Layer 0 fits,
    // so it takes its 8 KiB and layer 1 shares the other 8 slots: 7 usable and a spare.
    auto p = plan_draft_quotas(10 * KIB4, {{KIB4, 2}, {KIB4, 100}});
    require(p.layers[0].fully_resident && p.layers[0].usable == 2 && p.layers[0].spare == 0, "resident: a fully resident layer's size");
    require(p.layers[1].usable == 7 && p.layers[1].spare == 1 && !p.layers[1].fully_resident, "resident: the rest of the budget");
    require(p.leftover_bytes == 0, "resident: leftover");
}

static void test_slot_bytes_are_page_aligned() {
    auto p = plan_draft_quotas(64 * KIB4, {{5000, 16}});
    require(p.layers[0].slot_bytes == 8192, "alignment: 5000 bytes is an 8192-byte slot");
    require(p.layers[0].bytes() % KIB4 == 0, "alignment: bytes not a page multiple");
}

static void test_quotas_never_exceed_the_budget_and_are_deterministic() {
    uint64_t seed = 987654321ull;
    auto rnd = [&](uint64_t n) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return (seed >> 33) % n;
    };
    for (int run = 0; run < 500; ++run) {
        std::vector<DraftLayerShape> shapes;
        const uint64_t nl = 1 + rnd(5);
        for (uint64_t l = 0; l < nl; ++l) shapes.push_back({1 + rnd(20000), (int32_t) rnd(40)});
        const uint64_t budget = rnd(300) * 1000 + rnd(1000);
        auto a = plan_draft_quotas(budget, shapes);
        auto b = plan_draft_quotas(budget, shapes);
        uint64_t used = 0;
        for (size_t l = 0; l < shapes.size(); ++l) {
            const auto& q = a.layers[l];
            used += q.bytes();
            require(q.usable == b.layers[l].usable && q.spare == b.layers[l].spare, "fuzz: not deterministic");
            require(q.usable >= 0 && q.spare >= 0, "fuzz: a negative slot count");
            if (q.usable == 0 && q.spare == 0) continue;
            require(q.slot_bytes % KIB4 == 0, "fuzz: a slot is not page aligned");
            if (q.fully_resident) require(q.usable == q.n_experts && q.spare == 0, "fuzz: fully resident layer shape");
            else require(q.usable + q.spare < q.n_experts, "fuzz: a partial layer holds every expert");
        }
        require(used <= budget, "fuzz: the quotas exceed the budget");
        require(a.leftover_bytes == budget - used, "fuzz: leftover does not account for the budget");
    }
}

// --- the cache policy ------------------------------------------------------------------------------------------------

static void test_free_usable_slots_fill_before_any_eviction() {
    // 3 usable + 1 spare, four equally hot experts: three free-slot copies, and no fourth since nothing is resident to evict
    auto plan = plan_draft_quotas(4 * KIB4, {{KIB4, 16}});
    DraftExpertCache cache(plan, counts());
    for (int e = 0; e < 4; ++e)
        for (int k = 0; k < 3; ++k) cache.record(0, e, 0, false);
    auto got = cache.propose(0, 10);
    require(got.size() == 3, "free slots: expected exactly the three usable slots to fill");
    for (const auto& s : got) require(s.victim == -1, "free slots: a free slot must not evict");
    require(got[0].expert == 0 && got[1].expert == 1 && got[2].expert == 2, "free slots: stable tie order");
}

static void test_stable_ties_pick_the_lower_expert() {
    auto plan = plan_draft_quotas(2 * KIB4, {{KIB4, 8}});
    DraftExpertCache cache(plan, counts());
    for (int k = 0; k < 2; ++k) {   // two routes each: both clear the minimum, and their scores are exactly equal
        cache.record(0, 5, 0, true);
        cache.record(0, 3, 0, true);
    }
    auto got = cache.propose(0, 1);
    require(got.size() == 1 && got[0].expert == 3, "stable ties: the lower expert must win an exact tie");
}

static void test_a_tied_score_goes_to_the_more_recent_use() {
    // A: two routes at round 0 (weight 2); B: one route at round 1 (weight 1).  With decay 1/2 both score 1 at round 1.
    auto plan = plan_draft_quotas(2 * KIB4, {{KIB4, 8}});
    DraftCacheConfig c = counts(0.5);
    c.decay_per_round = 0.5;
    DraftExpertCache cache(plan, c);
    cache.record(0, 1, 0, false);
    cache.record(0, 1, 0, false);
    cache.record(0, 4, 1, false);
    require(cache.score(0, 1, 1) == cache.score(0, 4, 1), "tie setup: the scores must be exactly equal");
    auto got = cache.propose(1, 1);
    require(got.size() == 1 && got[0].expert == 4, "recency tie: the more recent expert must win an exact score tie");
}

static void test_decay_halves_per_draft_round_and_is_flat_within_a_round() {
    auto plan = plan_draft_quotas(2 * KIB4, {{KIB4, 8}});
    DraftCacheConfig c = counts();
    c.decay_per_round = 0.5;
    DraftExpertCache cache(plan, c);
    cache.record(0, 2, 0, true);
    require(cache.score(0, 2, 0) == 1.0, "decay: the score just after a route");
    require(cache.score(0, 2, 1) == 0.5, "decay: one round later");
    require(cache.score(0, 2, 3) == 0.125, "decay: three rounds later");
    require(cache.score(0, 2, 3) == cache.score(0, 2, 3), "decay: a read is pure");
}

static void test_min_score_keeps_cold_experts_out() {
    auto plan = plan_draft_quotas(2 * KIB4, {{KIB4, 8}});
    DraftExpertCache cache(plan, counts(2.0));
    cache.record(0, 6, 0, true);
    require(cache.propose(0, 4).empty(), "min score: one route is below the threshold of 2");
    cache.record(0, 6, 0, true);
    auto got = cache.propose(0, 4);
    require(got.size() == 1 && got[0].expert == 6, "min score: two routes must be admitted");
}

static void test_margin_blocks_churn_and_admits_a_clear_winner() {
    auto plan = plan_draft_quotas(2 * KIB4, {{KIB4, 8}});   // 1 usable + 1 spare
    DraftExpertCache cache(plan, counts(2.0, 1.25));
    for (int k = 0; k < 10; ++k) cache.record(0, 0, 0, true);
    auto first = cache.propose(0, 1);
    require(first.size() == 1 && first[0].expert == 0, "margin: the first resident");
    cache.complete(first[0], true, 0);

    for (int k = 0; k < 12; ++k) cache.record(0, 1, 1, false);   // 12 < 1.25 * 10: not enough to evict
    require(cache.propose(1, 4).empty(), "margin: 12 routes against a victim of 10 must not swap");

    for (int k = 0; k < 2; ++k) cache.record(0, 1, 1, false);    // 14 >= 12.5: a clear winner
    auto got = cache.propose(1, 4);
    require(got.size() == 1 && got[0].expert == 1 && got[0].victim == 0, "margin: 14 routes against 10 must swap the victim out");
}

static void test_the_victim_is_coldest_then_least_recent() {
    // usable 2 + spare 1.  Experts 2 and 5 are both resident at score 3; 2 was used last at round 0, 5 at round 4.
    auto plan = plan_draft_quotas(3 * KIB4, {{KIB4, 8}});
    DraftExpertCache cache(plan, counts(2.0, 1.0));
    for (int k = 0; k < 3; ++k) cache.record(0, 2, 0, true);
    auto a = cache.propose(0, 1);
    require(a.size() == 1 && a[0].expert == 2, "victim: setup 1");
    cache.complete(a[0], true, 0);
    for (int k = 0; k < 3; ++k) cache.record(0, 5, 4, true);
    auto b = cache.propose(4, 1);
    require(b.size() == 1 && b[0].expert == 5, "victim: setup 2");
    cache.complete(b[0], true, 4);

    for (int k = 0; k < 10; ++k) cache.record(0, 7, 4, true);
    auto got = cache.propose(4, 1);
    require(got.size() == 1 && got[0].expert == 7 && got[0].victim == 2,
            "victim: an equal score evicts the least recently used resident");
}

static void test_max_swaps_caps_the_proposals() {
    auto plan = plan_draft_quotas(8 * KIB4, {{KIB4, 32}});
    DraftExpertCache cache(plan, counts());
    for (int e = 0; e < 8; ++e)
        for (int k = 0; k < 3; ++k) cache.record(0, e, 0, false);
    require(cache.propose(0, 2).size() == 2, "max swaps: at most two copies");
}

static void test_hotter_layers_are_visited_in_index_order_under_one_cap() {
    auto plan = plan_draft_quotas(4 * KIB4, {{KIB4, 8}, {KIB4, 8}});   // two layers share 4 slots: each gets 1 usable + 1 spare
    DraftExpertCache cache(plan, counts());
    for (int k = 0; k < 3; ++k) {
        cache.record(0, 0, 0, false);
        cache.record(1, 0, 0, false);
    }
    auto got = cache.propose(0, 1);
    require(got.size() == 1 && got[0].layer == 0, "layer order: the first layer takes the shared cap");
}

static void test_in_flight_copies_are_bounded_and_not_repeated() {
    auto plan = plan_draft_quotas(2 * KIB4, {{KIB4, 8}});   // 1 usable + 1 spare
    DraftExpertCache cache(plan, counts(2.0, 1.25));
    for (int k = 0; k < 3; ++k) cache.record(0, 1, 0, true);
    auto a = cache.propose(0, 1);
    require(a.size() == 1 && a[0].expert == 1, "in flight: setup");
    cache.complete(a[0], true, 0);

    for (int k = 0; k < 20; ++k) cache.record(0, 2, 1, false);
    for (int k = 0; k < 20; ++k) cache.record(0, 3, 1, false);
    auto b = cache.propose(1, 10);
    require(b.size() == 1 && b[0].expert == 2 && b[0].victim == 1, "in flight: one eviction copy at a time (lower expert first)");
    require(cache.propose(1, 10).empty(), "in flight: the spare is taken, so no second copy");
    require(cache.resident(0, 1) && !cache.resident(0, 2), "in flight: the victim keeps its slot until the copy completes");

    cache.complete(b[0], true, 1);
    require(!cache.resident(0, 1) && cache.resident(0, 2), "in flight: the swap publishes after completion");
    require(cache.adaptive_resident(0) == 1, "in flight: still one resident");
    require(cache.propose(2, 10).empty(), "in flight: 20 routes against a resident of 20 do not clear the margin of 1.25");
}

static void test_a_failed_copy_keeps_the_old_mapping_and_cools_down() {
    auto plan = plan_draft_quotas(2 * KIB4, {{KIB4, 8}});
    DraftExpertCache cache(plan, counts(2.0, 1.25));
    for (int k = 0; k < 3; ++k) cache.record(0, 1, 0, true);
    auto a = cache.propose(0, 1);
    cache.complete(a[0], true, 0);
    for (int k = 0; k < 20; ++k) cache.record(0, 2, 1, false);

    auto b = cache.propose(1, 10);
    require(b.size() == 1 && b[0].expert == 2, "failed copy: setup");
    cache.complete(b[0], false, 1);
    require(cache.resident(0, 1) && !cache.resident(0, 2), "failed copy: the old mapping must stay");
    require(cache.stats(0).failed_copies == 1 && cache.stats(0).promotions == 1, "failed copy: counters");
    require(cache.propose(1, 10).empty(), "failed copy: the expert is in cooldown");
    require(cache.propose(16, 10).empty(), "failed copy: still in cooldown before round 17");
    auto again = cache.propose(17, 10);
    require(again.size() == 1 && again[0].expert == 2, "failed copy: retried once the cooldown has passed");
}

static void test_fully_resident_layers_hold_every_expert_and_never_swap() {
    auto plan = plan_draft_quotas(2 * KIB4, {{KIB4, 2}});   // 2 experts, 2 slots: all of them fit
    DraftExpertCache cache(plan, counts());
    require(plan.layers[0].fully_resident, "resident: setup");
    require(cache.resident(0, 0) && cache.resident(0, 1), "resident: every expert is on the GPU");
    cache.record(0, 0, 0, true);
    cache.record(0, 1, 0, true);
    require(cache.propose(0, 10).empty(), "resident: a fully resident layer never proposes a swap");
    require(cache.adaptive_resident(0) == 0, "resident: no adaptive set for it");
    require(cache.stats(0).gpu_hits == 2, "resident: routes still count");
}

static void test_a_resident_is_kept_when_nothing_challenges_it() {
    // A turn can pause, a tool call can run, and the rounds do not advance: a resident with no challenger is never evicted.
    auto plan = plan_draft_quotas(2 * KIB4, {{KIB4, 8}});
    DraftExpertCache cache(plan, counts(2.0, 1.25));
    for (int k = 0; k < 4; ++k) cache.record(0, 6, 0, true);
    auto a = cache.propose(0, 1);
    cache.complete(a[0], true, 0);
    require(cache.propose(0, 10).empty(), "retention: no challenger, no swap");
    require(cache.resident(0, 6), "retention: the resident stays");
}

// The engine starts with the placement's VRAM experts: they are adopted, fill the usable slots, and are the victims of the first
// swap (coldest, then lowest index: no routes yet), while the spare stays free for the copy.
static void test_adopted_placement_residents_fill_the_slots_and_are_the_victims() {
    auto plan = plan_draft_quotas(3 * KIB4, {{KIB4, 8}});   // usable 2 + spare 1
    DraftExpertCache cache(plan, counts(2.0, 1.25));
    cache.adopt_resident(0, 4);
    cache.adopt_resident(0, 1);
    require(cache.adaptive_resident(0) == 2 && cache.resident(0, 1) && cache.resident(0, 4), "adopt: the two usable slots");
    bool threw = false;
    try {
        cache.adopt_resident(0, 6);
    } catch (const std::logic_error&) {
        threw = true;
    }
    require(threw, "adopt: a third usable resident must be refused");
    for (int k = 0; k < 3; ++k) cache.record(0, 5, 0, false);
    auto got = cache.propose(0, 4);
    require(got.size() == 1 && got[0].expert == 5 && got[0].victim == 1,
            "adopt: the hottest candidate takes the spare and evicts the lowest-index adopted resident (no routes yet)");
    require(cache.resident(0, 1), "adopt: the victim keeps its slot until the copy completes");
}

static void test_misuse_is_an_error_not_a_silent_result() {
    auto plan = plan_draft_quotas(2 * KIB4, {{KIB4, 8}});
    DraftExpertCache cache(plan, counts());
    bool threw = false;
    try {
        cache.complete(DraftSwap{0, 4, -1}, true, 0);
    } catch (const std::logic_error&) {
        threw = true;
    }
    require(threw, "misuse: completing a swap that was never proposed must throw");

    cache.record(0, 1, 5, true);
    threw = false;
    try {
        cache.record(0, 1, 4, true);
    } catch (const std::logic_error&) {
        threw = true;
    }
    require(threw, "misuse: a route for one expert going back in rounds must throw");

    threw = false;
    try {
        cache.record(0, 99, 6, true);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    require(threw, "misuse: an expert outside the layer must throw");

    threw = false;
    try {
        DraftCacheConfig bad = counts();
        bad.decay_per_round = 0.0;
        DraftExpertCache refused(plan, bad);
        (void) refused;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "misuse: a zero decay must be refused");

    threw = false;
    try {
        DraftCacheConfig bad = counts(2.0, 0.5);
        DraftExpertCache refused(plan, bad);
        (void) refused;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "misuse: a margin below 1 would swap to a colder expert and must be refused");
}

static void test_reset_forgets_everything() {
    auto plan = plan_draft_quotas(2 * KIB4, {{KIB4, 8}});
    DraftExpertCache cache(plan, counts());
    for (int k = 0; k < 3; ++k) cache.record(0, 1, 0, true);
    cache.complete(cache.propose(0, 1)[0], true, 0);
    cache.reset();
    require(!cache.resident(0, 1) && cache.adaptive_resident(0) == 0, "reset: residents forgotten");
    require(cache.score(0, 1, 0) == 0.0, "reset: scores forgotten");
    require(cache.stats(0).gpu_hits == 0 && cache.stats(0).promotions == 0, "reset: counters forgotten");
    require(cache.propose(0, 10).empty(), "reset: nothing to propose from an empty histogram");
}

static void test_counters_keep_gpu_and_cpu_routes_apart() {
    auto plan = plan_draft_quotas(2 * KIB4, {{KIB4, 8}});
    DraftExpertCache cache(plan, counts());
    cache.record(0, 0, 0, true);
    cache.record(0, 1, 0, false);
    cache.record(0, 1, 1, false);
    require(cache.stats(0).gpu_hits == 1 && cache.stats(0).cpu_misses == 2, "counters: hits and misses");
    cache.complete(cache.propose(1, 1)[0], true, 1);
    require(cache.stats(0).promotions == 1 && cache.stats(0).promoted_bytes == KIB4, "counters: promotion bytes");
}

// The invariants hold after every step of a seeded run of records, proposals and completions (some failing): residents plus
// copies in flight never exceed the slots, a victim is resident until its swap completes, no expert is in flight twice.
static void test_fuzz_invariants_hold_through_random_histories() {
    uint64_t seed = 42ull;
    auto rnd = [&](uint64_t n) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return (seed >> 33) % n;
    };
    for (int run = 0; run < 60; ++run) {
        std::vector<DraftLayerShape> shapes;
        const uint64_t nl = 1 + rnd(3);
        for (uint64_t l = 0; l < nl; ++l) shapes.push_back({KIB4 * (1 + rnd(3)), (int32_t) (2 + rnd(20))});
        const auto plan = plan_draft_quotas(KIB4 * rnd(60), shapes);
        DraftCacheConfig cfg;
        cfg.decay_per_round = 0.9;
        cfg.min_score = 1.0;
        cfg.margin = 1.25;
        cfg.retry_cooldown_rounds = 4;
        DraftExpertCache cache(plan, cfg);
        std::vector<DraftSwap> open;
        uint64_t round = 0;
        for (int step = 0; step < 400; ++step) {
            const uint64_t op = rnd(10);
            if (op < 6) {
                const int32_t l = (int32_t) rnd(nl);
                cache.record(l, (int32_t) rnd((uint64_t) shapes[(size_t) l].n_experts), round, rnd(2) == 0);
            } else if (op < 8) {
                round += 1 + rnd(3);
                auto got = cache.propose(round, (int32_t) (1 + rnd(3)));
                open.insert(open.end(), got.begin(), got.end());
            } else if (!open.empty()) {
                const size_t k = rnd(open.size());
                cache.complete(open[k], rnd(3) != 0, round);
                open.erase(open.begin() + (std::ptrdiff_t) k);
            }

            for (int32_t l = 0; l < (int32_t) nl; ++l) {
                const DraftLayerQuota& q = plan.layers[(size_t) l];
                int32_t res = 0, out = 0;
                std::vector<int32_t> flying, victims;
                for (const auto& s : open) {
                    if (s.layer != l) continue;
                    ++out;
                    require(!q.fully_resident, "fuzz: a fully resident layer proposed a swap");
                    require(s.victim < 0 || cache.resident(l, s.victim), "fuzz: a victim left its slot before its copy completed");
                    require(!cache.resident(l, s.expert), "fuzz: a copy target is already resident");
                    for (int32_t f : flying) require(f != s.expert, "fuzz: an expert is in flight twice");
                    flying.push_back(s.expert);
                    if (s.victim >= 0) {
                        for (int32_t v : victims) require(v != s.victim, "fuzz: a victim was chosen by two copies");
                        victims.push_back(s.victim);
                    }
                }
                if (!q.fully_resident) {
                    for (int32_t e = 0; e < q.n_experts; ++e) res += cache.resident(l, e) ? 1 : 0;
                    require(res == cache.adaptive_resident(l), "fuzz: residents disagree with the count");
                    require(res <= q.usable, "fuzz: more residents than usable slots");
                    require(res + out <= q.usable + q.spare, "fuzz: residents and copies exceed the slots");
                }
            }
        }
    }
}

int main() {
    try {
        test_zero_budget_gives_nothing();
        test_empty_layers_take_nothing_and_the_rest_goes_on();
        test_partial_layer_gets_usable_plus_one_spare();
        test_a_share_too_small_for_a_spare_gets_nothing();
        test_fully_resident_layer_returns_its_share();
        test_slot_bytes_are_page_aligned();
        test_quotas_never_exceed_the_budget_and_are_deterministic();
        test_free_usable_slots_fill_before_any_eviction();
        test_stable_ties_pick_the_lower_expert();
        test_a_tied_score_goes_to_the_more_recent_use();
        test_decay_halves_per_draft_round_and_is_flat_within_a_round();
        test_min_score_keeps_cold_experts_out();
        test_margin_blocks_churn_and_admits_a_clear_winner();
        test_the_victim_is_coldest_then_least_recent();
        test_max_swaps_caps_the_proposals();
        test_hotter_layers_are_visited_in_index_order_under_one_cap();
        test_in_flight_copies_are_bounded_and_not_repeated();
        test_a_failed_copy_keeps_the_old_mapping_and_cools_down();
        test_fully_resident_layers_hold_every_expert_and_never_swap();
        test_a_resident_is_kept_when_nothing_challenges_it();
        test_misuse_is_an_error_not_a_silent_result();
        test_reset_forgets_everything();
        test_adopted_placement_residents_fill_the_slots_and_are_the_victims();
        test_counters_keep_gpu_and_cpu_routes_apart();
        test_fuzz_invariants_hold_through_random_histories();
        std::cout << "dspark_expert_plan_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dspark_expert_plan_test: " << error.what() << '\n';
        return 1;
    }
}
