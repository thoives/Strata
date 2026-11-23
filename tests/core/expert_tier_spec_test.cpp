// tests/core/expert_tier_spec_test.cpp - the one --expert-tier grammar (include/strata/core/expert_tier_spec.hpp): every
// case of tests/core/expert_tier_spec_cases.txt, which strata_control/expert_tier.py is checked against too, so the engines,
// setup and strata launch accept and refuse the same entries.  No I/O beyond reading the case file.
//   expert_tier_spec_test [cases.txt]   exit 0: all passed
#include "strata/core/expert_tier_spec.hpp"

#include <cstdio>
#include <fstream>
#include <string>

using namespace strata::core;


// The layout options (--expert-tier-layout, --expert-tier-stripe-kib): the same rules strata_control/expert_tier.py follows.
static int layout_checks() {
    int fails = 0;
    auto expect = [&](bool ok, const char* what) {
        if (!ok) { std::printf("FAIL: %s\n", what); ++fails; }
    };
    TierLayout l = TierLayout::Whole;
    expect(parse_tier_layout_name("whole", l) && l == TierLayout::Whole, "whole is a layout");
    expect(parse_tier_layout_name("STRIPED", l) && l == TierLayout::Striped, "striped is a layout in any case");
    expect(!parse_tier_layout_name("stripe", l), "an unknown layout name is refused");
    expect(!parse_tier_layout_name("", l), "an empty layout name is refused");
    uint32_t kib = 0;
    std::string why;
    expect(parse_stripe_kib("256", kib, why) && kib == 256, "256 KiB is a stripe");
    expect(parse_stripe_kib("4", kib, why) && kib == 4, "4 KiB is the smallest stripe");
    expect(parse_stripe_kib("16384", kib, why) && kib == 16384, "16384 KiB is the largest stripe");
    expect(!parse_stripe_kib("0", kib, why), "0 KiB is refused");
    expect(!parse_stripe_kib("6", kib, why), "6 KiB is not a multiple of 4");
    expect(!parse_stripe_kib("16388", kib, why), "a stripe above 16384 KiB is refused");
    expect(!parse_stripe_kib("-4", kib, why) && !parse_stripe_kib("1e3", kib, why) && !parse_stripe_kib("", kib, why),
           "a sign, an exponent or nothing is not a stripe size");
    expect(!why.empty(), "a refused stripe size gives a reason");
    expect(tier_stripe_kib(TierLayoutOpts{}) == 256, "the default stripe is 256 KiB");

    TierLayoutOpts o;   // nothing given: fine with no tier and with one (whole, as before)
    expect(validate_tier_layout(o, 0, why) && validate_tier_layout(o, 1, why), "nothing given is always fine");
    o.layout_given = true;   // a layout with no tier would do nothing: refused, so a typo does not look like it took effect
    expect(!validate_tier_layout(o, 0, why), "a layout with no --expert-tier is refused");
    o.layout = TierLayout::Striped;
    expect(!validate_tier_layout(o, 1, why), "striped with one device is refused");
    expect(validate_tier_layout(o, 2, why), "striped with two devices is fine");
    expect(validate_tier_layout(o, kTierStripedMaxDevices, why) && !validate_tier_layout(o, kTierStripedMaxDevices + 1, why),
           "striped takes at most the bounded number of devices");
    TierLayoutOpts s;   // a stripe size with the whole layout has no stripes to size: refused
    s.stripe_kib = 64;
    expect(!validate_tier_layout(s, 2, why), "a stripe size with the whole layout is refused");
    s.layout = TierLayout::Striped;
    s.layout_given = true;
    expect(validate_tier_layout(s, 2, why) && tier_stripe_kib(s) == 64, "striped with a stripe size is fine");
    TierLayoutOpts none_given_but_striped_default;   // striped with the default stripe: the caller applies 256 KiB
    none_given_but_striped_default.layout = TierLayout::Striped;
    none_given_but_striped_default.layout_given = true;
    expect(validate_tier_layout(none_given_but_striped_default, 3, why) && tier_stripe_kib(none_given_but_striped_default) == 256,
           "striped without a stripe size uses 256 KiB");
    return fails;
}

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : std::string(__FILE__);
    if (argc <= 1) path = path.substr(0, path.find_last_of("/\\") + 1) + "expert_tier_spec_cases.txt";
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::printf("expert_tier_spec_test: cannot read %s\n", path.c_str()); return 1; }
    int fails = 0, cases = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const size_t t1 = line.find('\t');
        if (t1 == std::string::npos) { std::printf("bad case line: %s\n", line.c_str()); ++fails; continue; }
        const std::string kind = line.substr(0, t1);
        const size_t t2 = line.find('\t', t1 + 1);
        const std::string entry = line.substr(t1 + 1, t2 == std::string::npos ? std::string::npos : t2 - t1 - 1);
        const std::string canon = t2 == std::string::npos ? std::string() : line.substr(t2 + 1);
        ++cases;
        TierSpec s;
        std::string why;
        const bool ok = parse_tier_spec(entry, s, why);
        if (kind == "ok") {
            if (!ok) { std::printf("FAIL: '%s' refused (%s)\n", entry.c_str(), why.c_str()); ++fails; continue; }
            if (tier_spec_canonical(s) != canon) {
                std::printf("FAIL: '%s' -> '%s', want '%s'\n", entry.c_str(), tier_spec_canonical(s).c_str(), canon.c_str());
                ++fails;
            }
            TierSpec again;   // the canonical form parses to the same entry
            if (!parse_tier_spec(canon, again, why) || tier_spec_canonical(again) != canon) {
                std::printf("FAIL: canonical '%s' does not round-trip\n", canon.c_str());
                ++fails;
            }
        } else if (kind == "bad") {
            if (ok) { std::printf("FAIL: '%s' accepted as '%s'\n", entry.c_str(), tier_spec_canonical(s).c_str()); ++fails; }
            else if (why.empty()) { std::printf("FAIL: '%s' refused without a reason\n", entry.c_str()); ++fails; }
        } else {
            std::printf("bad case kind: %s\n", kind.c_str());
            ++fails;
        }
    }
    if (cases < 20) { std::printf("FAIL: only %d cases read from %s\n", cases, path.c_str()); ++fails; }
    fails += layout_checks();
    std::printf("expert_tier_spec_test: %d cases, %s\n", cases, fails ? "FAILED" : "PASS");
    return fails ? 1 : 0;
}
