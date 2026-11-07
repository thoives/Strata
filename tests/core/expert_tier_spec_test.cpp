// tests/core/expert_tier_spec_test.cpp - the one --expert-tier grammar (include/strata/core/expert_tier_spec.hpp): every
// case of tests/core/expert_tier_spec_cases.txt, which strata_control/expert_tier.py is checked against too, so the engines,
// setup and strata launch accept and refuse the same entries.  No I/O beyond reading the case file.
//   expert_tier_spec_test [cases.txt]   exit 0: all passed
#include "strata/core/expert_tier_spec.hpp"

#include <cstdio>
#include <fstream>
#include <string>

using namespace strata::core;

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
    std::printf("expert_tier_spec_test: %d cases, %s\n", cases, fails ? "FAILED" : "PASS");
    return fails ? 1 : 0;
}
