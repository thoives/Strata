// include/strata/core/expert_tier_spec.hpp - one grammar for `--expert-tier`, shared by both engines.
//
//   PATH,mib=N[,reserve=M][,verify=header|full]
//
// The rules, in plain words (strata_control/expert_tier.py follows the same ones, and its tests and
// tests/core/expert_tier_spec_test.cpp check the same list of entries):
//   - PATH is everything before the first comma, kept exactly as given (a space is part of a file name). It may not be empty.
//   - every other field is key=value. A key is mib, reserve or verify, in any case (MIB=1 is mib=1). A field with no `=`,
//     an empty field (two commas in a row, or a comma at the end) or a key given twice is an error.
//   - mib and reserve are whole numbers of MiB: ASCII digits only (no sign, no space, no "0x"), and small enough that the
//     size in bytes fits 64 bits. mib is required and at least 1; reserve is less than mib.
//   - verify is header or full, in any case.
// No I/O here: the engines open the device afterwards (expert_tier_dev.hpp).
#pragma once

#include <cstdint>
#include <string>

namespace strata::core {

/// One `--expert-tier` entry, parsed.
struct TierSpec {
    std::string path;
    uint64_t mib = 0;
    uint64_t reserve_mib = 0;
    bool verify_full = false;
};

/// The largest mib whose size in bytes (mib << 20) still fits 64 bits.
constexpr uint64_t kTierMaxMib = UINT64_MAX >> 20;

inline uint64_t tier_capacity_bytes(const TierSpec& s) {   // capacity after the reserve, in bytes
    return (s.mib > s.reserve_mib ? s.mib - s.reserve_mib : 0) << 20;
}

namespace tier_spec_detail {
inline std::string lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = (char) (c - 'A' + 'a');
    return s;
}
inline bool mib_number(const std::string& v, uint64_t& out) {
    if (v.empty()) return false;
    uint64_t n = 0;
    for (char c : v) {
        if (c < '0' || c > '9') return false;
        const uint64_t d = (uint64_t) (c - '0');
        if (n > (kTierMaxMib - d) / 10) return false;   // past kTierMaxMib: the byte size would not fit
        n = n * 10 + d;
    }
    out = n;
    return true;
}
}  // namespace tier_spec_detail

/// Parses one entry. False with `why` (plain words) for anything the rules above refuse; `s` is then unspecified.
inline bool parse_tier_spec(const std::string& raw, TierSpec& s, std::string& why) {
    using namespace tier_spec_detail;
    s = TierSpec{};
    const size_t c0 = raw.find(',');
    if (c0 == std::string::npos) { why = "expected PATH,mib=N[,reserve=M][,verify=header|full]"; return false; }
    s.path = raw.substr(0, c0);
    if (s.path.empty()) { why = "the entry starts with the device path, and it is empty"; return false; }
    bool got_mib = false, got_reserve = false, got_verify = false;
    size_t pos = c0 + 1;
    for (;;) {
        const size_t nx = raw.find(',', pos);
        const std::string part = raw.substr(pos, nx == std::string::npos ? std::string::npos : nx - pos);
        if (part.empty()) { why = "an empty field (two commas in a row, or a comma at the end)"; return false; }
        const size_t eq = part.find('=');
        if (eq == std::string::npos) { why = "'" + part + "' is not key=value (mib=N, reserve=M or verify=header|full)"; return false; }
        const std::string k = lower(part.substr(0, eq));
        const std::string v = part.substr(eq + 1);
        if (k == "mib") {
            if (got_mib) { why = "mib is given twice"; return false; }
            got_mib = true;
            if (!mib_number(v, s.mib)) { why = "mib takes a whole number of MiB (digits only), not '" + v + "'"; return false; }
        } else if (k == "reserve") {
            if (got_reserve) { why = "reserve is given twice"; return false; }
            got_reserve = true;
            if (!mib_number(v, s.reserve_mib)) { why = "reserve takes a whole number of MiB (digits only), not '" + v + "'"; return false; }
        } else if (k == "verify") {
            if (got_verify) { why = "verify is given twice"; return false; }
            got_verify = true;
            const std::string lv = lower(v);
            if (lv == "full") s.verify_full = true;
            else if (lv != "header") { why = "verify takes header or full, not '" + v + "'"; return false; }
        } else {
            why = "unknown key '" + part.substr(0, eq) + "' (mib, reserve or verify)";
            return false;
        }
        if (nx == std::string::npos) break;
        pos = nx + 1;
    }
    if (!got_mib) { why = "mib=N is required (the MiB of the device the tier may use)"; return false; }
    if (s.mib == 0) { why = "mib must be at least 1"; return false; }
    if (s.reserve_mib >= s.mib) { why = "reserve=M must be less than mib=N (the rest is what the tier holds)"; return false; }
    return true;
}

/// The canonical spelling of a parsed entry: what setup and strata launch write into a run config.
inline std::string tier_spec_canonical(const TierSpec& s) {
    std::string out = s.path + ",mib=" + std::to_string((unsigned long long) s.mib);
    if (s.reserve_mib) out += ",reserve=" + std::to_string((unsigned long long) s.reserve_mib);
    if (s.verify_full) out += ",verify=full";
    return out;
}

// ---- the layout of the whole --expert-tier set (docs/EXPERT_TIER.md, "The layout").  It applies to every device given:
//   --expert-tier-layout whole|striped   whole (the default): each expert is stored on one device
//   --expert-tier-stripe-kib N           striped only: the stripe unit, a multiple of 4 KiB from 4 to 16384; default 256
// Refused before anything is written: a layout option with no --expert-tier (a typo would otherwise look as if it took
// effect), a stripe size with the whole layout, and striped with fewer than 2 or more than kTierStripedMaxDevices devices.
// strata_control/expert_tier.py follows the same rules (tests/core/expert_tier_spec_test.cpp and the Python tests check them).
enum class TierLayout : uint8_t { Whole = 0, Striped = 1 };
constexpr uint32_t kTierStripeKibDefault = 256;
constexpr uint32_t kTierStripeKibMin = 4;
constexpr uint32_t kTierStripeKibMax = 16384;
constexpr size_t kTierStripedMaxDevices = 8;   // bounds the placement table: 12 + 8 bytes per device for each expert

struct TierLayoutOpts {
    TierLayout layout = TierLayout::Whole;
    bool layout_given = false;   // --expert-tier-layout was given (whole or striped)
    uint32_t stripe_kib = 0;     // --expert-tier-stripe-kib as given; 0: not given
};

/// The stripe unit in KiB that a striped set uses (the default when the flag was not given).
inline uint32_t tier_stripe_kib(const TierLayoutOpts& o) { return o.stripe_kib ? o.stripe_kib : kTierStripeKibDefault; }

/// "whole" or "striped", in any case. False for anything else.
inline bool parse_tier_layout_name(const std::string& raw, TierLayout& out) {
    const std::string v = tier_spec_detail::lower(raw);
    if (v == "whole") { out = TierLayout::Whole; return true; }
    if (v == "striped") { out = TierLayout::Striped; return true; }
    return false;
}

/// A stripe size in KiB: digits only, a multiple of 4 and from kTierStripeKibMin to kTierStripeKibMax. False with `why`.
inline bool parse_stripe_kib(const std::string& v, uint32_t& out, std::string& why) {
    uint64_t n = 0;
    if (!tier_spec_detail::mib_number(v, n) || n > kTierStripeKibMax) {
        why = "--expert-tier-stripe-kib takes a whole number of KiB from " + std::to_string(kTierStripeKibMin) + " to " +
              std::to_string(kTierStripeKibMax) + " (a multiple of 4), not '" + v + "'";
        return false;
    }
    if (n < kTierStripeKibMin || n % 4 != 0) {
        why = "--expert-tier-stripe-kib must be a multiple of 4 KiB, from " + std::to_string(kTierStripeKibMin) + " to " +
              std::to_string(kTierStripeKibMax) + ", not '" + v + "'";
        return false;
    }
    out = (uint32_t) n;
    return true;
}

/// Whether the layout options are usable with `n_devices` --expert-tier entries. Nothing given is always fine (whole, as before).
inline bool validate_tier_layout(const TierLayoutOpts& o, size_t n_devices, std::string& why) {
    if (!o.layout_given && o.stripe_kib == 0) return true;
    if (n_devices == 0) {
        why = "--expert-tier-layout and --expert-tier-stripe-kib need at least one --expert-tier (with no tier they would do nothing)";
        return false;
    }
    if (o.stripe_kib != 0 && o.layout != TierLayout::Striped) {
        why = "--expert-tier-stripe-kib is for --expert-tier-layout striped (the whole layout has no stripes)";
        return false;
    }
    if (o.layout == TierLayout::Striped) {
        if (n_devices < 2) {
            why = "--expert-tier-layout striped needs at least two --expert-tier devices (it spreads each expert over them)";
            return false;
        }
        if (n_devices > kTierStripedMaxDevices) {
            why = "--expert-tier-layout striped takes at most " + std::to_string(kTierStripedMaxDevices) + " --expert-tier devices";
            return false;
        }
    }
    return true;
}

}  // namespace strata::core
