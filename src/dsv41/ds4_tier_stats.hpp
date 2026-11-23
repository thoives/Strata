// Striped tier diagnostics shared by the queue, tier and model snapshots.
#pragma once

#include <cstdint>

namespace strata::ds4 {

// Bytes are the requested stripe ranges, including blob padding; they are not physical disk traffic.
// A failed or cancelled job's bytes describe its range, not bytes delivered. Snapshots are not transactional.
struct StripeJobCount {
    uint64_t jobs = 0, bytes = 0;
};
struct StripeIoStats {
    StripeJobCount mapped_submitted, direct_submitted;
    StripeJobCount mapped_ok, mapped_failed, mapped_cancelled;
    StripeJobCount direct_ok, direct_fallback, direct_failed, direct_cancelled;
};
struct StripeHitStats {
    uint64_t hits = 0, demand = 0, prefetch = 0;
    uint64_t mapped_only = 0, direct_only = 0, mixed = 0;
};

}  // namespace strata::ds4
