// tests/core/tier_sim.hpp - §10 of docs/EXPERT_TIER_PLAN.md: the test-only shim (P0-sim).
//
// It wraps any `ExpertSource` and adds a delay per blob copy from a configured bandwidth and latency, recording
// when every copy happened. A test can then check the worker counts and the §7.3 split by the ORDER of the
// intervals — never by a speed number. It is not in the engine and not in a release build, and no environment
// variable turns it on (§10). Every value here is simulated; nothing is a claim about hardware.
#pragma once

#include "strata/core/expert_source.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

class SimSource : public ExpertSource {
public:
    // inner: where the bytes come from. bandwidth_gbps / latency_us: the simulated link (§10's "a bandwidth and
    // a latency"). layer_bytes: the blob size per layer, so the simulated time matches the copy's size.
    SimSource(ExpertSource& inner, std::vector<uint64_t> layer_bytes, double bandwidth_gbps, uint64_t latency_us)
        : inner_(inner), layer_bytes_(std::move(layer_bytes)), gbps_(bandwidth_gbps), latency_us_(latency_us),
          t0_(std::chrono::steady_clock::now()) {}

    const uint8_t* blob(int64_t layer, int64_t expert) override { return inner_.blob(layer, expert); }

    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override {
        const int64_t a = now_us();
        const bool ok = inner_.copy_blob(layer, expert, dst);
        const uint64_t bytes =
            layer >= 0 && (size_t) layer < layer_bytes_.size() ? layer_bytes_[(size_t) layer] : 0;
        // The simulated link's time for this blob: bytes at gbps (1 GiB = 2^30 bytes), plus the latency. Labelled
        // simulated: not a measurement, and no claim is made from it (§11).
        const double us = (double) bytes / (gbps_ * 1073741824.0) * 1000000.0 + (double) latency_us_;
        std::this_thread::sleep_until(std::chrono::steady_clock::now() +
                                      std::chrono::microseconds((int64_t) us));
        const int64_t b = now_us();
        std::lock_guard<std::mutex> lk(mu_);
        intervals_.push_back({a, b, layer, expert});
        return ok;
    }

    // When each shimmed copy happened, in microseconds since this source was created: the timing order a test
    // reads (§12.4: "checked by the timing order, not by a speed number").
    struct Copy {
        int64_t t0_us, t1_us;
        int64_t layer, expert;
    };
    std::vector<Copy> intervals() const {
        std::lock_guard<std::mutex> lk(mu_);
        return intervals_;
    }
    int64_t now_us() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0_)
            .count();
    }

private:
    ExpertSource& inner_;
    std::vector<uint64_t> layer_bytes_;
    double gbps_;
    uint64_t latency_us_;
    std::chrono::steady_clock::time_point t0_;
    mutable std::mutex mu_;
    std::vector<Copy> intervals_;
};

}  // namespace strata::core
