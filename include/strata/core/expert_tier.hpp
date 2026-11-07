// include/strata/core/expert_tier.hpp - expert tier plan v1 §8.2: one device, and the set.
//
// `ExpertTier` is one configured device: open read-write, check its size, map it, and (where a CUDA runtime is
// present) register the mapping so copies to the GPU can DMA.  `ExpertTierSet` is the whole tier: open every
// device, check the identity, reuse the placement or refill it, answer the lookups, and run the prompt path's
// per-device copy workers (§7.3).  The tier's contents are fixed after the fill (§4.4); a blob pointer stays
// valid as long as the tier is open (§7.1).  With no tier configured, nothing in this file is touched (acceptance
// criterion 1: byte-identical behaviour without --expert-tier).
#pragma once

#include "strata/core/expert_tier_dev.hpp"
#include "strata/core/expert_tier_layout.hpp"
#include "strata/core/expert_tier_spec.hpp"   // TierSpec, parse_tier_spec: the one --expert-tier grammar

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

class ExpertSource;   // src/core/expert_source.hpp: fill() copies blobs out of it

/// §6.2: the pack's identity as "sha:<64 hex>" over EVERY regular file under `pack` (a single file is a pack of one):
/// its path relative to the pack, its size and its modification time at the file system's own resolution, in sorted path
/// order - so a change to any one file (a same-size rewrite, a swapped shard, a touched mtime) changes the id, which an
/// aggregate "total size + newest mtime" cannot promise.  `exclude` names files that are identified some other way (the
/// profile) and would otherwise make every profile save look like a changed pack.  `extra`: model files the engine reads
/// from outside `pack` (a GGUF read in place): each adds its file name, size and modification time, so replacing one refills
/// too.  False with `err` when nothing is readable or any file's size or time cannot be read.
bool tier_pack_identity(const std::string& pack, const std::vector<std::string>& exclude, std::string& id, std::string& err,
                        const std::vector<std::string>& extra = {});

/// Every regular file under `pack` (or `pack` itself when it is a file): the files a tier device must never be (§5.4).
std::vector<std::string> tier_pack_files(const std::string& pack);

/// One device. Opened once at start, closed at exit; mapping and base stay put for the whole run.
class ExpertTier {
public:
    ExpertTier() = default;
    ~ExpertTier();                      // idempotent: closes if open
    ExpertTier(const ExpertTier&) = delete;
    ExpertTier& operator=(const ExpertTier&) = delete;
    bool open(const TierSpec& s, std::string& err);   // open rw, size >= mib, mmap/MapViewOfFile, register
    void close();                                       // flushes a refill marker if the run asked for one
    const std::string& path() const { return spec_.path; }
    uint64_t capacity() const;                          // configured capacity, bytes (after reserve)
    uint8_t* base() const;                          // the mapping's base, or null (write-side; the tier owns it)
    bool registered() const;                            // cudaHostRegister succeeded (false: pageable today)
    bool read_header(TierHeader& h, std::string& why) const;   // header A, else its twin B (§6.4)
    bool write_header(const TierHeader& h, std::string& err);  // A then B, flushed; fill_state last (§6.4)
    uint64_t size() const;                                // the mapped file's length
    bool flush();                                         // msync / FlushViewOfFile + FlushFileBuffers; false: not on disk
    void note_refill();                                   // a chunk failed at read time: mark this device for refill
    bool refill_requested() const;
    /// Whether a fill may write over this device: a tier header, or every byte the fill may write (the mib) is zero (§5.4).
    bool blank_or_tier() const;
    const TierFileId& file_id() const { return id_; }   // what the device is (expert_tier_dev.hpp), for the overlap checks
    // Tests only: makes flush() fail when it returns true (a full disk, an I/O error), so the persistence paths can be checked.
    void set_flush_fault(std::function<bool()> f) { flush_fault_ = std::move(f); }

private:
    TierSpec spec_{};
    TierFileId id_{};
    std::function<bool()> flush_fault_;
    int posix_fd_ = -1;          // POSIX: the open read-write handle
    void* os_handle_ = nullptr;  // Windows: the file-mapping handle
    void* win_fh_ = nullptr;     // Windows: the read-write file handle
    void* mapping_ = nullptr;
    uint8_t* base_ = nullptr;
    uint64_t size_ = 0;
    bool registered_ = false;
    bool refill_ = false;   // close() writes fill_state = 1 when set (§6.4's mark-for-refill)
};

/// The whole tier: the devices in configuration order (order is part of the placement, §6.3), the lookup index
/// in RAM (12 bytes per expert, §8.2), and the fill.  A checksum failure found at start (verify=full) falls
/// back to the file tier per blob and marks the device for refill at the next start (§6.4).
class ExpertTierSet {
public:
    // Opens every device in order and decides reuse or refill (§6.3). A device that cannot be opened, mapped
    // or sized stops the start with the reason (§5.4); nothing half-opens.  So does a device that is another device of
    // this set (by identity, not by path), one of the `protect` files (the model's files, the profile) or the block device
    // they live on, or one that holds data that is neither blank nor an expert tier (tier_fd_blank_or_tier, expert_tier_dev.hpp).
    bool open(const std::vector<TierSpec>& specs, const TierIdentity& id, std::string& err, bool& reused,
              const std::vector<std::string>& protect = {});
    // Plans (plan_expert_tiers over the caller's ranked remainder) and copies every tier-held blob once,
    // then writes the headers with fill_state = 2 last. Records the fill time and bytes for the startup log.
    bool fill(ExpertSource& src, const std::vector<std::pair<int32_t, int32_t>>& ranked_rest,
              const TierIdentity& id, std::string& err);
    const uint8_t* blob(int64_t layer, int64_t expert) const;   // nullptr: not held
    bool has(int64_t layer, int64_t expert) const;
    bool registered_of(int64_t layer, int64_t expert) const;   // §7.6: the holding device registered (pinned() reads it)
    // Prompt path: copies the held blobs of a batch, one worker thread per device, in rank order (§7.3).
    // The worker cap per device starts at 2 and is a measured value on hardware (P0-hw).
    bool copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n,
                    uint64_t& bytes_out);
    void set_worker_cap(int n);
    uint64_t bytes_held() const;
    int64_t experts_held() const;
    // Fill-phase results for the startup log (§8.5); "tier reused" when nothing was copied.
    double fill_seconds() const;
    uint64_t fill_bytes() const;
    const std::vector<std::string>& notes() const;   // per-device one-liners, and refill/verify notes
    void close();
    // Tests only: `f(device)` true makes that device's next flush fail (expert_tier_test's persistence-failure cases).
    void set_flush_fault(std::function<bool(size_t)> f);

private:
    TierIdentity id_;
    // internals used by the fill and by the open/reuse decision, kept here so the tests and the source see only
    // the public API: lookup rebuild after reuse or fill, and the §6.4 full verification.
    bool build_lookup(const std::vector<TierPlacementEntry>& entries, int entry_bytes, std::string& why);
    bool check_geometry(std::string& why);   // a reused placement lies inside each device and its chunk table fits
    void clear_lookup();
    void verify_chunks();
    std::vector<std::unique_ptr<ExpertTier>> tiers_;   // the devices, in configuration order
    std::vector<uint64_t> data_off_;  // per device: where its blobs begin (after the header areas and tables)
    std::vector<int32_t> dev_of_;     // [layer * n_expert + expert] -> device, -1 not held
    std::vector<uint64_t> off_of_;    // the same, byte offset into the device's data region
    std::vector<uint8_t> valid_;      // 0: a bad chunk made this expert's tier copy untrustworthy
    std::vector<std::vector<uint8_t>> bad_chunk_;   // verify=full failures: 1 = chunk untrustworthy
    std::vector<uint64_t> used_;      // per device: bytes filled in the data region
    int worker_cap_ = 2;
    uint64_t bytes_held_ = 0;
    int64_t experts_held_ = 0;
    double fill_s_ = 0;
    uint64_t fill_b_ = 0;
    std::vector<std::string> notes_;
    std::function<bool(size_t)> flush_fault_;   // tests only (set_flush_fault)
};

}  // namespace strata::core
