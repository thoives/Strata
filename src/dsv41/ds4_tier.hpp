// Strata. src/dsv41/ds4_tier.hpp - the expert tier of the DeepSeek-V4.1 engine (docs/EXPERT_TIER_DSV41_PLAN.md): the band
// below the RAM store (ds4_store.hpp) and above the file reads.  A tier is one or more configured devices (a disk node,
// a file on a host-visible mount); each holds whole experts in the store's own slot layout (E2: slot_bytes = the store's
// slot: the blob with kSlack before and after, the matrices at tr[r]), so a tier hit copies bytes into a store slot with
// the store's own IoQueue::Job mem form (ds4_pool.hpp:217-218), and the tier's fill reads through the store's job builder
// into its mapping (§9).
//
// Contents are decided once from the profile (D2): the planner is the shared uniform deal (include/strata/core/
// expert_tier_plan.hpp, E1), given capacity in bytes with every blob priced at slot_bytes, so capacity lands in whole
// slots (§6.2).  The list it gets is the profile order minus the first vram_guard minus the first store_guard: the guard
// rule (§5.2) - planned against the guaranteed minimums, a store that shrinks later (store_watch) never leaves a rank
// with no home, and one that keeps more only duplicates tier copies.
//
// Persistence (D6, D7): each device carries the placement in its own header (A and its twin B: a torn write falls to the
// other) with the same CRC32C, placement-table and chunk-table codecs as the shared plan (include/strata/core/
// expert_tier_layout.hpp) plus this engine's fields (§7.3: pack identity, the profile's sha as information only, the
// store's slot size and layout version, the guards).  fill_state is written last: a start with a torn or corrupt header,
// a changed pack, slot size, layout, shape or guard refills the whole tier; a profile change does not (§7.2).
//
// One IoQueue per device (E3, DS4_TIER_THREADS workers, default 2, §8.6): a narrow link stays busy with its own queue and
// a slow device never blocks a fast one.  The tier is host-visible only (D1): the GPU keeps reading store slots (§8.3);
// v1 does not map it for the device (open item 2).  No speed claim lives in this file.
#pragma once

#include "ds4_store.hpp"   // ExpertStore's kSlack / LayerGeo: the tier's slot is the store's slot (E2)

#include "strata/core/expert_tier_dev.hpp"
#include "strata/core/expert_tier_layout.hpp"
#include "strata/core/expert_tier_plan.hpp"
#include "strata/core/expert_tier_spec.hpp"   // the one --expert-tier grammar, shared with the Qwen engine

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#if !defined(_WIN32)
#include <sys/file.h>   // flock: the tier is held by one process at a time (§5.4)
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::ds4 {

/// One `--expert-tier PATH,mib=N[,reserve=M][,verify=header|full]` entry (§11: parsed in init_tier; strata launch and setup
/// pass the canonical string through args).  The grammar is the shared one (include/strata/core/expert_tier_spec.hpp), so
/// the two engines, setup and strata launch accept and refuse the same entries.
using TierSpec = strata::core::TierSpec;
using strata::core::parse_tier_spec;
using strata::core::tier_capacity_bytes;

/// §7.1: the pack's identity, "sha:" over the sorted (shard name, size, modification time in the file system's own
/// resolution) list, and the effective expert geometry (`geometry`: a digest of where every expert's bytes sit in the shards -
/// the file, the offset and the size of each piece, as the loader laid them out). The geometry is in the identity because the
/// same files with moved bytes (a re-packed or re-ordered model that kept the names, sizes and times) would otherwise keep the
/// old tier's identity and serve the wrong bytes: a change to it refills the tier (§7.1, the metadata case).  The shards are
/// found exactly where the loader opens them (Pack::load: model_dir + "/" + name), whatever the working directory is; a shard
/// whose size or time cannot be read is an error, not a zero (a zero would let two different model folders share one
/// identity).  The folder itself is not part of the identity: the same files moved elsewhere are the same pack.
inline bool ds4_pack_identity(const std::string& model_dir, const std::vector<std::string>& files, const std::string& geometry,
                              std::string& id, std::string& err) {
    namespace fs = std::filesystem;
    if (files.empty()) { err = "the pack names no model files"; return false; }
    if (geometry.empty()) { err = "the pack's expert geometry is not known"; return false; }
    std::vector<std::string> names = files;
    std::sort(names.begin(), names.end());
    std::string list;
    for (const std::string& f : names) {
        const fs::path p = fs::path(model_dir + "/" + f);
        std::error_code ec;
        const auto sz = fs::file_size(p, ec);
        if (ec || sz == 0) { err = "cannot read the size of " + p.string() + (ec ? ": " + ec.message() : std::string(": empty")); return false; }
        const auto mt = fs::last_write_time(p, ec);
        if (ec) { err = "cannot read the time of " + p.string() + ": " + ec.message(); return false; }
        list += f + '\t' + std::to_string((unsigned long long) sz) + '\t' + std::to_string((long long) mt.time_since_epoch().count()) + '\n';
    }
    list += "geometry\t" + geometry + '\n';
    id = "sha:" + strata::core::tier_sha256_hex((const uint8_t*) list.data(), list.size());
    return true;
}

// ---- the tier's on-disk header (§7.3) -----------------------------------------------------------------------
// The shared plan's shape (A, its twin B, then the tables) with this engine's fields; 176 pinned little-endian bytes,
// header_crc = CRC32C over the 172 before it.  A header whose crc does not match is read from its twin (B if A fails),
// and both failing means the device is refilled - the same rule as the shared plan's §6.4.
constexpr const char kDs4TierMagic[8] = { 'S', 'T', 'R', 'P', 'T', 'D', 'S', '4' };
constexpr uint32_t kDs4TierVersion = 1;         // this header struct
constexpr uint32_t kDs4SlotLayoutVersion = 1;   // the store's slot layout (LayerGeo's tr[], kSlack): a change refills
constexpr uint64_t kTierHeaderArea = 64ull << 10;   // header A's area at 0, header B's at 64 KiB
constexpr uint64_t kTierChunk = strata::core::kTierChunk;   // the chunk CRC unit: 4 MiB
constexpr int kTierHeaderBytes = 176;

struct TierHdr {
    char magic[8];
    uint32_t version = kDs4TierVersion;
    uint32_t device_index = 0, device_count = 0;
    char pack_id[32] = {};      // §7.1: "sha:<64 hex>" over the sorted shard list (name, size, mtime), or the hashes
    char profile_sha[32] = {};  // §7.2: information only: a changed profile does not force a refill
    uint32_t slot_layout = kDs4SlotLayoutVersion;
    uint32_t fill_state = 0;    // 0 empty, 1 filling, 2 complete: written last
    uint32_t pad0 = 0;
    uint32_t slots32 = 0;       // this device's slots at fill time (slots_per_device)
    uint64_t slot_bytes = 0;    // the store's slot size at fill time: a start whose slot size differs refills
    uint64_t n_layers = 0, n_expert = 0;
    uint64_t vram_guard = 0, store_guard = 0;   // the guards the placement was planned against (§5.2)
    uint64_t n_held = 0;        // placement entries: the whole placement, identical on every device (§6.3)
    uint64_t chunk_off = 0, chunk_count = 0;
    uint32_t placement_crc = 0;
    uint32_t header_crc = 0;
};
static_assert(sizeof(TierHdr) == kTierHeaderBytes, "the tier header's fields and their padding are pinned (§7.3)");

namespace ds4_tier {

inline void put32(uint8_t* b, uint32_t v) { for (int i = 0; i < 4; ++i) b[i] = (uint8_t)(v >> (8 * i)); }
inline void put64(uint8_t* b, uint64_t v) { put32(b, (uint32_t)v); put32(b + 4, (uint32_t)(v >> 32)); }
inline uint32_t get32(const uint8_t* b) {
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}
inline uint64_t get64(const uint8_t* b) { return (uint64_t)get32(b) | ((uint64_t)get32(b + 4) << 32); }

inline void th_encode(const TierHdr& h, uint8_t* buf) {
    std::memcpy(buf, h.magic, 8);
    uint8_t* b = buf + 8;
    put32(b, h.version); b += 4;
    put32(b, h.device_index); b += 4;
    put32(b, h.device_count); b += 4;
    std::memcpy(b, h.pack_id, 32); b += 32;
    std::memcpy(b, h.profile_sha, 32); b += 32;
    put32(b, h.slot_layout); b += 4;
    put32(b, h.fill_state); b += 4;
    put32(b, h.pad0); b += 4;
    put32(b, h.slots32); b += 4;
    put64(b, h.slot_bytes); b += 8;
    put64(b, h.n_layers); b += 8;
    put64(b, h.n_expert); b += 8;
    put64(b, h.vram_guard); b += 8;
    put64(b, h.store_guard); b += 8;
    put64(b, h.n_held); b += 8;
    put64(b, h.chunk_off); b += 8;
    put64(b, h.chunk_count); b += 8;
    put32(b, h.placement_crc); b += 4;   // 164-167
    put32(b, 0); b += 4;                // 168-171: reserved (pad1: the on-disk header is 176 bytes, crc over the first 172)
    put32(b, strata::core::tier_crc32c(buf, 172));   // 172-175: every field before header_crc
}
inline bool th_decode(const uint8_t* buf, TierHdr& h, std::string& why) {
    if (std::memcmp(buf, kDs4TierMagic, 8) != 0) { why = "not a DeepSeek expert tier header (magic)"; return false; }
    if (strata::core::tier_crc32c(buf, 172) != get32(buf + 172)) { why = "header_crc does not match its fields"; return false; }
    TierHdr g{};
    std::memcpy(g.magic, buf, 8);
    const uint8_t* b = buf + 8;
    g.version = get32(b); b += 4;
    if (g.version != kDs4TierVersion) { why = "header version " + std::to_string(g.version) + ", this engine writes " + std::to_string(kDs4TierVersion); return false; }
    g.device_index = get32(b); b += 4;
    g.device_count = get32(b); b += 4;
    std::memcpy(g.pack_id, b, 32); b += 32;
    std::memcpy(g.profile_sha, b, 32); b += 32;
    g.slot_layout = get32(b); b += 4;
    g.fill_state = get32(b); b += 4;
    g.pad0 = get32(b); b += 4;
    g.slots32 = get32(b); b += 4;
    g.slot_bytes = get64(b); b += 8;
    g.n_layers = get64(b); b += 8;
    g.n_expert = get64(b); b += 8;
    g.vram_guard = get64(b); b += 8;
    g.store_guard = get64(b); b += 8;
    g.n_held = get64(b); b += 8;
    g.chunk_off = get64(b); b += 8;
    g.chunk_count = get64(b); b += 8;
    g.placement_crc = get32(b); b += 4;   // 164-167
    b += 4;                            // 168-171: reserved (pad1)
    g.header_crc = get32(b);
    h = g;
    return true;
}

}  // ds4_tier

/// The tier across its devices: open devices in configuration order (order is part of the placement, §6.3/§7.4), the
/// lookup the store consults on its misses (§8.1), one IoQueue per device (§8.6), and the fill (§9).  Owned by the model
/// for the run's life; ds4_store.hpp never includes this file - the store reaches the tier through three std::functions
/// on its Cfg, so the job builder stays the store's own (E2).
class ExpertTierSet {
public:
    struct Fill {   // §9: what the fill phase did (counted in the load statistics, not the decode ones)
        double seconds = 0;
        uint64_t bytes = 0;   // the blobs placed (the store counts the matrices themselves)
        int64_t placed = 0;
        std::vector<int64_t> per_dev;   // (the experts each device took: §9's per-device progress line)
    };

    ExpertTierSet() = default;
    ~ExpertTierSet() { close(); }
    ExpertTierSet(const ExpertTierSet&) = delete;
    ExpertTierSet& operator=(const ExpertTierSet&) = delete;

    /// §5.4: open every device read-write, check its size is at least mib, map it.  The first failure stops the start with
    /// the reason; nothing half-opens.  Duplicate paths are an error (§6.2).  Each device's queue gets
    /// DS4_TIER_THREADS workers (default 2, §8.6).
    /// Also refused, before anything is written: two entries that are one device (by identity: a link, another name, a
    /// partition of the other), a device that is - or holds - one of the `protect` files (the shards, the pack's own files,
    /// the profile), and a device holding data that is neither blank nor an expert tier (tier_fd_blank_or_tier).
    bool open(const std::vector<TierSpec>& specs, std::string& err, const std::vector<std::string>& protect = {});
    void close();   // idempotent; a device marked (a failed copy, an abandoned fill) gets fill_state = 1 on both headers
    /// Finishes every job queued on the devices' queues and joins their threads; afterwards queue(dev) is null (a store
    /// asking falls back to the files).  The jobs write into the STORE's slots and read this tier's mapping, so the order of
    /// teardown is: the store calls this first (its Cfg::tier_drain), then releases its arena; the mappings close after (close()
    /// drains again, a no-op).  Idempotent.
    void drain();

    /// §7.4 + §5.2: reuse or plan.  Reuse: every device's header pair readable, all matching the start (device count,
    /// pack, slot size, slot layout, shape, both guards, fill_state == 2), the placement table's crc matching, and
    /// (verify=full) every stored chunk crc still matching - then lookup is rebuilt from the stored tables and nothing
    /// is copied.  Anything else (including --expert-tier-refill): the shared planner over `ranked_rest` (capacity in
    /// whole slots), every device's header rewritten with fill_state = 1 first, and `fill` below does the copies.
    bool decide(uint64_t slot_bytes, uint64_t vram_guard, uint64_t store_guard, uint64_t n_layers, uint64_t n_expert,
                const std::string& pack_id, const std::string& profile_sha, bool force_refill,
                const std::vector<std::pair<int32_t, int32_t>>& ranked_rest,
                bool& reused, std::string& why, std::vector<std::string>& notes);

    /// §9: place every tier-held expert once (rank order), each expert's three matrices as the store's read jobs into
    /// this device's mapping, on the device's own queue.  `read` starts the jobs (the model wires it to the store) and
    /// counts `left` down; `bad` set means that expert failed: every device is then marked for the next start's refill.
    /// When a device's writes are all done: flush (msync), its chunk CRC table over what it wrote, then the header with
    /// fill_state = 2, A then B (§6.4 of the shared plan: a torn fill refills).
    bool fill(const std::function<bool(int layer, int e, uint8_t* slot_base, int dev, std::atomic<int>& left,
                                       std::atomic<bool>& bad, uint64_t& bytes_out)>& read,
              Fill& out, std::string& err);

    bool find(int layer, int e, int& dev, uint64_t& slot) const {   // a tier hit the store can copy from now (§8.1)
        // readable_ is published with release once the fill's bytes are all down (fill); this acquire pairs with it, so a
        // store thread (the background preload, a decode read) that sees it set also sees the lookup and the tier's bytes
        if (!readable_.load(std::memory_order_acquire) || layer < 0 || e < 0 || (uint64_t)layer >= n_layers_ || (uint64_t)e >= n_expert_ || lookup_.empty()) return false;
        const Held& h = lookup_[(size_t)layer * n_expert_ + (size_t)e];
        if (h.dev < 0 || h.dev >= (int)devs_.size() || devs_[(size_t)h.dev].mark->load(std::memory_order_acquire)) return false;   // (a marked device falls back to the files, §9)
        dev = h.dev; slot = h.slot;
        return true;
    }
    uint8_t* slot_base(int dev, uint64_t k) const {   // the blob's first byte (after kSlack, like a store slot's base)
        if (dev < 0 || dev >= (int)devs_.size() || devs_[(size_t)dev].base == nullptr) return nullptr;
        return devs_[(size_t)dev].base + data_off_ + k * slot_bytes_ + (uint64_t)ExpertStore::kSlack;
    }
    IoQueue* queue(int dev) const { return dev >= 0 && dev < (int)devs_.size() ? devs_[(size_t)dev].io.get() : nullptr; }
    size_t device_count() const { return devs_.size(); }
    const TierSpec& spec_at(size_t i) const { return devs_[(size_t)i].spec; }
    /// A chunk the store's reads of tier bytes hit that does not match (verify=full at a later start cannot catch a
    /// read-time EIO): mark the device, its expert falls back to the file (the caller's) and the tier refills next start.
    void note_refill(int dev);

    bool readable() const { return readable_.load(std::memory_order_acquire); }   // lookups answer only after a reuse or a completed fill
    // Tests only: `f(device)` true makes that device's next flush fail (a full disk, an I/O error).
    void set_flush_fault(std::function<bool(size_t)> f) { flush_fault_ = std::move(f); }
    int64_t experts_held() const { return held_; }
    uint64_t slot_bytes() const { return slot_bytes_; }
    uint64_t data_off() const { return data_off_; }
    const std::vector<std::string>& notes() const { return notes_; }

    struct Held {
        int32_t dev = -1;
        uint64_t slot = 0;
    };

private:
    struct Dev {
        TierSpec spec;
        uint8_t* base = nullptr;
        uint64_t size = 0;
        std::unique_ptr<IoQueue> io;
        TierHdr hdr{};
        bool opened = false;
        bool filling = false;   // this start wrote fill_state 1: its bytes are on the way
        // close() flips fill_state to 1 (§6.4: a bad device refills at the next start).  Atomic: the store's I/O threads set
        // it (note_refill) and read it (find) while the main thread may be filling
        std::unique_ptr<std::atomic<bool>> mark{new std::atomic<bool>(false)};
        size_t index = 0;       // the device's place in the configuration (the flush fault's argument)
#if !defined(_WIN32)
        int fd = -1;
#else
        void* fh = nullptr;
        void* mm = nullptr;
#endif
    };

    std::vector<Dev> devs_;
    uint64_t slot_bytes_ = 0, n_layers_ = 0, n_expert_ = 0, data_off_ = 0;
    std::vector<Held> lookup_;
    std::vector<strata::core::TierPlacementEntry> entries_;   // the plan (fill path) in rank order: fill iterates this
    std::atomic<bool> readable_{false};   // published (release) after a reuse or a completed, flushed fill
    std::function<bool(size_t)> flush_fault_;   // tests only
    int64_t held_ = 0;
    std::vector<std::string> notes_;

    static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }
    bool open_one(Dev& d, std::string& err);
    void close_one(Dev& d);
    bool flush(Dev& d);   // false: the bytes may not be on the disk (msync / FlushViewOfFile / FlushFileBuffers failed)
    bool write_headers(Dev& d, const TierHdr& h);   // A, flush, B, flush (§6.4: the twin written after A); false on any failure
    bool read_headers(Dev& d, TierHdr& h, std::string& why);   // A, else B
};

// ------------------------------------------------------------------------ implementation ---------------------------------------------

inline bool ExpertTierSet::open(const std::vector<TierSpec>& specs, std::string& err, const std::vector<std::string>& protect) {
    close();   // (a set opened again: the previous devices' handles and mappings are released first - and their marks make the next decide write fill_state 1, §6.4)
    if (specs.empty()) { err = "expert tier: no --expert-tier entries"; return false; }
    for (size_t i = 0; i < specs.size(); ++i)
        for (size_t j = i + 1; j < specs.size(); ++j)
            if (specs[i].path == specs[j].path) {
                err = "expert tier: " + specs[i].path + " given twice; device paths must be distinct (§6.2)";
                return false;
            }
    {   // §6.2/§5.4: two names of one device, or a model file, are refused before any device is opened (by what they are, not by the lock)
        std::vector<std::string> pp;
        std::vector<strata::core::TierFileId> pid;
        for (const TierSpec& s : specs) {
            strata::core::TierFileId f;
            pp.push_back(s.path);
            if (!strata::core::tier_path_id(s.path, f)) f = strata::core::TierFileId{};   // (an unopenable device: its own open says why)
            pid.push_back(f);
        }
        if (!strata::core::tier_check_overlaps(pp, pid, protect, err)) return false;
    }
    devs_.clear();
    const int th = std::getenv("DS4_TIER_THREADS") ? std::max(1, std::atoi(std::getenv("DS4_TIER_THREADS"))) : 2;
    for (const TierSpec& s : specs) {
        Dev d;
        d.spec = s;
        if (!open_one(d, err)) {
            drain();
            for (auto& x : devs_) close_one(x);
            devs_.clear();
            return false;
        }
        d.io = std::make_unique<IoQueue>(th);
        d.index = devs_.size();
        devs_.push_back(std::move(d));
    }
    // §6.2: the devices by what they ARE (not by path), and none of them one of the files the model is read from
    std::vector<std::string> paths;
    std::vector<strata::core::TierFileId> ids;
    bool ok = true;
    for (Dev& d : devs_) {
        strata::core::TierFileId id;
#if !defined(_WIN32)
        const bool got = strata::core::tier_fd_id(d.fd, id);
#else
        const bool got = strata::core::tier_handle_id(d.fh, id);
#endif
        if (!got) { err = "expert tier " + d.spec.path + ": cannot tell which file it is"; ok = false; break; }
        paths.push_back(d.spec.path);
        ids.push_back(id);
    }
    if (ok) ok = strata::core::tier_check_overlaps(paths, ids, protect, err);
    if (ok)   // §5.4: written over only when blank or already an expert tier
        for (Dev& d : devs_)
#if !defined(_WIN32)
            if (!strata::core::tier_fd_blank_or_tier(d.fd, d.size, d.spec.mib << 20)) { err = strata::core::tier_foreign_data_message(d.spec.path); ok = false; break; }
#else
            if (!strata::core::tier_handle_blank_or_tier(d.fh, d.size, d.spec.mib << 20)) { err = strata::core::tier_foreign_data_message(d.spec.path); ok = false; break; }
#endif
    if (!ok) {
        drain();
        for (auto& x : devs_) close_one(x);
        devs_.clear();
        return false;
    }
    return true;
}

#if !defined(_WIN32)
inline bool ExpertTierSet::open_one(Dev& d, std::string& err) {
    std::string why;
    const int fd = strata::core::tier_open_rw(d.spec.path, why);   // (a Linux block device O_EXCL: one the system uses is refused)
    if (fd < 0) { err = "expert tier " + d.spec.path + ": " + why; return false; }
    uint64_t dev_size = 0;
    bool is_block = false;
    if (!strata::core::tier_fd_size(fd, dev_size, is_block, why)) {   // (a regular file's length, or a Linux block device's BLKGETSIZE64)
        err = "expert tier " + d.spec.path + ": " + why;
        ::close(fd);
        return false;
    }
    if (dev_size < (d.spec.mib << 20)) {
        char b[192];
        std::snprintf(b, sizeof b, "expert tier %s: device is %.2f GiB, smaller than mib=%llu MiB - not a silent cut",
                      d.spec.path.c_str(), (double)dev_size / 1073741824.0, (unsigned long long)d.spec.mib);
        err = b;
        ::close(fd);
        return false;
    }
    // the lifetime lock (§5.4): a regular file is held exclusively until close(); a block device by its O_EXCL open
    if (!is_block && ::flock(fd, LOCK_EX | LOCK_NB) != 0) {
        err = "expert tier " + d.spec.path + ": " + (errno == EWOULDBLOCK ? std::string(strata::core::tier_busy_file_message()) : std::string(std::strerror(errno)));
        ::close(fd);
        return false;
    }
    void* m = ::mmap(nullptr, (size_t)dev_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) {
        err = "expert tier " + d.spec.path + ": mmap failed (errno " + std::to_string(errno) + ")";
        ::close(fd);
        return false;
    }
    d.fd = fd; d.base = (uint8_t*)m; d.size = dev_size; d.opened = true;
    return true;
}
inline bool ExpertTierSet::flush(Dev& d) {
    if (d.base == nullptr) return false;
    if (flush_fault_ && flush_fault_(d.index)) return false;   // (tests: an I/O error at this flush)
    return ::msync((void*)d.base, (size_t)d.size, MS_SYNC) == 0;
}
inline void ExpertTierSet::close_one(Dev& d) {
    if (!d.opened) return;
    if (d.mark->load() && d.base != nullptr) {   // fill_state = 1 on both headers, each header's crc kept true (§6.4)
        for (int which = 0; which < 2; ++which) {
            uint8_t* hp = d.base + (size_t)which * kTierHeaderArea;
            TierHdr cur{};
            std::string why;
            if (ds4_tier::th_decode(hp, cur, why)) { cur.fill_state = 1; ds4_tier::th_encode(cur, hp); }
        }
        flush(d);
    }
    ::munmap((void*)d.base, (size_t)d.size);
    if (d.fd >= 0) ::close(d.fd);
    d.base = nullptr; d.fd = -1; d.opened = false;
}
#else
inline bool ExpertTierSet::open_one(Dev& d, std::string& err) {
    HANDLE fh = CreateFileA(d.spec.path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (!fh || fh == INVALID_HANDLE_VALUE) {
        err = "expert tier " + d.spec.path + ": cannot open read-write (Windows error " + std::to_string((unsigned long)GetLastError()) + ")";
        return false;
    }
    uint64_t dev_size = 0;
    std::string why;
    if (!strata::core::tier_handle_size((void*)fh, dev_size, why)) {
        err = "expert tier " + d.spec.path + ": " + why;
        CloseHandle(fh);
        return false;
    }
    if (dev_size < (d.spec.mib << 20)) {
        err = "expert tier " + d.spec.path + ": device smaller than mib - not a silent cut";
        CloseHandle(fh);
        return false;
    }
    OVERLAPPED lock_at{};   // the lifetime lock (§5.4), a byte range past the end: as ExpertTier::open (not built or run here)
    lock_at.Offset = (DWORD)(dev_size & 0xFFFFFFFFull);
    lock_at.OffsetHigh = (DWORD)(dev_size >> 32);
    if (!LockFileEx(fh, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &lock_at)) {
        err = "expert tier " + d.spec.path + ": " + strata::core::tier_busy_file_message();
        CloseHandle(fh);
        return false;
    }
    HANDLE mm = CreateFileMappingA(fh, nullptr, PAGE_READWRITE, 0, 0, nullptr);
    if (!mm) { err = "expert tier " + d.spec.path + ": CreateFileMapping failed (Windows error " + std::to_string((unsigned long)GetLastError()) + ")"; CloseHandle(fh); return false; }
    void* p = MapViewOfFile(mm, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
    if (!p) { err = "expert tier " + d.spec.path + ": MapViewOfFile failed (Windows error " + std::to_string((unsigned long)GetLastError()) + ")"; CloseHandle(mm); CloseHandle(fh); return false; }
    d.fh = (void*)fh; d.mm = (void*)mm; d.base = (uint8_t*)p; d.size = dev_size; d.opened = true;
    return true;
}
inline bool ExpertTierSet::flush(Dev& d) {
    if (d.base == nullptr) return false;
    if (flush_fault_ && flush_fault_(d.index)) return false;   // (tests: an I/O error at this flush)
    if (!FlushViewOfFile((LPCVOID)d.base, 0)) return false;           // the dirty pages of the view to the file
    return d.fh == nullptr || FlushFileBuffers((HANDLE)d.fh) != 0;   // and the file's cache to the disk
}
inline void ExpertTierSet::close_one(Dev& d) {
    if (!d.opened) return;
    if (d.mark->load() && d.base != nullptr) {
        for (int which = 0; which < 2; ++which) {
            uint8_t* hp = d.base + (size_t)which * kTierHeaderArea;
            TierHdr cur{};
            std::string why;
            if (ds4_tier::th_decode(hp, cur, why)) { cur.fill_state = 1; ds4_tier::th_encode(cur, hp); }
        }
    }
    if (d.mark->load() && d.base != nullptr) (void) flush(d);   // (best effort at close: a failure leaves the twin, or a refill)
    if (d.base) UnmapViewOfFile(d.base);
    if (d.mm) CloseHandle((HANDLE)d.mm);
    if (d.fh) CloseHandle((HANDLE)d.fh);
    d.base = nullptr; d.mm = d.fh = nullptr; d.opened = false;
}
#endif

inline bool ExpertTierSet::read_headers(Dev& d, TierHdr& h, std::string& why) {
    if (d.base == nullptr) { why = "device is closed"; return false; }
    TierHdr ha {}, hb {};
    std::string wa, wb;
    const bool a_ok = ds4_tier::th_decode(d.base, ha, wa);
    const bool b_ok = ds4_tier::th_decode(d.base + kTierHeaderArea, hb, wb);
    if (a_ok) { h = ha; return true; }
    if (b_ok) { h = hb; return true; }
    why = wa + (b_ok ? "" : " (twin: " + wb + ")");
    return false;
}

inline bool ExpertTierSet::write_headers(Dev& d, const TierHdr& h) {
    if (d.base == nullptr || d.size < 2 * kTierHeaderArea + 4096) return false;
    uint8_t buf[kTierHeaderBytes];
    ds4_tier::th_encode(h, buf);
    std::memcpy(d.base, buf, kTierHeaderBytes);
    if (!flush(d)) return false;
    std::memcpy(d.base + kTierHeaderArea, buf, kTierHeaderBytes);
    return flush(d);
}

inline bool ExpertTierSet::decide(uint64_t slot_bytes, uint64_t vram_guard, uint64_t store_guard, uint64_t n_layers,
                                  uint64_t n_expert, const std::string& pack_id, const std::string& profile_sha,
                                  bool force_refill, const std::vector<std::pair<int32_t, int32_t>>& ranked_rest,
                                  bool& reused, std::string& why, std::vector<std::string>& notes) {
    notes.clear();
    reused = false;
    why.clear();
    readable_.store(false, std::memory_order_release);   // (nothing answers until this decision is reused or filled)
    entries_.clear();
    if (devs_.empty()) { why = "expert tier: no device is open"; return false; }
    if (slot_bytes == 0) { why = "expert tier: the store's slot size is 0"; return false; }
    slot_bytes_ = slot_bytes; n_layers_ = n_layers; n_expert_ = n_expert;

    // The tables and the data region are fixed by the shape, the guards (the list's length) and this start's device sizes,
    // before any header is read: a reuse reads exactly where the fill wrote, and a start that configured different sizes
    // (or whose placement could no longer have the same shape) does not match the stored headers and refills (§7.4
    // check 2).  The placement cannot exceed the ranked list, so the tables' room - and with it the data region's start -
    // depends only on what every matching start recomputes the same way.
    const uint64_t poff = 2 * kTierHeaderArea;
    const int eb = strata::core::tier_placement_entry_bytes((int64_t) n_layers, (int64_t) n_expert, (int) devs_.size());
    const uint64_t chunk_off = align_up(poff + (uint64_t) ranked_rest.size() * (uint64_t) eb, 64);
    uint64_t max_cap = 0;
    for (const Dev& d : devs_) max_cap = std::max(max_cap, tier_capacity_bytes(d.spec));
    const uint64_t chunk_count_max = (max_cap + kTierChunk - 1) / kTierChunk;
    data_off_ = align_up(chunk_off + chunk_count_max * 4, 4096);
    std::vector<uint64_t> slots(devs_.size(), 0);
    for (size_t i = 0; i < devs_.size(); ++i) {
        const uint64_t cap = tier_capacity_bytes(devs_[i].spec);
        slots[i] = cap > data_off_ ? (cap - data_off_) / slot_bytes : 0;
    }

    // §7.4 step 1: every device's header (A, else its twin B).  A device with none is a refill, not an error.
    bool ok = !force_refill;
    std::string first_why;
    if (force_refill) first_why = "--expert-tier-refill";
    for (size_t i = 0; i < devs_.size(); ++i) {
        TierHdr h {};
        std::string w;
        if (!read_headers(devs_[i], h, w)) {
            ok = false;
            if (first_why.empty()) first_why = devs_[i].spec.path + ": " + w;
            continue;
        }
        devs_[i].hdr = h;
        const TierHdr& hh = h;
        char pid32[32];
        strata::core::id_to_field(pack_id, pid32);
        const uint64_t plen = hh.n_held * (uint64_t) eb;
        bool bad = false;
        if (hh.device_count != devs_.size()) bad = true, w = "device count " + std::to_string(hh.device_count) + " != this run's " + std::to_string(devs_.size());
        else if (hh.device_index != (uint32_t) i) bad = true, w = "the device was filled as number " + std::to_string(hh.device_index) + " and is number " + std::to_string(i) + " in this run (reordered or swapped devices)";
        else if (hh.chunk_off != chunk_off || (uint64_t) hh.slots32 != slots[i]) bad = true, w = "the tables moved or the device's slots changed: filled " + std::to_string(hh.slots32) + ", this run has " + std::to_string(slots[i]);
        else if (std::memcmp(hh.pack_id, pid32, 32) != 0) bad = true, w = "a different pack (identity §7.1)";
        else if (hh.slot_bytes != slot_bytes) bad = true, w = "slot_bytes " + std::to_string(hh.slot_bytes) + " != this store's " + std::to_string(slot_bytes);
        else if (hh.slot_layout != kDs4SlotLayoutVersion) bad = true, w = "slot layout v" + std::to_string(hh.slot_layout) + " != v" + std::to_string(kDs4SlotLayoutVersion);
        else if (hh.n_layers != n_layers || hh.n_expert != n_expert) bad = true, w = "a different model shape";
        else if (hh.vram_guard != vram_guard || hh.store_guard != store_guard) {
            bad = true;
            w = "guards vram " + std::to_string(hh.vram_guard) + " / store " + std::to_string(hh.store_guard) +
                " != this run's " + std::to_string(vram_guard) + " / " + std::to_string(store_guard);
        } else if (hh.fill_state != 2) bad = true, w = "fill_state " + std::to_string(hh.fill_state) + ": a torn or unfinished fill";
        else if (poff + plen > devs_[i].size) bad = true, w = "the placement table does not fit the device";
        else if (hh.placement_crc != strata::core::tier_crc32c(devs_[i].base + poff, (size_t)plen)) bad = true, w = "placement_crc does not match its table";
        else if (hh.chunk_off + hh.chunk_count * 4 > devs_[i].size || hh.chunk_off + hh.chunk_count * 4 > data_off_) bad = true, w = "the chunk table does not fit before the data";
        else if (hh.chunk_count != (slots[i] == 0 ? 0 : (slots[i] * slot_bytes + kTierChunk - 1) / kTierChunk)) bad = true, w = "the chunk table's length does not match this device's slots";
        else if (devs_[i].spec.verify_full) {   // every chunk of this device's written bytes against its table (§6.4/§11)
            const uint64_t used = (uint64_t)hh.slots32 * slot_bytes;
            const uint64_t doff = data_off_;   // (the fixed data region: the same on every device and every matching start)
            for (uint64_t c = 0; c < hh.chunk_count; ++c) {
                const uint64_t a = c * kTierChunk, b = std::min(used, (c + 1) * kTierChunk);
                if (b <= a) { bad = true, w = "chunk " + std::to_string(c) + " is past the slots this device filled"; break; }
                const uint32_t want = ds4_tier::get32(devs_[i].base + hh.chunk_off + c * 4);
                const uint32_t crc_now = strata::core::tier_crc32c(devs_[i].base + doff + a, (size_t)(b - a));
                if (crc_now != want) { bad = true, w = "chunk " + std::to_string(c) + " does not match its crc (verify=full)"; break; }
            }
        }
        if (bad) { ok = false; if (first_why.empty()) first_why = devs_[i].spec.path + ": " + w; }
    }
    reused = ok;
    if (!ok && why.empty()) why = first_why;
    if (reused) {
        // Every device's header matched this start's tables and data region (the checks above): the placement itself
        // is identical on every device (§6.3), so device 0's table is decoded and its entries answer the lookups.
        // The placement is the same on every device (§6.3), so each device's table must say so: the same length and the same
        // bytes as device 0's.  A device carrying another table (a disk from another set, a half-rewritten one) refills.
        const TierHdr& h0 = devs_[0].hdr;
        for (size_t i = 1; i < devs_.size() && reused; ++i) {
            if (devs_[i].hdr.n_held != h0.n_held ||
                std::memcmp(devs_[i].base + poff, devs_[0].base + poff, (size_t)(h0.n_held * (uint64_t) eb)) != 0) {
                reused = false;
                why = "devices " + devs_[0].spec.path + " and " + devs_[i].spec.path + " do not carry the same placement table";
            }
        }
        std::vector<strata::core::TierPlacementEntry> ent;
        std::string tw;
        if (!reused) {
            ok = false;
        } else if (!strata::core::decode_placement_table(devs_[0].base + poff, h0.n_held * (uint64_t) eb, eb, ent, tw)) {
            reused = false;
            why = "placement table: " + tw;
            ok = false;
        } else {
            lookup_.assign((size_t)n_layers * (size_t)n_expert, Held{});
            held_ = 0;
            std::vector<std::vector<char>> taken(devs_.size());   // (a slot of a device holds one expert)
            for (size_t i = 0; i < devs_.size(); ++i) taken[i].assign((size_t) slots[i], 0);
            for (const auto& x : ent) {
                if (x.layer < 0 || x.expert < 0 || (uint64_t)x.layer >= n_layers || (uint64_t)x.expert >= n_expert || x.device < 0 || x.device >= (int)devs_.size()) {
                    reused = false; why = "a placement entry names an out-of-range layer/expert/device"; break;
                }
                const uint64_t k = x.offset / slot_bytes;
                if (x.offset % slot_bytes != 0 || k >= slots[(size_t)x.device] || taken[(size_t)x.device][(size_t)k]) {
                    reused = false; why = "a placement entry's slot is not a free slot of its device"; break;
                }
                taken[(size_t)x.device][(size_t)k] = 1;
                Held& hd = lookup_[(size_t)x.layer * n_expert + (size_t)x.expert];
                if (hd.dev >= 0) { reused = false; why = "an expert is placed twice"; break; }
                hd = Held{ x.device, k };
                ++held_;
            }
            if (reused) {
                readable_.store(true, std::memory_order_release);
                for (size_t i = 0; i < devs_.size(); ++i)
                    notes.push_back("expert tier " + devs_[i].spec.path + ": reused, " + std::to_string(devs_[i].hdr.slots32) +
                                    " slots, fill_state 2, nothing copied (§7.4)");
                return true;
            }
            lookup_.clear();
            held_ = 0;
        }
    }

    // The refill path (§7.4: any failed check refills the whole tier): the shared planner over `ranked_rest`, capacity
    // in whole slots (§6.2: every blob is priced at slot_bytes), on the layout computed above.
    std::vector<strata::core::TierDevice> dv(devs_.size());
    for (size_t i = 0; i < devs_.size(); ++i) {
        dv[i].capacity_bytes = slots[i] * slot_bytes;
        if (slots[i] == 0)
            notes.push_back("expert tier " + devs_[i].spec.path + ": no room even for one slot: dropped from the deal (§6.2)");
    }
    std::vector<uint64_t> blob(n_layers, slot_bytes);   // (E2: one slot is one expert: capacity is counted in whole slots)
    const auto pl = strata::core::plan_expert_tiers(ranked_rest, blob, dv);   // (one entry per ranked expert, aligned with the list: device -1 stays in the file tier)
    entries_.clear();
    for (size_t i = 0; i < pl.size() && i < ranked_rest.size(); ++i)
        if (pl[i].device >= 0)
            entries_.push_back(strata::core::TierPlacementEntry{ ranked_rest[i].first, ranked_rest[i].second, pl[i].device, pl[i].offset });
    held_ = (int64_t)entries_.size();

    // Every device: the new header (fill_state 1: a start that dies mid-fill refills, §9), then the placement table
    // (identical on every device, §6.3), then the header again with placement_crc over that table.
    const std::vector<uint8_t> table = strata::core::encode_placement_table(entries_, eb);
    for (size_t i = 0; i < devs_.size(); ++i) {
        TierHdr h {};
        std::memcpy(h.magic, kDs4TierMagic, 8);
        h.device_index = (uint32_t)i; h.device_count = (uint32_t)devs_.size();
        {
            char f[32];
            strata::core::id_to_field(pack_id, f);
            std::memcpy(h.pack_id, f, 32);
            strata::core::id_to_field(profile_sha, f);
            std::memcpy(h.profile_sha, f, 32);
        }
        h.slot_layout = kDs4SlotLayoutVersion;
        h.fill_state = 1;
        h.slots32 = (uint32_t)slots[i];
        h.slot_bytes = slot_bytes;
        h.n_layers = n_layers; h.n_expert = n_expert;
        h.vram_guard = vram_guard; h.store_guard = store_guard;
        h.n_held = (uint64_t)entries_.size();
        h.chunk_off = chunk_off;
        h.chunk_count = slots[i] == 0 ? 0 : (slots[i] * slot_bytes + kTierChunk - 1) / kTierChunk;
        if (poff + table.size() > devs_[i].size) { why = "device " + devs_[i].spec.path + ": too small to hold the placement table"; return false; }
        if (!write_headers(devs_[i], h)) { why = "device " + devs_[i].spec.path + ": cannot write its header (the write or its flush failed)"; return false; }
        if (!table.empty()) std::memcpy(devs_[i].base + poff, table.data(), table.size());
        if (!flush(devs_[i])) { why = "device " + devs_[i].spec.path + ": the placement table could not be written (flush failed)"; return false; }
        h.placement_crc = table.empty() ? strata::core::tier_crc32c(table.data(), 0) : strata::core::tier_crc32c(table.data(), table.size());
        devs_[i].hdr = h;
        devs_[i].filling = true;
        if (!write_headers(devs_[i], h)) { why = "device " + devs_[i].spec.path + ": header after the placement table failed"; return false; }
    }
    lookup_.assign((size_t)n_layers * (size_t)n_expert, Held{});
    for (const auto& x : entries_)
        lookup_[(size_t)x.layer * n_expert + (size_t)x.expert] = Held{ x.device, x.offset / slot_bytes };
    return true;   // the bytes follow in fill(); with nothing to place fill() still closes the headers at fill_state 2
}

inline bool ExpertTierSet::fill(const std::function<bool(int, int, uint8_t*, int, std::atomic<int>&, std::atomic<bool>&, uint64_t&)>& read,
                                Fill& out, std::string& err) {
    out = Fill{};
    if (devs_.empty()) { err = "expert tier: no device is open"; return false; }
    out.per_dev.assign(devs_.size(), 0);
    // §9: rank order (entries_ is the placement in rank order), every expert's three matrices as the store's read jobs,
    // on the holding device's queue: the devices fill in parallel, each link keeps its own queue busy (§8.6).
    struct Job {
        int layer = 0, e = 0, dev = 0;
        std::atomic<int> left{0};
        std::atomic<bool> bad{false};
    };
    std::vector<std::unique_ptr<Job>> jobs;
    jobs.reserve(entries_.size());
    // The reads that were started hold pointers to the Jobs above and into the mappings: whatever way this function leaves,
    // they are waited for first (a read in flight must not outlive the counters it counts down).
    auto wait_all = [&] {
        for (;;) {
            bool busy = false;
            for (const auto& j : jobs)
                if (j->left.load(std::memory_order_acquire) > 0) { busy = true; break; }
            if (!busy) return;
            for (auto& d : devs_) if (d.io) d.io->wait_any(5000);
        }
    };
    for (const auto& x : entries_) {
        auto j = std::make_unique<Job>();
        j->layer = x.layer; j->e = x.expert; j->dev = x.device;
        uint8_t* base = slot_base(x.device, x.offset / slot_bytes_);
        if (base == nullptr) { wait_all(); for (auto& d : devs_) d.mark->store(true); err = "expert tier: the device of a placed expert is not open"; return false; }
        j->left.store(3, std::memory_order_release);
        Job* raw = j.get();
        jobs.push_back(std::move(j));   // (kept before the read starts: the wait covers it even when the start fails half way)
        if (!read(raw->layer, raw->e, base, x.device, raw->left, raw->bad, out.bytes)) {
            raw->left.store(0, std::memory_order_release);   // (nothing of this one was queued)
            wait_all();
            for (auto& d : devs_) d.mark->store(true);   // (the fill is abandoned half done: the next start writes it again)
            err = "expert tier: a fill read could not start";
            return false;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    wait_all();   // until every read is done (the same wait shape as ExpertStore::preload's tail)
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    bool failed = false;
    for (const auto& j : jobs) {
        if (j->bad.load(std::memory_order_acquire)) { failed = true; break; }
        ++out.placed;
        ++out.per_dev[(size_t) j->dev];
    }
    if (failed && err.empty()) err = "a read of the model's files into the tier failed";
    // §9: every device flushes (msync) its data, writes its chunk CRC table over the slots it filled and flushes that, and
    // only when EVERY device's bytes are down does any header say fill_state = 2 (A then B).  A copy or a flush that failed
    // marks every device: bytes half-placed are worse than none (D7), and a header never claims bytes that may not be there.
    for (size_t i = 0; i < devs_.size() && !failed; ++i) {
        TierHdr& h = devs_[i].hdr;
        if (!flush(devs_[i])) { failed = true; err = "device " + devs_[i].spec.path + ": the experts could not be written (flush failed)"; break; }
        const uint64_t used = (uint64_t)h.slots32 * slot_bytes_;
        if (used > 0) {
            if (h.chunk_off + h.chunk_count * 4 > devs_[i].size) { failed = true; err = "device " + devs_[i].spec.path + ": no room for its chunk table"; break; }
            for (uint64_t c = 0; c < h.chunk_count; ++c) {
                const uint64_t a = c * kTierChunk, b = std::min(used, (c + 1) * kTierChunk);
                const uint32_t v = strata::core::tier_crc32c(devs_[i].base + data_off_ + a, (size_t)(b - a));
                ds4_tier::put32(devs_[i].base + h.chunk_off + c * 4, v);
            }
            if (!flush(devs_[i])) { failed = true; err = "device " + devs_[i].spec.path + ": the chunk table could not be written (flush failed)"; break; }
        }
    }
    for (size_t i = 0; i < devs_.size() && !failed; ++i) {
        TierHdr& h = devs_[i].hdr;
        h.fill_state = 2;
        if (!write_headers(devs_[i], h)) { failed = true; err = "device " + devs_[i].spec.path + ": the header could not be written (the write or its flush failed)"; break; }
    }
    if (failed) {
        for (auto& d : devs_) d.mark->store(true);
        return false;
    }
    readable_.store(true, std::memory_order_release);   // the lookup answers now: the store's misses become tier copies (§8.1)
    return true;
}

inline void ExpertTierSet::note_refill(int dev) {
    if (dev >= 0 && dev < (int)devs_.size()) devs_[(size_t)dev].mark->store(true, std::memory_order_release);
}

inline void ExpertTierSet::drain() {
    for (auto& d : devs_) d.io.reset();   // (~IoQueue runs the queued jobs to the end, then joins: nothing is dropped, nothing is left running)
}

inline void ExpertTierSet::close() {
    drain();   // (before any mapping goes: a queued copy reads the mapping)
    for (auto& d : devs_) close_one(d);
    devs_.clear();
    lookup_.clear();
    entries_.clear();
    readable_.store(false, std::memory_order_release);
    held_ = 0;
}

}  // namespace strata::ds4
