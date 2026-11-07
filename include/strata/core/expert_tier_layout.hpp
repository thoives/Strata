// include/strata/core/expert_tier_layout.hpp - expert tier plan v1 §5.2: the device's on-disk shape.
//
// No I/O here either: the structs, the byte codecs and the checksums.  `ExpertTier` writes these bytes; the
// layout test corrupts copies of them.  Everything the header must never change without a version bump lives
// in this file, so §12.2 can pin it with `static_assert`s.
//
//   offset 0        header A  (the fixed fields below, in this order, little-endian; the area is 64 KiB)
//   offset 64 KiB   header B  (a copy of A, written after A; read when A's `header_crc` does not match)
//   ...             placement table  (tier-held experts in rank order: layer, expert, device, offset)
//   ...             chunk checksum table  (one CRC32C per 4 MiB of THIS device's data)
//   ...             data: the blobs, each at its recorded offset, aligned to 4 KiB
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

constexpr const char kTierMagic[8] = {'S', 'T', 'R', 'P', 'T', 'I', 'E', 'R'};
constexpr uint32_t kTierLayoutVersion = 2;          // header §5.2: started at 1; 2: header_crc covers fill_state too
                                                    // (a version 1 header is not read: such a tier refills once)
constexpr uint64_t kTierHeaderArea = 64ull << 10;   // the fixed fields' area, A then B
constexpr uint64_t kTierChunk = 4ull << 20;         // the chunk checksum table's unit
constexpr uint64_t kTierAlign = 4ull << 10;         // blob alignment

/// The header's fixed fields, in §5.2's order.  On disk they are 160 little-endian bytes (`kTierHeaderBytes`);
/// this struct is the in-memory form.  `pack_id` and `profile_id` are 32 bytes of identity (§6.2): a SHA-256
/// when the engine has one, else the pack's size and modification time spelled out.
struct TierHeader {
    char magic[8];
    uint32_t version = kTierLayoutVersion;
    uint32_t device_index = 0;
    uint32_t device_count = 0;
    char pack_id[32] = {};
    char profile_id[32] = {};
    uint32_t reserved0 = 0;
    uint64_t n_layers = 0;
    uint64_t n_expert = 0;
    uint64_t capacity_bytes = 0;    // this device's capacity at fill time
    uint64_t placement_off = 0;
    uint64_t placement_len = 0;
    uint32_t placement_crc = 0;
    uint32_t reserved1 = 0;
    uint64_t chunk_off = 0;
    uint64_t chunk_count = 0;
    uint32_t fill_state = 0;        // 0 empty, 1 filling, 2 complete (§6.4, written last)
    uint32_t header_crc = 0;        // CRC32C of the 156 bytes before it, fill_state included (§6.4)
};
static_assert(sizeof(TierHeader) == 160, "the header's fixed fields: §5.2's sizes, padding pinned as reserved");

constexpr uint64_t kTierHeaderBytes = 160;
constexpr uint64_t kTierHeaderCrcBytes = kTierHeaderBytes - 4;   // header_crc covers every byte before it, fill_state too
constexpr uint64_t kTierFillStateOffset = 152;                   // where fill_state sits in the encoded header

// ---- CRC32C (Castagnoli: the reflected 0x82F63B78), §6.4's chunk and header checksums.
constexpr uint32_t kCrc32cPoly = 0x82F63B78u;
constexpr std::array<uint32_t, 256> make_crc32c_table() {
    std::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t r = i;
        for (int k = 0; k < 8; ++k) r = (r & 1) ? (r >> 1) ^ kCrc32cPoly : r >> 1;
        t[i] = r;
    }
    return t;
}
// Streaming form: feed bytes in order (the fill's chunk CRCs are computed as the blobs are written, §6.4/§6.5),
// then `tier_crc32c_finish` closes it.  `tier_crc32c` is this pair with empty state.
constexpr uint32_t kCrc32cInit = 0xFFFFFFFFu;
inline uint32_t tier_crc32c_feed(uint32_t r, const void* data, size_t n) {
    constexpr auto table = make_crc32c_table();
    const uint8_t* p = (const uint8_t*) data;
    for (size_t i = 0; i < n; ++i) r = table[(r ^ p[i]) & 0xFF] ^ (r >> 8);
    return r;
}
inline uint32_t tier_crc32c_finish(uint32_t r) { return r ^ 0xFFFFFFFFu; }
inline uint32_t tier_crc32c(const void* data, size_t n) { return tier_crc32c_finish(tier_crc32c_feed(kCrc32cInit, data, n)); }

// ---- header codec.  `encode_header(h, buf)` fills the 160 bytes (with `header_crc` computed over the first
// 156, so a flipped fill_state is a crc mismatch, not a "complete" tier); `decode_header(buf, h, why)` refuses a bad
// magic, version or crc, in that order.  The disk bytes are the header's identity: the same fields always give the
// same bytes.
void encode_header(const TierHeader& h, uint8_t* buf);
bool decode_header(const uint8_t* buf, TierHeader& h, std::string& why);   // checks magic, version and header_crc
// §6.4: a header whose `header_crc` does not match is read from its twin (B if A fails).  Pure on two
// `kTierHeaderBytes` buffers; neither matching means the device is refilled.
bool read_header_pair(const uint8_t* a, const uint8_t* b, TierHeader& h, int& which, std::string& why);   // which: 0 A, 1 B, -1 both bad
void header_clear_crc(TierHeader& h);   // zeroes header_crc for the CRC pass

// id fields (32 bytes on disk): a SHA is stored as its 32 raw bytes (the string is "sha:<64 hex>"), anything else
// as a short string in its first bytes (§6.2's size+mtime fallback).  Round trip:
void id_to_field(const std::string& id, char field[32]);
std::string id_from_field(const char field[32]);

// ---- placement table (§6.1: identical on every device).  Entries are the tier-held experts in rank order.
// Small form (9 bytes/entry, what §5.2's 221 KiB Flash-Next figure prices): layer u16, expert u16, device i8,
// offset in 4 KiB units u32 — exact because blobs align to 4 KiB, so an offset needs no low 12 bits.  When the
// model shape or the device count does not fit the small form, the large form (layer i32, expert i32, device
// i32, offset u64; 20 bytes/entry) is used; both sides derive the form from `n_layers`, `n_expert` and
// `device_count`, which the header carries.
struct TierPlacementEntry {
    int32_t layer = 0, expert = 0, device = 0;
    uint64_t offset = 0;    // byte offset into the device's data region
};
constexpr int tier_placement_entry_bytes(int64_t n_layers, int64_t n_expert, int device_count) {
    return (n_layers <= 0xFFFF && n_expert <= 0xFFFF && device_count <= 127) ? 9 : 20;
}
std::vector<uint8_t> encode_placement_table(const std::vector<TierPlacementEntry>& entries, int entry_bytes);
bool decode_placement_table(const uint8_t* bytes, uint64_t len, int entry_bytes,
                            std::vector<TierPlacementEntry>& entries, std::string& why);

// ---- chunk checksum table: one CRC32C per 4 MiB of this device's data, big-endian-free LE u32.
std::vector<uint8_t> encode_chunk_table(const std::vector<uint32_t>& crcs);
std::vector<uint32_t> decode_chunk_table(const uint8_t* bytes, uint64_t count);

// ---- what a placement is valid for (§6.2).  Everything here is compared on every start; any change refills
// the whole tier (§6.3, decision taken #2).
struct TierIdentity {
    std::string pack_id;       // "sha:<64 hex>" when setup verified the pack, else "st:<size>:<mtime>" (§6.2)
    std::string profile_id;    // "sha:<64 hex>": SHA-256 of the profile file's bytes
    int64_t n_layers = 0, n_expert = 0;
    std::vector<uint64_t> layer_blob_bytes;   // checked through the pack's layout (not stored separately, §6.2)
};

// ---- the device before a fill (§5.4): the engines write over a device only when it is blank or already an expert tier.
// Blank: every byte the fill may write (the first `mib` MiB of the device) is zero, as a new file, a fallocate'd file, a
// hole-punched file or a zeroed disk are. The whole range, not only the first 128 KiB: a sparse file can hold someone's
// data past its prefix. A tier: header A or B starts with either engine's magic (STRPTIER here, STRPTDS4 for strata-dsv41),
// whatever its state. Anything else is someone's data: the engine stops and says how to confirm (strata launch / setup ask,
// or --expert-tier-overwrite clears the first `mib` MiB after the person said yes, so the engine then finds it blank).
// The reads over a device are in expert_tier_dev.hpp (tier_fd_blank_or_tier, tier_handle_blank_or_tier); these are the
// pieces they share.
constexpr uint64_t kTierClaimBytes = 128ull << 10;   // the two header areas (A at 0, B at 64 KiB)
/// True when the first `n` bytes of a device (`head`) hold either engine's tier magic at 0 or at 64 KiB.
inline bool tier_header_magic(const uint8_t* head, uint64_t n) {
    static const char* const magics[2] = {"STRPTIER", "STRPTDS4"};
    for (uint64_t off : {(uint64_t) 0, (uint64_t) (64ull << 10)})
        for (const char* m : magics)
            if (off + 8 <= n) {
                bool same = true;
                for (int i = 0; i < 8 && same; ++i) same = head[off + (uint64_t) i] == (uint8_t) m[i];
                if (same) return true;
            }
    return false;
}
/// Blank or tier over an in-memory view of the device's first `size` bytes (every byte of it is checked, the way the fill may
/// write it). The engines check the device itself (tier_fd_blank_or_tier); this is the same rule for a buffer.
inline bool tier_blank_or_tier(const uint8_t* base, uint64_t size) {
    if (tier_header_magic(base, size)) return true;
    for (uint64_t i = 0; i < size; ++i)
        if (base[i] != 0) return false;
    return true;
}
inline std::string tier_foreign_data_message(const std::string& path) {
    return "expert tier " + path + ": the device holds data that is not a Strata expert tier (the part the tier may use is not empty), "
           "and filling the tier would overwrite it, so nothing was written. If that data may go, confirm it: strata launch or "
           "setup ask (or take --expert-tier-overwrite), or clear the device's first mib MiB yourself; otherwise give an empty "
           "file or device";
}

// SHA-256 for the ids (implemented in src/core/expert_tier.cpp; the profile is small, so this is cheap per start).
std::string tier_sha256_hex(const uint8_t* data, size_t n);
std::string tier_sha256_file(const std::string& path);   // empty on failure

}  // namespace strata::core
