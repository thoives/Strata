// tests/core/expert_tier_layout_test.cpp - §12.2: the on-disk shapes, pinned and corrupted on purpose.
// No device and no file: the codecs are pure, so the round trip and the torn-header rules run anywhere.
#include "strata/core/expert_tier_layout.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::core;

static void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}

// §5.2: header area 64 KiB; fixed fields 160 bytes with §5.2's sizes.
static_assert(kTierHeaderBytes == 160);
static_assert(kTierHeaderArea == 65536);
static_assert(sizeof(TierHeader) == 160);
static_assert(tier_placement_entry_bytes(48, 512, 3) == 9, "Flash-Next shape uses the 9-byte entry (§5.2's 221 KiB)");
static_assert(tier_placement_entry_bytes(48, 512, 128) == 20, "too many devices for the i8 form: the 20-byte form");

static void header_round_trip() {
    TierHeader h{};
    std::memcpy(h.magic, kTierMagic, 8);
    h.device_index = 1;
    h.device_count = 3;
    id_to_field("sha:ba7816bf8f01cfea4144050c93e7fb65fcfc4b11c27d6459b38b71d5198f2cc8", h.pack_id);
    id_to_field("st:34123456789:1730430000", h.profile_id);
    h.n_layers = 48;
    h.n_expert = 512;
    h.capacity_bytes = 34ull << 30;
    h.placement_off = 2 * kTierHeaderArea;
    h.placement_len = 1024;
    h.placement_crc = 0x12345678;
    h.chunk_off = h.placement_off + h.placement_len;
    h.chunk_count = 97;
    h.fill_state = 2;

    uint8_t a[kTierHeaderBytes], b[kTierHeaderBytes];
    encode_header(h, a);
    TierHeader back{};
    std::string why;
    require(decode_header(a, back, why), "round trip: " + why);
    require(std::memcmp(back.pack_id, h.pack_id, 32) == 0, "pack_id field lost");
    require(id_from_field(back.profile_id) == "st:34123456789:1730430000", "short-string id does not round trip");
    require(id_from_field(back.pack_id) == "sha:ba7816bf8f01cfea4144050c93e7fb65fcfc4b11c27d6459b38b71d5", "sha id: 28 raw bytes expected");
    require(back.capacity_bytes == h.capacity_bytes && back.fill_state == 2 && back.chunk_count == 97, "fields lost");

    // the same fields give the same bytes (the header is part of the identity)
    uint8_t a2[kTierHeaderBytes];
    encode_header(h, a2);
    require(std::memcmp(a, a2, kTierHeaderBytes) == 0, "encode is not deterministic");

    // §12.2: header A corrupted → header B is read.
    encode_header(h, a);
    encode_header(h, b);
    a[40] ^= 0xFF;   // inside the fields A covers
    TierHeader chosen{};
    int which = -1;
    std::string w;
    require(read_header_pair(a, b, chosen, which, w), "A and B both refused");
    require(which == 1, "the corrupted A was not skipped for its twin");

    // both corrupted → refill
    b[41] ^= 0xFF;
    TierHeader dead{};
    require(!read_header_pair(a, b, dead, which, w), "two dead headers read as alive");
    require(which == -1, "both-fail must report -1");
}

static void placement_table_codec() {
    std::vector<TierPlacementEntry> e = {{0, 511, 0, 0}, {47, 0, 2, 4096}, {47, 511, 1, 1 << 22}};
    const std::vector<uint8_t> bytes = encode_placement_table(e, 9);
    require(bytes.size() == 9 * e.size(), "9-byte entries (§5.2)");
    std::vector<TierPlacementEntry> back;
    std::string why;
    require(decode_placement_table(bytes.data(), bytes.size(), 9, back, why), why);
    require(back.size() == e.size() && back[0].layer == 0 && back[0].expert == 511 && back[0].device == 0 && back[1].device == 2 && back[2].offset == (1u << 22), "entries lost");

    // a placement table whose CRC does not match: refill (§12.2). The CRC check lives with the header, but the
    // table bytes are what it covers, so flip one byte and check the CRC moves.
    const uint32_t good = tier_crc32c(bytes.data(), bytes.size());
    std::vector<uint8_t> torn = bytes;
    torn[3] ^= 1;
    require(tier_crc32c(torn.data(), torn.size()) != good, "a flipped placement byte keeps the table CRC: no refill would fire");

    // the 20-byte form keeps an 8 GiB offset exactly
    const std::vector<uint8_t> big = encode_placement_table(e, 20);
    require(big.size() == 20 * e.size(), "20-byte entries");
    require(decode_placement_table(big.data(), big.size(), 20, back, why) && back[2].offset == (1u << 22), "big form lost the offset");
}

static void chunk_table_codec() {
    const std::vector<uint32_t> crcs = {0, 1, 0xFFFFFFFFu, 0x01020304u};
    const std::vector<uint8_t> b = encode_chunk_table(crcs);
    const std::vector<uint32_t> back = decode_chunk_table(b.data(), crcs.size());
    require(back == crcs, "chunk table round trip");
}

// The CRC32C forms must agree byte for byte: the table loop is the reference; the hardware form (when this CPU has
// SSE4.2) is checked against it for every length and alignment that matters, and for streamed feeds split at random
// points.  The dispatched form (tier_crc32c) is checked too; it is the hardware form only under STRATA_TIER_HW_CRC=1.
static void crc32c_forms_agree() {
    const std::string check = "123456789";
    require(tier_crc32c(check.data(), check.size()) == 0xE3069283u, "CRC32C check value (dispatch)");
    require(tier_crc32c_finish(tier_crc32c_feed_portable(kCrc32cInit, check.data(), check.size())) == 0xE3069283u,
            "CRC32C check value (table)");
    std::mt19937 rng(20260122u);
    std::vector<uint8_t> buf((1u << 20) + 16);
    for (auto& b : buf) b = (uint8_t) rng();
    const bool hw = tier_crc32c_hw_available();
    if (!hw) std::cout << "expert_tier_layout_test: no SSE4.2 on this CPU: the hardware form is not checked\n";
    const size_t lengths[] = {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 63, 64, 65, 255, 4096, 4097, 65536, (1u << 20) - 3, 1u << 20};
    for (size_t n : lengths)
        for (size_t off : {size_t(0), size_t(1), size_t(3), size_t(7)}) {
            const uint8_t* p = buf.data() + off;
            const uint32_t want = tier_crc32c_finish(tier_crc32c_feed_portable(kCrc32cInit, p, n));
            require(tier_crc32c(p, n) == want, "the dispatched CRC32C differs from the table");
            if (hw)
                require(tier_crc32c_finish(tier_crc32c_feed_hw(kCrc32cInit, p, n)) == want, "the hardware CRC32C differs from the table");
            // streamed: the same bytes fed in random pieces, alternating the forms, give the same result
            uint32_t r = kCrc32cInit;
            size_t left = n, at = 0;
            bool use_hw = hw;
            while (left > 0) {
                const size_t take = std::min<size_t>(left, 1 + (size_t) (rng() % 300));
                r = (use_hw ? tier_crc32c_feed_hw(r, p + at, take) : tier_crc32c_feed_portable(r, p + at, take));
                if (hw) use_hw = !use_hw;
                at += take;
                left -= take;
            }
            require(tier_crc32c_finish(r) == want, "a streamed CRC32C (mixed forms) differs from the one-shot table value");
        }
}


// The striped layout (docs/EXPERT_TIER.md): its header version is its own, the reader takes either layout's version (the engine
// checks which one it expects), and the striped table round-trips and refuses the shapes that cannot be a table of this start.
static void striped_version_and_table() {
    TierHeader h{};
    std::memcpy(h.magic, kTierMagic, 8);
    h.version = kTierLayoutVersionStriped;
    h.reserved0 = 65536;   // the stripe unit
    h.fill_state = 2;
    uint8_t buf[kTierHeaderBytes];
    encode_header(h, buf);
    TierHeader back{};
    std::string why;
    require(decode_header(buf, back, why), "a striped header did not decode: " + why);
    require(back.version == kTierLayoutVersionStriped && back.reserved0 == 65536, "striped header fields did not survive");
    require(kTierLayoutVersionStriped != kTierLayoutVersion, "the striped layout must have its own version");

    h.version = 4;   // a layout this engine does not know is refused, and the refusal names the versions it reads
    encode_header(h, buf);
    require(!decode_header(buf, back, why), "a version 4 header was read");
    require(why.find("2 and 3") != std::string::npos, "the refusal does not name the versions it reads: " + why);

    std::vector<StripedTableEntry> es(2);
    es[0] = {3, 7, 1, {4096, 8192}};
    es[1] = {4, 0, 0, {0, 12288}};
    const std::vector<uint8_t> bytes = encode_striped_table(es, 2);
    require(striped_entry_bytes(2) == 28 && bytes.size() == 2 * 28, "a striped entry is 12 + 8 bytes per device");
    std::vector<StripedTableEntry> got;
    require(decode_striped_table(bytes.data(), bytes.size(), 2, got, why), "the striped table did not decode: " + why);
    require(got.size() == 2 && got[0].layer == 3 && got[0].expert == 7 && got[0].start == 1, "striped table entry fields");
    require(got[0].offset == es[0].offset && got[1].offset == es[1].offset, "striped table offsets");
    require(decode_striped_table(bytes.data(), 0, 2, got, why) && got.empty(), "an empty striped table is a table of no experts");

    require(!decode_striped_table(bytes.data(), bytes.size() - 1, 2, got, why), "a table that is not whole entries was read");
    std::vector<uint8_t> bad = bytes;
    bad[8] = 2;                                   // the first entry's start device is 2, and there are 2 devices
    require(!decode_striped_table(bad.data(), bad.size(), 2, got, why), "a start device outside the set was read");
    bad = bytes;
    bad[12] = 1;                                  // the first device's offset 4097: not 4 KiB aligned
    require(!decode_striped_table(bad.data(), bad.size(), 2, got, why), "an unaligned piece offset was read");
}

int main() {
    try {
        header_round_trip();
        placement_table_codec();
        chunk_table_codec();
        crc32c_forms_agree();
        striped_version_and_table();
        std::cout << "expert_tier_layout_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "expert_tier_layout_test: " << error.what() << '\n';
        return 1;
    }
}
