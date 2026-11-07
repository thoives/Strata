// src/core/expert_tier.cpp - expert tier plan v1: §5.2's bytes, §5.3 opening, §6 persistence, §7.3 copy workers.
//
// Backend-neutral except one call: the §5.3 registration.  It is compiled only where the build says so with
// STRATA_TIER_CUDA_REGISTER (CMakeLists.txt sets it on strata_engine, the CUDA and HIP engine, whose runtime target
// carries cuda_runtime.h - the hip_compat shim on HIP).  A header that merely happens to be on the include path no longer
// decides it: the SYCL and Metal builds, the host-only tests and strata-dsv41 make no CUDA runtime call from here, even
// on a PC with the CUDA toolkit installed.  The portable flag means a failure is recoverable and the device keeps
// running as pageable memory, which is what #1250 already does for the RAM copy; an unregistered device logs so (§13).
#include "strata/core/expert_tier.hpp"
#include "strata/core/expert_tier_dev.hpp"
#include "strata/core/expert_tier_layout.hpp"
#include "strata/core/expert_tier_plan.hpp"
#include "strata/core/expert_source.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#if defined(STRATA_TIER_CUDA_REGISTER)
#  include <cuda_runtime.h>
#endif

#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace strata::core {

// ---------------------------------------------------------------- §5.2 byte codecs ------------------------------------
static void put_u32(uint8_t* b, uint32_t v) {
    b[0] = (uint8_t) v; b[1] = (uint8_t) (v >> 8); b[2] = (uint8_t) (v >> 16); b[3] = (uint8_t) (v >> 24);
}
static void put_u64(uint8_t* b, uint64_t v) { put_u32(b, (uint32_t) v); put_u32(b + 4, (uint32_t) (v >> 32)); }
static uint32_t get_u32(const uint8_t* b) {
    return (uint32_t) b[0] | ((uint32_t) b[1] << 8) | ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24);
}
static uint64_t get_u64(const uint8_t* b) { return (uint64_t) get_u32(b) | ((uint64_t) get_u32(b + 4) << 32); }

static_assert(sizeof(TierHeader) == kTierHeaderBytes, "the header struct and its byte form must agree");

static char hex_byte(char hi, char lo) {
    auto v = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
    const int h = v(hi), l = v(lo);
    if (h < 0 || l < 0) return '?';
    return (char) ((h << 4) | l);
}

void id_to_field(const std::string& id, char field[32]) {
    std::memset(field, 0, 32);
    // A SHA-256 is kept as "sha:" plus the raw digest's first 28 bytes: identity comparisons are on these bytes,
    // and 28 bytes of a digest cannot collide (§6.2).  Anything else (the size+mtime fallback) is a plain short string.
    if (id.size() >= 68 && id.compare(0, 4, "sha:") == 0) {
        std::memcpy(field, "sha:", 4);
        for (size_t i = 0; i < 28; ++i)
            field[4 + i] = hex_byte(id[4 + 2 * i], id[5 + 2 * i]);
        return;
    }
    const size_t n = id.size() < 31 ? id.size() : 31;
    std::memcpy(field, id.data(), n);
}
std::string id_from_field(const char field[32]) {
    if (std::memcmp(field, "sha:", 4) == 0) {
        std::string out = "sha:";
        static const char* digits = "0123456789abcdef";
        for (int i = 0; i < 28; ++i) {
            out += digits[((uint8_t) field[4 + i]) >> 4];
            out += digits[((uint8_t) field[4 + i]) & 15];
        }
        return out;
    }
    return std::string(field);
}

void encode_header(const TierHeader& h, uint8_t* buf) {
    uint8_t* b = buf;
    std::memcpy(b, h.magic, 8); b += 8;
    put_u32(b, h.version); b += 4;
    put_u32(b, h.device_index); b += 4;
    put_u32(b, h.device_count); b += 4;
    std::memcpy(b, h.pack_id, 32); b += 32;
    std::memcpy(b, h.profile_id, 32); b += 32;
    put_u32(b, h.reserved0); b += 4;
    put_u64(b, h.n_layers); b += 8;
    put_u64(b, h.n_expert); b += 8;
    put_u64(b, h.capacity_bytes); b += 8;
    put_u64(b, h.placement_off); b += 8;
    put_u64(b, h.placement_len); b += 8;
    put_u32(b, h.placement_crc); b += 4;
    put_u32(b, h.reserved1); b += 4;
    put_u64(b, h.chunk_off); b += 8;
    put_u64(b, h.chunk_count); b += 8;
    put_u32(b, h.fill_state); b += 4;
    put_u32(b, 0);
    static_assert(kTierFillStateOffset + 4 == kTierHeaderCrcBytes, "fill_state is the last field the crc covers");
    const uint32_t crc = tier_crc32c(buf, (size_t) kTierHeaderCrcBytes);   // every field before header_crc, fill_state too
    put_u32(buf + kTierHeaderCrcBytes, crc);
}

bool decode_header(const uint8_t* buf, TierHeader& h, std::string& why) {
    if (std::memcmp(buf, kTierMagic, 8) != 0) { why = "header magic is not STRPTIER"; return false; }
    const uint8_t* b = buf + 8;
    const uint32_t version = get_u32(b); b += 4;
    // the version first: a header of another layout says so instead of "crc mismatch" (its crc covers other bytes)
    if (version != kTierLayoutVersion) { why = "layout version " + std::to_string(version) + ", this engine reads " + std::to_string(kTierLayoutVersion); return false; }
    if (tier_crc32c(buf, (size_t) kTierHeaderCrcBytes) != get_u32(buf + kTierHeaderCrcBytes)) {
        why = "header_crc does not match its fields";
        return false;
    }
    std::memcpy(h.magic, buf, 8);   // keep the magic so a decoded header can be re-encoded unchanged
    h.version = version;
    h.device_index = get_u32(b); b += 4;
    h.device_count = get_u32(b); b += 4;
    std::memcpy(h.pack_id, b, 32); b += 32;
    std::memcpy(h.profile_id, b, 32); b += 32;
    h.reserved0 = get_u32(b); b += 4;
    h.n_layers = get_u64(b); b += 8;
    h.n_expert = get_u64(b); b += 8;
    h.capacity_bytes = get_u64(b); b += 8;
    h.placement_off = get_u64(b); b += 8;
    h.placement_len = get_u64(b); b += 8;
    h.placement_crc = get_u32(b); b += 4;
    h.reserved1 = get_u32(b); b += 4;
    h.chunk_off = get_u64(b); b += 8;
    h.chunk_count = get_u64(b); b += 8;
    h.fill_state = get_u32(b); b += 4;
    h.header_crc = get_u32(b);
    return true;
}

bool read_header_pair(const uint8_t* a, const uint8_t* b, TierHeader& h, int& which, std::string& why) {
    std::string why_a;
    if (decode_header(a, h, why_a)) { which = 0; return true; }
    std::string why_b;
    if (decode_header(b, h, why_b)) { which = 1; return true; }
    why = why_a + " (and twin: " + why_b + ")";
    which = -1;
    return false;
}

static void put_i32(uint8_t* b, int32_t v) { put_u32(b, (uint32_t) v); }
static int32_t get_i32(const uint8_t* b) { return (int32_t) get_u32(b); }

std::vector<uint8_t> encode_placement_table(const std::vector<TierPlacementEntry>& entries, int entry_bytes) {
    std::vector<uint8_t> out((size_t) entries.size() * (size_t) entry_bytes);
    for (size_t i = 0; i < entries.size(); ++i) {
        uint8_t* b = out.data() + i * (size_t) entry_bytes;
        const TierPlacementEntry& e = entries[i];
        if (entry_bytes == 9) {
            put_u32(b, ((uint32_t) e.layer << 16) | ((uint32_t) e.expert & 0xFFFF)); b += 4;
            b[0] = (uint8_t) e.device; b += 1;
            put_u32(b, (uint32_t) (e.offset / kTierAlign)); b += 4;   // exact: offsets are 4 KiB aligned
        } else {
            put_i32(b, e.layer); b += 4;
            put_i32(b, e.expert); b += 4;
            put_i32(b, e.device); b += 4;
            put_u64(b, e.offset); b += 8;
        }
    }
    return out;
}

bool decode_placement_table(const uint8_t* bytes, uint64_t len, int entry_bytes,
                            std::vector<TierPlacementEntry>& entries, std::string& why) {
    if (entry_bytes != 9 && entry_bytes != 20) { why = "internal: bad placement entry size"; return false; }
    if (len % (uint64_t) entry_bytes != 0) { why = "placement table length is not a whole number of entries"; return false; }
    entries.clear();
    entries.reserve((size_t) (len / (uint64_t) entry_bytes));
    for (uint64_t i = 0; i < len; i += (uint64_t) entry_bytes) {
        const uint8_t* b = bytes + i;
        TierPlacementEntry e{};
        if (entry_bytes == 9) {
            const uint32_t le = get_u32(b);
            e.layer = (int32_t) (le >> 16); e.expert = (int32_t) (le & 0xFFFF);
            e.device = (int8_t) b[4];
            e.offset = (uint64_t) get_u32(b + 5) * kTierAlign;
        } else {
            e.layer = get_i32(b); e.expert = get_i32(b + 4); e.device = get_i32(b + 8); e.offset = get_u64(b + 12);
        }
        if (e.device < 0) { why = "placement entry names device " + std::to_string(e.device); return false; }
        if (e.offset % kTierAlign != 0) { why = "placement entry offset is not 4 KiB aligned"; return false; }
        entries.push_back(e);
    }
    return true;
}

std::vector<uint8_t> encode_chunk_table(const std::vector<uint32_t>& crcs) {
    std::vector<uint8_t> out(crcs.size() * 4);
    for (size_t i = 0; i < crcs.size(); ++i) put_u32(out.data() + i * 4, crcs[i]);
    return out;
}
std::vector<uint32_t> decode_chunk_table(const uint8_t* bytes, uint64_t count) {
    std::vector<uint32_t> out((size_t) count);
    for (uint64_t i = 0; i < count; ++i) out[(size_t) i] = get_u32(bytes + i * 4);
    return out;
}

// ---------------------------------------------------------------- SHA-256 for the ids (§6.2) --------------------
// The constants (IV = frac(sqrt p), K = frac(cbrt p)) are computed, not typed: a typo in a hand-copied table is
// the classic SHA-256 bug, and the profile is small so the setup cost is free. Verified against sha256sum in
// tests/core/expert_tier_test.cpp.
static void sha256_round(uint32_t h[8], const uint32_t* w, const uint32_t* k) {
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ ((uint32_t) ~e & g);
        const uint32_t temp1 = hh + S1 + ch + k[i] + w[i];
        const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = S0 + maj;
        hh = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
    }
    for (int i = 0; i < 8; ++i) h[i] += (i == 0 ? a : i == 1 ? b : i == 2 ? c : i == 3 ? d : i == 4 ? e : i == 5 ? f : i == 6 ? g : hh);
}

struct Sha256Consts { std::vector<uint32_t> iv, k; };
static Sha256Consts sha256_constants() {
    // primes 2..19 for IV, 2..311 for K; long double keeps the fractional parts exact far past bit 32.
    std::vector<int> primes;
    for (int n = 2; (int) primes.size() < 64; ++n) {
        bool p = true;
        for (int d = 2; d * d <= n; ++d) if (n % d == 0) { p = false; break; }
        if (p) primes.push_back(n);
    }
    Sha256Consts out;
    out.iv.resize(8);
    out.k.resize(64);
    for (int i = 0; i < 8; ++i) {
        const long double s = sqrtl((long double) primes[(size_t) i]) - (long double) (int64_t) sqrtl((long double) primes[(size_t) i]);
        out.iv[(size_t) i] = (uint32_t) (s * 4294967296.0L);
    }
    for (int i = 0; i < 64; ++i) {
        const long double c = cbrtl((long double) primes[(size_t) i]);
        out.k[(size_t) i] = (uint32_t) ((c - (long double) (int64_t) c) * 4294967296.0L);
    }
    return out;
}

std::string tier_sha256_hex(const uint8_t* data, size_t n) {
    const Sha256Consts cs = sha256_constants();
    uint32_t h[8];
    for (int i = 0; i < 8; ++i) h[i] = cs.iv[(size_t) i];
    // the padded message: data, 0x80, zeros, big-endian 64-bit length
    std::vector<uint8_t> buf(data, data + n);
    buf.push_back(0x80);
    while (buf.size() % 64 != 56) buf.push_back(0);
    for (int i = 7; i >= 0; --i) buf.push_back((uint8_t) ((uint64_t) n * 8 >> (8 * i)));
    for (size_t blk = 0; blk < buf.size(); blk += 64) {
        uint32_t w[64];
        for (int t = 0; t < 16; ++t)
            w[t] = ((uint32_t) buf[blk + 4 * t] << 24) | ((uint32_t) buf[blk + 4 * t + 1] << 16) |
                   ((uint32_t) buf[blk + 4 * t + 2] << 8) | (uint32_t) buf[blk + 4 * t + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t x = w[i - 15], y = w[i - 2];
            const uint32_t s0 = (x >> 7 | x << 25) ^ (x >> 18 | x << 14) ^ (x >> 3);
            const uint32_t s1 = (y >> 17 | y << 15) ^ (y >> 19 | y << 13) ^ (y >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        sha256_round(h, w, cs.k.data());
    }
    static const char* digits = "0123456789abcdef";
    std::string out;
    for (int i = 0; i < 8; ++i) for (int s = 28; s >= 0; s -= 4) out += digits[(h[i] >> s) & 15];
    return out;
}

std::string tier_sha256_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return "";
    std::vector<uint8_t> buf;
    uint8_t tmp[65536];
    size_t got = 0;
    while ((got = std::fread(tmp, 1, sizeof tmp, f)) > 0) buf.insert(buf.end(), tmp, tmp + got);
    std::fclose(f);
    return tier_sha256_hex(buf.data(), buf.size());
}

std::vector<std::string> tier_pack_files(const std::string& pack) {
    namespace fs = std::filesystem;
    std::error_code ec;
    std::vector<std::string> out;
    const fs::path root(pack);
    if (fs::is_regular_file(root, ec)) { out.push_back(root.string()); return out; }
    for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end; it.increment(ec))
        if (it->is_regular_file(ec)) out.push_back(it->path().string());
    return out;
}

bool tier_pack_identity(const std::string& pack, const std::vector<std::string>& exclude, std::string& id, std::string& err,
                        const std::vector<std::string>& extra) {
    namespace fs = std::filesystem;
    std::error_code ec;
    std::vector<fs::path> skip;
    for (const std::string& x : exclude)
        if (!x.empty()) skip.push_back(fs::weakly_canonical(fs::path(x), ec));
    std::vector<fs::path> files;
    const fs::path root(pack);
    if (fs::is_regular_file(root, ec)) files.push_back(root);
    else {
        for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end; it.increment(ec))
            if (it->is_regular_file(ec)) files.push_back(it->path());
    }
    if (ec) { err = "cannot list " + pack + ": " + ec.message(); return false; }
    std::vector<std::pair<std::string, fs::path>> keyed;
    for (const fs::path& f : files) {
        const fs::path canon = fs::weakly_canonical(f, ec);
        bool skipped = false;
        for (const fs::path& s : skip) if (canon == s) { skipped = true; break; }
        if (skipped) continue;
        const std::string rel = fs::is_regular_file(root, ec) ? f.filename().generic_string() : f.lexically_relative(root).generic_string();
        keyed.emplace_back(rel, f);
    }
    std::sort(keyed.begin(), keyed.end());
    // the model files read from outside the pack (a GGUF in place), after the pack's own, in the order given: by file name, so
    // a pack moved with its GGUF keeps its identity, and a replaced shard (another size or time) does not
    for (const std::string& x : extra) {
        if (x.empty()) continue;
        keyed.emplace_back("+" + fs::path(x).filename().generic_string(), fs::path(x));
    }
    std::string list;
    for (const auto& kf : keyed) {
        const auto sz = fs::file_size(kf.second, ec);
        if (ec) { err = "cannot read the size of " + kf.second.string() + ": " + ec.message(); return false; }
        const auto mt = fs::last_write_time(kf.second, ec);
        if (ec) { err = "cannot read the time of " + kf.second.string() + ": " + ec.message(); return false; }
        list += kf.first + '\t' + std::to_string((unsigned long long) sz) + '\t' + std::to_string((long long) mt.time_since_epoch().count()) + '\n';
    }
    if (keyed.empty()) { err = "no model files under " + pack; return false; }
    id = "sha:" + tier_sha256_hex((const uint8_t*) list.data(), list.size());
    return true;
}

// ---------------------------------------------------------------- §5.3: one device ------------------------------
static bool tier_flush(ExpertTier& t);

static bool tier_write_fill_state(ExpertTier& t, uint32_t state) {
    // flip fill_state on both headers and keep each header's own CRC true: the header areas' bytes, in place.
    uint8_t* base = (uint8_t*) t.base();
    if (base == nullptr) return false;
    bool ok = true;
    for (int which = 0; which < 2; ++which) {
        uint8_t* h = base + (size_t) which * (size_t) kTierHeaderArea;
        if (std::memcmp(h, kTierMagic, 8) != 0) { ok = false; continue; }
        TierHeader cur{};
        std::string why;
        if (!decode_header(h, cur, why)) continue;   // unreadable: leave it; the twin (or the refill) will decide
        cur.fill_state = state;
        encode_header(cur, h);
    }
    if (!tier_flush(t)) return false;
    return ok;
}

bool ExpertTier::open(const TierSpec& s, std::string& err) {
    spec_ = s;
    if (s.path.empty()) { err = "expert tier: empty path"; return false; }
    if (s.mib == 0) { err = "expert tier " + s.path + ": mib=0 names no capacity"; return false; }
#ifdef _WIN32
    HANDLE fh = CreateFileA(s.path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == nullptr || fh == INVALID_HANDLE_VALUE) {
        err = "expert tier " + s.path + ": cannot open read-write (Windows error " + std::to_string((unsigned long) GetLastError()) + ")";
        return false;
    }
    uint64_t dev_size = 0;
    std::string why;
    if (!tier_handle_size((void*) fh, dev_size, why)) {
        err = "expert tier " + s.path + ": " + why;
        CloseHandle(fh);
        return false;
    }
    if (!tier_handle_id((void*) fh, id_)) {
        err = "expert tier " + s.path + ": cannot tell which file it is (Windows error " + std::to_string((unsigned long) GetLastError()) + ")";
        CloseHandle(fh);
        return false;
    }
    if (dev_size < (s.mib << 20)) {
        char buf[192];
        std::snprintf(buf, sizeof buf, "expert tier %s: device is %.2f GiB, smaller than mib=%llu MiB - not a silent cut", s.path.c_str(),
                      (double) dev_size / 1073741824.0, (unsigned long long) s.mib);
        err = buf;
        CloseHandle(fh);
        return false;
    }
    // the lifetime lock (§5.4): one process uses a tier at a time. A byte range past the end of the file (setup and strata launch
    // lock the same range, strata_control/expert_tier.py device_lock); the handle's close releases it. Not built or run here.
    OVERLAPPED lock_at{};
    lock_at.Offset = (DWORD) (dev_size & 0xFFFFFFFFull);
    lock_at.OffsetHigh = (DWORD) (dev_size >> 32);
    if (!LockFileEx(fh, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &lock_at)) {
        err = "expert tier " + s.path + ": " + tier_busy_file_message();
        CloseHandle(fh);
        return false;
    }
    // a read-write mapping of the whole file; the view is requested with the matching access flags (FILE_MAP_READ | FILE_MAP_WRITE)
    HANDLE mm = CreateFileMappingA(fh, nullptr, PAGE_READWRITE, 0, 0, nullptr);
    if (mm == nullptr) { err = "expert tier " + s.path + ": CreateFileMapping failed (Windows error " + std::to_string((unsigned long) GetLastError()) + ")"; CloseHandle(fh); return false; }
    void* p = MapViewOfFile(mm, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
    if (p == nullptr) { err = "expert tier " + s.path + ": MapViewOfFile failed (Windows error " + std::to_string((unsigned long) GetLastError()) + ")"; CloseHandle(mm); CloseHandle(fh); return false; }
    os_handle_ = (void*) mm;
    win_fh_ = (void*) fh;
    posix_fd_ = -1;
    mapping_ = p;
    base_ = (uint8_t*) p;
    size_ = dev_size;
#else
    std::string why;
    const int fd = tier_open_rw(s.path, why);   // (a Linux block device O_EXCL: one the system uses is refused, expert_tier_dev.hpp)
    if (fd < 0) {
        err = "expert tier " + s.path + ": " + why;
        return false;
    }
    uint64_t dev_size = 0;
    bool is_block = false;
    if (!tier_fd_size(fd, dev_size, is_block, why) || !tier_fd_id(fd, id_)) {
        err = "expert tier " + s.path + ": " + (why.empty() ? std::string("cannot stat it") : why);
        ::close(fd);
        return false;
    }
    // the lifetime lock (§5.4): a regular file is held exclusively until close(); a block device is held by its O_EXCL open above
    if (!is_block && ::flock(fd, LOCK_EX | LOCK_NB) != 0) {
        err = "expert tier " + s.path + ": " + (errno == EWOULDBLOCK ? std::string(tier_busy_file_message()) : std::string(std::strerror(errno)));
        ::close(fd);
        return false;
    }
    if (dev_size < (s.mib << 20)) {
        char buf[192];
        std::snprintf(buf, sizeof buf, "expert tier %s: device is %.2f GiB, smaller than mib=%llu MiB - not a silent cut",
                      s.path.c_str(), (double) dev_size / 1073741824.0, (unsigned long long) s.mib);
        err = buf;
        ::close(fd);
        return false;
    }
    void* m = ::mmap(nullptr, (size_t) dev_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) {
        err = "expert tier " + s.path + ": mmap failed (errno " + std::to_string(errno) + ")";
        ::close(fd);
        return false;
    }
    posix_fd_ = fd;
    mapping_ = m;
    base_ = (uint8_t*) m;
    size_ = dev_size;
#endif

    // §5.3 step 5: register the mapping so GPU copies can DMA. The portable flag makes failure recoverable; the
    // device stays, unregistered, and copies go through the CPU exactly as #1250 does for pageable memory today.
#if defined(STRATA_TIER_CUDA_REGISTER)
    if (cudaHostRegister((void*) base_, (size_t) size_, cudaHostRegisterPortable) == cudaSuccess) registered_ = true;
    else {
        (void) cudaGetLastError();
        registered_ = false;
    }
#endif
    return true;
}

void ExpertTier::close() {
    if (base_ == nullptr) return;
    if (refill_) tier_write_fill_state(*this, 1);   // §6.4: mark for the next start's refill
#ifdef _WIN32
    UnmapViewOfFile(base_);
    if (os_handle_) CloseHandle((HANDLE) os_handle_);
    if (win_fh_) CloseHandle((HANDLE) win_fh_);
#else
    ::munmap((void*) base_, (size_t) size_);
    if (posix_fd_ >= 0) ::close(posix_fd_);
#endif
    posix_fd_ = -1;
    os_handle_ = nullptr;
    win_fh_ = nullptr;
    mapping_ = nullptr;
    base_ = nullptr;
    size_ = 0;
    registered_ = false;
    refill_ = false;
    id_ = TierFileId{};
}

static bool tier_flush(ExpertTier& t) { return t.flush(); }

bool ExpertTier::flush() {
    if (base_ == nullptr) return false;
    if (flush_fault_ && flush_fault_()) return false;   // (tests: an I/O error at this flush)
#ifndef _WIN32
    return ::msync((void*) base_, (size_t) size_, MS_SYNC) == 0;
#else
    // push the dirty pages of the view to the file, then the file's own cache to the disk
    if (!FlushViewOfFile((LPCVOID) base_, 0)) return false;
    return win_fh_ == nullptr || FlushFileBuffers((HANDLE) win_fh_) != 0;
#endif
}

uint64_t ExpertTier::capacity() const { return tier_capacity_bytes(spec_); }
uint8_t* ExpertTier::base() const { return base_; }
bool ExpertTier::registered() const { return registered_; }
uint64_t ExpertTier::size() const { return size_; }
void ExpertTier::note_refill() { refill_ = true; }
bool ExpertTier::blank_or_tier() const {
#ifdef _WIN32
    return win_fh_ != nullptr && tier_handle_blank_or_tier(win_fh_, size_, (uint64_t) spec_.mib << 20);
#else
    return posix_fd_ >= 0 && tier_fd_blank_or_tier(posix_fd_, size_, (uint64_t) spec_.mib << 20);
#endif
}
bool ExpertTier::refill_requested() const { return refill_; }

bool ExpertTier::read_header(TierHeader& h, std::string& why) const {
    if (base_ == nullptr) { why = "device is closed"; return false; }
    int which = -1;
    return read_header_pair(base_, base_ + kTierHeaderArea, h, which, why);
}

bool ExpertTier::write_header(const TierHeader& h, std::string& err) {
    if (base_ == nullptr) { err = "device is closed"; return false; }
    if (size_ < 2 * kTierHeaderArea + kTierAlign) { err = "device too small for the header areas"; return false; }
    uint8_t buf[kTierHeaderBytes];
    encode_header(h, buf);
    std::memcpy(base_, buf, kTierHeaderBytes);
    if (!tier_flush(*this)) { err = "flush after header A failed"; return false; }
    std::memcpy(base_ + kTierHeaderArea, buf, kTierHeaderBytes);
    if (!tier_flush(*this)) { err = "flush after header B failed"; return false; }
    return true;
}

// ---------------------------------------------------------------- §7.3 / §12.4: the set -----------------------------
struct ChunkCrc {
    uint64_t cursor = 0;
    std::vector<uint32_t> state;
    explicit ChunkCrc(size_t chunks) : state(chunks, kCrc32cInit) {}
    void feed(const uint8_t* p, size_t n) {
        while (n) {
            const uint64_t pos = cursor % kTierChunk;
            const size_t take = (size_t) std::min<uint64_t>(n, kTierChunk - pos);
            state[(size_t) (cursor / kTierChunk)] = tier_crc32c_feed(state[(size_t) (cursor / kTierChunk)], p, take);
            p += take; cursor += take; n -= take;
        }
    }
    std::vector<uint32_t> finish() {
        std::vector<uint32_t> out(state.size());
        for (size_t i = 0; i < state.size(); ++i) out[i] = tier_crc32c_finish(state[i]);
        return out;
    }
};

void ExpertTierSet::set_flush_fault(std::function<bool(size_t)> f) {
    flush_fault_ = std::move(f);
    for (size_t d = 0; d < tiers_.size(); ++d)
        tiers_[d]->set_flush_fault([this, d] { return flush_fault_ && flush_fault_(d); });
}

bool ExpertTierSet::open(const std::vector<TierSpec>& specs, const TierIdentity& id, std::string& err, bool& reused,
                         const std::vector<std::string>& protect) {
    id_ = id;
    tiers_.clear();
    if (specs.empty()) { err = "expert tier: no --expert-tier entries"; return false; }
    for (size_t i = 0; i < specs.size(); ++i)
        for (size_t j = i + 1; j < specs.size(); ++j)
            if (specs[i].path == specs[j].path) {
                err = "expert tier: --expert-tier " + specs[i].path + " given twice; device paths must be distinct";
                return false;
            }
    // §5.4, before any device is opened: two names of one device (or a model file) are refused by what they are, not by the lock
    // that the second name would fail on
    {
        std::vector<std::string> paths;
        std::vector<TierFileId> ids;
        for (const TierSpec& s : specs) {
            TierFileId fid;
            paths.push_back(s.path);
            if (!tier_path_id(s.path, fid)) fid = TierFileId{};   // (an unopenable device: its own open says why)
            ids.push_back(fid);
        }
        if (!tier_check_overlaps(paths, ids, protect, err)) return false;
    }
    for (const TierSpec& s : specs) {
        auto t = std::make_unique<ExpertTier>();
        if (!t->open(s, err)) { close(); return false; }
        if (flush_fault_) {
            const size_t d = tiers_.size();
            t->set_flush_fault([this, d] { return flush_fault_ && flush_fault_(d); });
        }
        tiers_.push_back(std::move(t));
    }
    // §5.4: the devices are compared by what they ARE (a symlink, a hard link, two names of one block device, a partition
    // and its disk), and none may be - or hold - a file the engine reads the model from.  Nothing is written before this.
    {
        std::vector<std::string> paths;
        std::vector<TierFileId> ids;
        for (const auto& t : tiers_) { paths.push_back(t->path()); ids.push_back(t->file_id()); }
        if (!tier_check_overlaps(paths, ids, protect, err)) { close(); return false; }
    }
    // §5.4: a device is written over only when it is blank or already an expert tier; anything else is someone's data
    for (const auto& t : tiers_)
        if (!t->blank_or_tier()) { err = tier_foreign_data_message(t->path()); close(); return false; }

    // §6.3: every header valid, the identity the same everywhere, the placement tables identical, fill_state 2.
    reused = true;
    const int entry_bytes = tier_placement_entry_bytes(id.n_layers, id.n_expert, (int) specs.size());
    char pack_field[32], profile_field[32];
    id_to_field(id.pack_id, pack_field);
    id_to_field(id.profile_id, profile_field);
    std::vector<std::vector<uint8_t>> tables(tiers_.size());
    for (size_t i = 0; i < tiers_.size() && reused; ++i) {
        TierHeader h{};
        std::string why;
        if (!tiers_[i]->read_header(h, why)) { notes_.push_back("tier " + specs[i].path + ": " + why + " -> refill"); reused = false; continue; }
        if (h.capacity_bytes != tier_capacity_bytes(specs[i])) {
            notes_.push_back("tier " + specs[i].path + ": header capacity_bytes " + std::to_string(h.capacity_bytes) + " for this start's " + std::to_string(tier_capacity_bytes(specs[i])) + " -> refill");
            reused = false;
            continue;
        }
        if (h.device_count != specs.size() || h.device_index != i) {
            notes_.push_back("tier " + specs[i].path + ": header device_index " + std::to_string(h.device_index) + " of " + std::to_string(h.device_count) + " for slot " + std::to_string(i) + " -> refill");
            reused = false;
            continue;
        }
        if (h.n_layers != (uint64_t) id.n_layers || h.n_expert != (uint64_t) id.n_expert || h.version != kTierLayoutVersion) {
            notes_.push_back("tier " + specs[i].path + ": header shape/version is for another model -> refill");
            reused = false;
            continue;
        }
        if (std::memcmp(h.pack_id, pack_field, 32) != 0 || std::memcmp(h.profile_id, profile_field, 32) != 0) {
            notes_.push_back("tier " + specs[i].path + ": identity (pack or profile) changed -> refill");
            reused = false;
            continue;
        }
        if (h.fill_state != 2) {
            notes_.push_back("tier " + specs[i].path + ": fill_state " + std::to_string(h.fill_state) + " (torn or never finished) -> refill");
            reused = false;
            continue;
        }
        if (h.placement_len % (uint64_t) entry_bytes != 0) {
            notes_.push_back("tier " + specs[i].path + ": placement table size does not match this shape -> refill");
            reused = false;
            continue;
        }
        const uint8_t* tb = tiers_[i]->base() + (size_t) h.placement_off;
        if (h.placement_off + h.placement_len > tiers_[i]->size()) {
            notes_.push_back("tier " + specs[i].path + ": placement table past the device end -> refill");
            reused = false;
            continue;
        }
        if (tier_crc32c(tb, (size_t) h.placement_len) != h.placement_crc) {
            notes_.push_back("tier " + specs[i].path + ": placement table CRC does not match -> refill");
            reused = false;
            continue;
        }
        tables[i].assign(tb, tb + h.placement_len);
    }
    if (reused) {
        for (size_t i = 1; i < tables.size(); ++i)
            if (tables[i] != tables[0]) {
                notes_.push_back("tier devices do not carry the same placement table -> refill");
                reused = false;
                break;
            }
    }
    if (reused && id.layer_blob_bytes.size() < (size_t) id.n_layers) {
        notes_.push_back("expert tier: the layout is missing blob sizes -> refill");
        reused = false;
    }
    if (reused) {
        std::vector<TierPlacementEntry> entries;
        std::string why;
        if (!decode_placement_table(tables[0].data(), tables[0].size(), entry_bytes, entries, why)) {
            notes_.push_back("placement table: " + why + " -> refill");
            reused = false;
        } else if (!build_lookup(entries, entry_bytes, why)) {
            notes_.push_back("placement table: " + why + " -> refill");
            clear_lookup();
            reused = false;
        } else if (!check_geometry(why)) {
            notes_.push_back("tier geometry: " + why + " -> refill");
            clear_lookup();
            reused = false;
        } else {
            for (const TierSpec& s : specs)
                if (s.verify_full) { verify_chunks(); break; }
        }
    }
    for (size_t i = 0; i < tiers_.size(); ++i) {
        char buf[256];
        std::snprintf(buf, sizeof buf, "expert tier: %s %.2f GiB %s", tiers_[i]->path().c_str(),
                      (double) tiers_[i]->capacity() / 1073741824.0,
                      tiers_[i]->registered() ? "registered (DMA copies)" : "unregistered (CPU copies)");
        notes_.push_back(buf);
    }
    return true;
}

void ExpertTierSet::clear_lookup() {
    dev_of_.clear();
    off_of_.clear();
    valid_.clear();
    data_off_.clear();
    used_.clear();
    bad_chunk_.clear();
    bytes_held_ = 0;
    experts_held_ = 0;
}

bool ExpertTierSet::build_lookup(const std::vector<TierPlacementEntry>& entries, int entry_bytes, std::string& why) {
    const size_t total = (size_t) id_.n_layers * (size_t) id_.n_expert;
    dev_of_.assign(total, -1);
    off_of_.assign(total, 0);
    valid_.assign(total, 0);
    data_off_.assign(tiers_.size(), 0);
    used_.assign(tiers_.size(), 0);
    bad_chunk_.assign(tiers_.size(), {});
    bytes_held_ = 0;
    experts_held_ = 0;
    for (const auto& e : entries) {
        // a table that CRC-checks but names an expert outside the shape, a device that is not configured, or the same
        // expert twice was not written by this engine for this start: refill rather than guess
        if (e.layer < 0 || e.layer >= id_.n_layers || e.expert < 0 || e.expert >= id_.n_expert || e.device < 0 ||
            (size_t) e.device >= tiers_.size()) {
            why = "an entry names layer " + std::to_string(e.layer) + ", expert " + std::to_string(e.expert) + ", device " + std::to_string(e.device) + " outside this start's shape";
            return false;
        }
        const size_t idx = (size_t) e.layer * (size_t) id_.n_expert + (size_t) e.expert;
        if (valid_[idx]) {
            why = "layer " + std::to_string(e.layer) + ", expert " + std::to_string(e.expert) + " is placed twice";
            return false;
        }
        dev_of_[idx] = e.device;
        off_of_[idx] = e.offset;
        valid_[idx] = 1;
        ++experts_held_;
        bytes_held_ += id_.layer_blob_bytes[(size_t) e.layer];
        used_[(size_t) e.device] = std::max(used_[(size_t) e.device], e.offset + id_.layer_blob_bytes[(size_t) e.layer]);
    }
    // §5.2: the data area starts after header A, header B, the placement table and the chunk table.
    for (size_t d = 0; d < tiers_.size(); ++d) {
        const uint64_t placement_len = (uint64_t) entries.size() * (uint64_t) entry_bytes;
        const uint64_t chunk_count = (used_[d] + kTierChunk - 1) / kTierChunk;
        data_off_[d] = (2 * kTierHeaderArea + placement_len + chunk_count * 4 + kTierAlign - 1) / kTierAlign * kTierAlign;
    }
    return true;
}

// A reused placement is read straight out of the mapping, so before it is trusted every blob must lie inside its
// device's configured capacity, and each device's chunk table must be where, and as long as, this start computes it.
bool ExpertTierSet::check_geometry(std::string& why) {
    for (size_t idx = 0; idx < dev_of_.size(); ++idx) {
        if (!valid_[idx]) continue;
        const size_t d = (size_t) dev_of_[idx];
        const uint64_t bytes = id_.layer_blob_bytes[idx / (size_t) id_.n_expert];
        if (data_off_[d] + off_of_[idx] + bytes > tiers_[d]->capacity() || data_off_[d] + off_of_[idx] + bytes > tiers_[d]->size()) {
            why = tiers_[d]->path() + ": a blob reaches past the device's capacity";
            return false;
        }
    }
    for (size_t d = 0; d < tiers_.size(); ++d) {
        TierHeader h{};
        std::string hw;
        if (!tiers_[d]->read_header(h, hw)) { why = tiers_[d]->path() + ": " + hw; return false; }
        const uint64_t want_chunks = (used_[d] + kTierChunk - 1) / kTierChunk;
        if (h.chunk_count != want_chunks) {
            why = tiers_[d]->path() + ": chunk table holds " + std::to_string(h.chunk_count) + " chunks, this placement needs " + std::to_string(want_chunks);
            return false;
        }
        if (h.chunk_off + h.chunk_count * 4 > tiers_[d]->size() || h.chunk_off + h.chunk_count * 4 > data_off_[d]) {
            why = tiers_[d]->path() + ": chunk table does not fit before the data";
            return false;
        }
    }
    return true;
}

// §6.4 verify=full: every chunk of every device against its stored CRC.  EVERY bad chunk is kept (not the last one
// found): each expert touching one is served from the model files for this run, and the device is marked for refill.
void ExpertTierSet::verify_chunks() {
    for (size_t d = 0; d < tiers_.size(); ++d) {
        TierHeader h{};
        std::string why;
        if (!tiers_[d]->read_header(h, why)) continue;   // open() already read it; belt and braces
        const uint8_t* cb = tiers_[d]->base() + (size_t) h.chunk_off;
        std::vector<uint32_t> table = decode_chunk_table(cb, h.chunk_count);
        bad_chunk_[d].assign((size_t) h.chunk_count, 0);
        uint64_t n_bad = 0, first_bad = 0;
        for (uint64_t c = 0; c < h.chunk_count; ++c) {
            const uint64_t begin = c * kTierChunk;
            const uint64_t end = std::min<uint64_t>(used_[d], (c + 1) * kTierChunk);
            const uint32_t mine = tier_crc32c(tiers_[d]->base() + (size_t) (data_off_[d] + begin), (size_t) (end - begin));
            if (table[(size_t) c] == mine) continue;
            if (n_bad++ == 0) first_bad = c;
            bad_chunk_[d][(size_t) c] = 1;
        }
        if (n_bad == 0) continue;
        char buf[256];
        std::snprintf(buf, sizeof buf, "expert tier: %s %llu chunk(s) CRC mismatch (first: %llu): their blobs read from the file tier, and the device is marked for refill at the next start",
                      tiers_[d]->path().c_str(), (unsigned long long) n_bad, (unsigned long long) first_bad);
        notes_.push_back(buf);
        tiers_[d]->note_refill();
    }
    // invalidate every held entry that touches a bad chunk (§6.4: the blob is served from the file tier instead)
    for (size_t idx = 0; idx < dev_of_.size(); ++idx) {
        if (dev_of_[idx] < 0 || !valid_[idx]) continue;
        const size_t d = (size_t) dev_of_[idx];
        if (bad_chunk_[d].empty()) continue;
        const uint64_t bytes = id_.layer_blob_bytes[idx / (size_t) id_.n_expert];
        const uint64_t lo = off_of_[idx], hi = off_of_[idx] + bytes;
        for (uint64_t c = lo / kTierChunk; c * kTierChunk < hi; ++c)
            if (c < bad_chunk_[d].size() && bad_chunk_[d][(size_t) c]) { valid_[idx] = 0; break; }
    }
}

ExpertTier::~ExpertTier() { close(); }

static uint64_t entries_bytes(const std::vector<TierPlacementEntry>& entries, const TierIdentity& id) {
    uint64_t total = 0;
    for (const auto& e : entries) total += id.layer_blob_bytes[(size_t) e.layer];
    return total;
}

bool ExpertTierSet::fill(ExpertSource& src, const std::vector<std::pair<int32_t, int32_t>>& ranked_rest,
                         const TierIdentity& id, std::string& err) {
    id_ = id;
    if (tiers_.empty()) { err = "expert tier fill: the tier is not open"; return false; }
    const int entry_bytes = tier_placement_entry_bytes(id.n_layers, id.n_expert, (int) tiers_.size());
    if (id.layer_blob_bytes.size() < (size_t) id.n_layers) { err = "expert tier fill: the layout is missing blob sizes"; return false; }

    // §5.2: the header areas, the placement table and the chunk table are overhead on the device; the planner
    // gets what is left of the configured capacity.
    uint64_t min_blob = UINT64_MAX;
    for (const auto& pr : ranked_rest)
        min_blob = std::min(min_blob, id.layer_blob_bytes[(size_t) pr.first]);
    if (ranked_rest.empty()) min_blob = 0;
    const uint64_t placement_max = (uint64_t) ranked_rest.size() * (uint64_t) entry_bytes;
    std::vector<TierDevice> devices(tiers_.size());
    for (size_t d = 0; d < tiers_.size(); ++d) {
        const uint64_t cap = tiers_[d]->capacity();
        const uint64_t chunk_max = cap / kTierChunk + 2;   // the chunk table's upper bound for this device
        // header areas + placement table + chunk table, rounded UP to the blob alignment: the data region begins at
        // an aligned offset, so what the planner may spend is the capacity less the aligned overhead
        const uint64_t overhead = (2 * kTierHeaderArea + placement_max + chunk_max * 4 + kTierAlign - 1) / kTierAlign * kTierAlign;
        devices[d].capacity_bytes = cap > overhead ? cap - overhead : 0;
    }
    const auto pl = plan_expert_tiers(ranked_rest, id.layer_blob_bytes, devices);

    // the placement table: tier-held entries in rank order, identical on every device (§6.1)
    std::vector<TierPlacementEntry> entries;
    std::vector<uint64_t> used(tiers_.size(), 0);
    for (size_t i = 0; i < ranked_rest.size(); ++i) {
        if (pl[i].device < 0) continue;
        entries.push_back({ranked_rest[i].first, ranked_rest[i].second, pl[i].device, pl[i].offset});
        used[(size_t) pl[i].device] = std::max(used[(size_t) pl[i].device], pl[i].offset + id.layer_blob_bytes[(size_t) ranked_rest[i].first]);
    }
    const uint64_t placement_bytes = (uint64_t) entries.size() * (uint64_t) entry_bytes;
    std::vector<uint32_t> chunk_count(tiers_.size(), 0);
    for (size_t d = 0; d < tiers_.size(); ++d) chunk_count[d] = (uint32_t) ((used[d] + kTierChunk - 1) / kTierChunk);

    // every blob must fit on its device: the budget above is an upper bound, this is the exact check. The offsets
    // recorded for the reuse path come from the same formula (build_lookup), so both sides agree.
    for (size_t d = 0; d < tiers_.size(); ++d) {
        const uint64_t data_off = (2 * kTierHeaderArea + placement_bytes + (uint64_t) chunk_count[d] * 4 + kTierAlign - 1) / kTierAlign * kTierAlign;
        if (data_off + used[d] > tiers_[d]->capacity() || data_off + used[d] > tiers_[d]->size()) {   // (the reserve is never written)
            err = "expert tier fill: " + tiers_[d]->path() + " cannot hold the placement inside the device";
            return false;
        }
    }
    data_off_.resize(tiers_.size());
    for (size_t d = 0; d < tiers_.size(); ++d)
        data_off_[d] = (2 * kTierHeaderArea + placement_bytes + (uint64_t) chunk_count[d] * 4 + kTierAlign - 1) / kTierAlign * kTierAlign;

    // pass 1: identity headers with fill_state = 1, so a fill interrupted anywhere is found and refilled (§6.4)
    const std::vector<uint8_t> placement_table = encode_placement_table(entries, entry_bytes);
    for (size_t d = 0; d < tiers_.size(); ++d) {
        TierHeader h{};
        std::memcpy(h.magic, kTierMagic, 8);
        h.version = kTierLayoutVersion;
        h.device_index = (uint32_t) d;
        h.device_count = (uint32_t) tiers_.size();
        id_to_field(id.pack_id, h.pack_id);
        id_to_field(id.profile_id, h.profile_id);
        h.n_layers = (uint64_t) id.n_layers;
        h.n_expert = (uint64_t) id.n_expert;
        h.capacity_bytes = tiers_[d]->capacity();
        h.placement_off = 2 * kTierHeaderArea;
        h.placement_len = placement_bytes;
        h.placement_crc = placement_table.empty() ? 0u : tier_crc32c(placement_table.data(), placement_table.size());
        h.chunk_off = h.placement_off + h.placement_len;
        h.chunk_count = chunk_count[d];
        h.fill_state = 1;
        std::string werr;
        if (!tiers_[d]->write_header(h, werr)) { err = "expert tier fill: " + tiers_[d]->path() + ": " + werr; return false; }
    }
    for (size_t d = 0; d < tiers_.size(); ++d)
        std::memcpy(tiers_[d]->base() + (size_t) (2 * kTierHeaderArea), placement_table.data(), placement_table.size());
    // From here a failure must never leave a header that claims a complete tier: every device is marked for refill, so its
    // close() writes fill_state 1 over both headers (a header whose own flush failed may still reach the disk later).
    auto abandon = [&] { for (auto& t : tiers_) t->note_refill(); };

    // §6.4: data and chunk checksums first.  Blobs stream into the device in rank order; the 4 KiB gaps are
    // written as zeros so a later refill with a different placement cannot leave stale bytes inside a chunk.
    uint64_t max_blob = 0;
    for (const auto& pr : ranked_rest) max_blob = std::max(max_blob, id.layer_blob_bytes[(size_t) pr.first]);
    std::vector<uint8_t> stage((size_t) (max_blob ? max_blob : 1));
    std::vector<ChunkCrc> crcs;
    for (size_t d = 0; d < tiers_.size(); ++d) crcs.emplace_back((size_t) chunk_count[d]);
    std::vector<uint64_t> cursor(tiers_.size(), 0);
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto& e : entries) {
        uint8_t* dev_base = (uint8_t*) tiers_[(size_t) e.device]->base();
        const uint64_t bytes = id.layer_blob_bytes[(size_t) e.layer];
        if (!src.copy_blob(e.layer, e.expert, stage.data())) {
            err = "expert tier fill: the file tier cannot serve (layer " + std::to_string(e.layer) + ", expert " + std::to_string(e.expert) + ")";
            abandon();
            return false;
        }
        if (e.offset > cursor[(size_t) e.device]) {
            uint8_t* gap = dev_base + (size_t) (data_off_[(size_t) e.device] + cursor[(size_t) e.device]);
            std::memset(gap, 0, (size_t) (e.offset - cursor[(size_t) e.device]));
            crcs[(size_t) e.device].feed(gap, (size_t) (e.offset - cursor[(size_t) e.device]));
        }
        uint8_t* dst = dev_base + (size_t) (data_off_[(size_t) e.device] + e.offset);
        std::memcpy(dst, stage.data(), (size_t) bytes);
        crcs[(size_t) e.device].feed(dst, (size_t) bytes);
        cursor[(size_t) e.device] = e.offset + bytes;
    }

    // §6.4's write order: the data, the placement table and the chunk checksums of EVERY device are on the disk (flushed, the
    // flush checked) before any header says fill_state = 2; only then header A, then B.  A flush that fails stops here with
    // every header still at fill_state 1, so a later start refills instead of trusting bytes that may not have landed.
    std::vector<TierHeader> done(tiers_.size());
    for (size_t d = 0; d < tiers_.size(); ++d) {
        const std::vector<uint32_t> mine = crcs[d].finish();
        if (mine.size() != (size_t) chunk_count[d]) { err = "expert tier fill: chunk count drift"; abandon(); return false; }
        const std::vector<uint8_t> tbl = encode_chunk_table(mine);
        std::string why;
        if (!tiers_[d]->read_header(done[d], why)) { err = "expert tier fill: " + tiers_[d]->path() + ": header lost: " + why; abandon(); return false; }
        std::memcpy(tiers_[d]->base() + (size_t) done[d].chunk_off, tbl.data(), tbl.size());
        if (!tiers_[d]->flush()) {
            err = "expert tier fill: " + tiers_[d]->path() + ": the experts could not be written to the device (flush failed); "
                  "the tier stays marked unfinished and is written again at the next start";
            abandon();
            return false;
        }
    }
    for (size_t d = 0; d < tiers_.size(); ++d) {
        done[d].fill_state = 2;
        std::string werr;
        if (!tiers_[d]->write_header(done[d], werr)) {
            err = "expert tier fill: " + tiers_[d]->path() + ": " + werr + "; the tier is written again at the next start";
            abandon();
            return false;
        }
    }

    fill_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fill_b_ = entries_bytes(entries, id);
    std::string lwhy;
    if (!build_lookup(entries, entry_bytes, lwhy)) { err = "expert tier fill: the placement just written is not usable: " + lwhy; clear_lookup(); return false; }
    return true;
}

const uint8_t* ExpertTierSet::blob(int64_t layer, int64_t expert) const {
    if (layer < 0 || expert < 0 || layer >= id_.n_layers || expert >= id_.n_expert || dev_of_.empty()) return nullptr;
    const size_t idx = (size_t) layer * (size_t) id_.n_expert + (size_t) expert;
    if (dev_of_[idx] < 0 || !valid_[idx]) return nullptr;
    const size_t d = (size_t) dev_of_[idx];
    const uint8_t* b = tiers_[d]->base();
    if (b == nullptr) return nullptr;
    return b + (size_t) (data_off_[d] + off_of_[idx]);
}

bool ExpertTierSet::has(int64_t layer, int64_t expert) const { return blob(layer, expert) != nullptr; }

bool ExpertTierSet::registered_of(int64_t layer, int64_t expert) const {
    if (layer < 0 || expert < 0 || layer >= id_.n_layers || expert >= id_.n_expert || dev_of_.empty()) return false;
    const size_t idx = (size_t) layer * (size_t) id_.n_expert + (size_t) expert;
    if (dev_of_[idx] < 0 || !valid_[idx]) return false;
    return tiers_[(size_t) dev_of_[idx]]->registered();   // §7.6: pinned() is true exactly when the device is registered
}

uint64_t ExpertTierSet::bytes_held() const { return bytes_held_; }
int64_t ExpertTierSet::experts_held() const { return experts_held_; }
double ExpertTierSet::fill_seconds() const { return fill_s_; }
uint64_t ExpertTierSet::fill_bytes() const { return fill_b_; }
const std::vector<std::string>& ExpertTierSet::notes() const { return notes_; }
void ExpertTierSet::set_worker_cap(int n) { worker_cap_ = n < 1 ? 1 : n; }

// §7.3: the prompt path's tier share.  One worker thread per device, up to the cap, each copying its device's
// share of the batch in layer order; the parallelism across devices is the point (§7.3, §12.4).
bool ExpertTierSet::copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n,
                               uint64_t& bytes_out) {
    bytes_out = 0;
    if (tiers_.empty()) return false;
    std::vector<std::vector<size_t>> by_dev(tiers_.size());
    for (size_t i = 0; i < n; ++i) {
        const int64_t l = layers[i], e = experts[i];
        if (l < 0 || e < 0 || l >= id_.n_layers || e >= id_.n_expert || dst[i] == nullptr) return false;
        const size_t idx = (size_t) l * (size_t) id_.n_expert + (size_t) e;
        if (dev_of_[idx] < 0 || !valid_[idx]) return false;   // the caller routes non-held blobs to the file path
        by_dev[(size_t) dev_of_[idx]].push_back(i);
    }
    std::atomic<uint64_t> total{0};
    std::vector<std::thread> workers;
    for (size_t d = 0; d < by_dev.size(); ++d) {
        auto& list = by_dev[d];
        // layer order for the device's share (the batch's own order is file order)
        std::sort(list.begin(), list.end(), [&](size_t a, size_t b) {
            if (layers[a] != layers[b]) return layers[a] < layers[b];
            return experts[a] < experts[b];
        });
        const int w = std::min<int>(worker_cap_, (int) list.size());
        for (int t = 0; t < w; ++t) {
            workers.emplace_back([this, d, &list, t, w, &total, layers, experts, dst] {
                for (size_t j = (size_t) t; j < list.size(); j += (size_t) w) {
                    const size_t i = list[j];
                    const uint8_t* p = blob(layers[i], experts[i]);
                    if (p == nullptr) return;
                    const uint64_t bytes = id_.layer_blob_bytes[(size_t) layers[i]];
                    std::memcpy(dst[i], (void*) p, (size_t) bytes);
                    total.fetch_add(bytes, std::memory_order_relaxed);
                }
            });
        }
    }
    for (auto& w : workers) w.join();
    bytes_out = total.load();
    return true;
}

void ExpertTierSet::close() {
    for (auto& t : tiers_) t->close();   // §6.4: a device that asked for a refill gets its fill_state = 1 here
    tiers_.clear();
    clear_lookup();
}

}  // namespace strata::core
