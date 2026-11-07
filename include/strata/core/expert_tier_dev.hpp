// include/strata/core/expert_tier_dev.hpp - an expert tier device: how big it is, what it is, and whether it is safe to write.
// One answer for both engines.
//
// Size. A regular file reports its length in st_size. A Linux block device (a partition, an LVM volume, a loop device)
// reports st_size == 0 from stat(): its capacity comes from ioctl(BLKGETSIZE64). Both engines map the device MAP_SHARED,
// which Linux allows for a block device. Other kinds of file (a pipe, a character device) have no capacity to map and are
// refused with the reason. Mac: regular files only (the tier is not offered there, docs/EXPERT_TIER_PLAN.md §13).
// Windows: regular files only; a raw \\.\PhysicalDriveN cannot be mapped as a file, so GetFileSizeEx fails on it.
//
// Identity. A device is compared by what the system says it IS, never by the path it was given: a symlink, a hard link, a
// second name for the same file or two nodes of one block device are the same device. POSIX: (st_dev, st_ino) for a file,
// st_rdev for a block device; Windows: the volume serial number and the file index.
//
// Safety. A Linux block device is opened O_EXCL: the kernel then refuses (EBUSY) a device that is mounted, used as swap,
// part of a RAID or LVM volume, or has a partition in any of those uses - so the engine never writes over a file system the
// system is using. A device that holds one of the model's files, or IS one of them, is refused by the callers
// (tier_same_file / tier_holds_file below).
//
// Lock. A regular-file tier is held with flock (POSIX) or a byte range past the end (LockFileEx, Windows) from its open until its
// close, so one process uses a tier at a time; the Python side (strata_control/expert_tier.py device_lock) takes the same lock.
#pragma once

#include "strata/core/expert_tier_layout.hpp"   // kTierClaimBytes, tier_header_magic

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <cerrno>
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <unistd.h>
#  if defined(__linux__)
#    include <linux/fs.h>
#    include <sys/ioctl.h>
#    include <sys/sysmacros.h>
#  endif
#endif

namespace strata::core {

/// Said when another process holds a regular-file tier (its lock is taken by the engine at open and kept until close; setup and
/// strata launch take the same lock while they check or clear a device). A Linux block device is held by its O_EXCL open.
inline const char* tier_busy_file_message() {
    return "another process holds this expert tier (a running Strata engine, setup or launch): one tier is used by one process "
           "at a time";
}

/// What a device or a file is, as the system sees it.  `valid` false: it could not be looked at.
struct TierFileId {
    bool valid = false;
    bool block = false;   // a block device: `rdev` is the device it names
    uint64_t dev = 0;     // the file system the file lives on (POSIX st_dev; Windows: the volume serial number)
    uint64_t ino = 0;     // the file on it (POSIX st_ino; Windows: the file index)
    uint64_t rdev = 0;    // a block device's own number
};

#ifndef _WIN32
inline TierFileId tier_id_of_stat(const struct stat& st) {
    TierFileId id;
    id.valid = true;
    id.block = S_ISBLK(st.st_mode);
    id.dev = (uint64_t) st.st_dev;
    id.ino = (uint64_t) st.st_ino;
    id.rdev = id.block ? (uint64_t) st.st_rdev : 0;
    return id;
}
inline bool tier_fd_id(int fd, TierFileId& id) {
    struct stat st {};
    if (::fstat(fd, &st) != 0) return false;
    id = tier_id_of_stat(st);
    return true;
}
/// The identity of a path, following symlinks (the file a symlink names is what would be written).
inline bool tier_path_id(const std::string& path, TierFileId& id) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) return false;
    id = tier_id_of_stat(st);
    return true;
}
#else
inline bool tier_handle_id(void* handle, TierFileId& id) {
    BY_HANDLE_FILE_INFORMATION fi;
    if (!GetFileInformationByHandle((HANDLE) handle, &fi)) return false;
    id = TierFileId{};
    id.valid = true;
    id.dev = (uint64_t) fi.dwVolumeSerialNumber;
    id.ino = ((uint64_t) fi.nFileIndexHigh << 32) | (uint64_t) fi.nFileIndexLow;
    return true;
}
inline bool tier_path_id(const std::string& path, TierFileId& id) {
    HANDLE h = CreateFileA(path.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                           FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    const bool ok = tier_handle_id((void*) h, id);
    CloseHandle(h);
    return ok;
}
#endif

/// Linux: the whole disk a partition belongs to (sysfs), or `dev` itself when it is not a partition or this is not Linux.
inline uint64_t tier_whole_disk(uint64_t dev) {
#if defined(__linux__)
    char p[96];
    std::snprintf(p, sizeof p, "/sys/dev/block/%u:%u/partition", (unsigned) major((dev_t) dev), (unsigned) minor((dev_t) dev));
    if (::access(p, F_OK) != 0) return dev;
    std::snprintf(p, sizeof p, "/sys/dev/block/%u:%u/../dev", (unsigned) major((dev_t) dev), (unsigned) minor((dev_t) dev));
    std::FILE* f = std::fopen(p, "r");
    if (f == nullptr) return dev;
    unsigned ma = 0, mi = 0;
    const bool ok = std::fscanf(f, "%u:%u", &ma, &mi) == 2;
    std::fclose(f);
    return ok ? (uint64_t) makedev(ma, mi) : dev;
#else
    return dev;
#endif
}

/// The same device, or two block devices of which one is a partition of the other (writing one writes the other).
inline bool tier_same_file(const TierFileId& a, const TierFileId& b) {
    if (!a.valid || !b.valid) return false;
    if (a.block != b.block) return false;
    if (a.block)
        return a.rdev == b.rdev || tier_whole_disk(a.rdev) == b.rdev || tier_whole_disk(b.rdev) == a.rdev;
    return a.dev == b.dev && a.ino == b.ino;
}

/// A block device that carries the file system `file` lives on (the device itself, or the disk of its partition).
inline bool tier_holds_file(const TierFileId& tier, const TierFileId& file) {
    if (!tier.valid || !file.valid || !tier.block || file.block) return false;
    return file.dev == tier.rdev || tier_whole_disk(file.dev) == tier.rdev;
}

/// Checks the tier devices against each other and against the files that must never be written (the model's files, the
/// expert profile).  `ids[i]` belongs to `paths[i]`.  A protected path that does not exist is skipped.  False with `why`.
inline bool tier_check_overlaps(const std::vector<std::string>& paths, const std::vector<TierFileId>& ids,
                                const std::vector<std::string>& protect, std::string& why) {
    for (size_t i = 0; i < ids.size(); ++i)
        for (size_t j = i + 1; j < ids.size(); ++j)
            if (tier_same_file(ids[i], ids[j])) {
                why = "expert tier: " + paths[i] + " and " + paths[j] +
                      " are the same device (a link, another name for the same file, or a partition of the other); give each device once";
                return false;
            }
    for (const std::string& p : protect) {
        if (p.empty()) continue;
        TierFileId f;
        if (!tier_path_id(p, f)) continue;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (tier_same_file(ids[i], f)) {
                why = "expert tier " + paths[i] + ": it is the model's file " + p +
                      " (the same file under another name): filling the tier would overwrite the model. Put the tier outside the model's folder";
                return false;
            }
            if (tier_holds_file(ids[i], f)) {
                why = "expert tier " + paths[i] + ": the model's file " + p + " is on this block device: filling the tier would overwrite it";
                return false;
            }
        }
    }
    return true;
}

/// Whether the device may be written over by a fill of `mib_bytes`: a tier header (either engine's magic at 0 or 64 KiB, whatever
/// its state) or every byte the fill may write is zero. Reads the device itself, not a mapping of it, so a sparse 64 GiB file
/// costs its data extents only. Data past the first 128 KiB counts: a prefix check calls a sparse file with data 3 MiB in blank.
/// Any byte the system cannot read is treated as data (refused, never guessed).
#ifndef _WIN32
inline bool tier_fd_has_data(int fd, uint64_t end) {
    auto read_nonzero = [fd](uint64_t start, uint64_t stop) {
        std::vector<uint8_t> buf((size_t) std::min<uint64_t>(stop - start, 1ull << 20));
        for (uint64_t pos = start; pos < stop;) {
            const size_t want = (size_t) std::min<uint64_t>(buf.size(), stop - pos);
            const ssize_t n = ::pread(fd, buf.data(), want, (off_t) pos);
            if (n < 0) { if (errno == EINTR) continue; return true; }
            if (n == 0) return false;
            for (ssize_t i = 0; i < n; ++i)
                if (buf[(size_t) i] != 0) return true;
            pos += (uint64_t) n;
        }
        return false;
    };
    if (end == 0) return false;
#if defined(SEEK_DATA) && defined(SEEK_HOLE)
    bool unanswered = false;
    for (uint64_t pos = 0; pos < end;) {
        const off_t data = ::lseek(fd, (off_t) pos, SEEK_DATA);
        if (data < 0) {
            if (errno == ENXIO) return false;            // no data at or after pos
            unanswered = true;                           // not answered by this file system: read it all below
            break;
        }
        if ((uint64_t) data >= end) return false;
        const off_t hole = ::lseek(fd, data, SEEK_HOLE);
        const uint64_t stop = (hole < 0 || (uint64_t) hole <= (uint64_t) data) ? end : std::min<uint64_t>((uint64_t) hole, end);
        if (read_nonzero((uint64_t) data, stop)) return true;
        pos = stop;
    }
    if (!unanswered) return false;
#endif
    return read_nonzero(0, end);
}
/// POSIX: header magic or a blank range, for the device `fd` of `size` bytes that a fill of `mib_bytes` may write over.
inline bool tier_fd_blank_or_tier(int fd, uint64_t size, uint64_t mib_bytes) {
    uint8_t head[kTierClaimBytes] = {};
    const size_t got = (size_t) std::min<uint64_t>(sizeof head, size);
    if (got > 0 && ::pread(fd, head, got, 0) < 0) return false;   // unreadable: refused
    if (tier_header_magic(head, got)) return true;
    return !tier_fd_has_data(fd, std::min<uint64_t>(size, mib_bytes));
}
#else
inline bool tier_handle_has_data(void* handle, uint64_t end) {
    HANDLE h = (HANDLE) handle;
    LARGE_INTEGER zero{};
    if (end == 0) return false;
    if (!SetFilePointerEx(h, zero, nullptr, FILE_BEGIN)) return true;
    std::vector<uint8_t> buf((size_t) std::min<uint64_t>(end, 1ull << 20));
    for (uint64_t pos = 0; pos < end;) {
        const DWORD want = (DWORD) std::min<uint64_t>(buf.size(), end - pos);
        DWORD got = 0;
        if (!ReadFile(h, buf.data(), want, &got, nullptr)) return true;
        if (got == 0) return false;
        for (DWORD i = 0; i < got; ++i)
            if (buf[i] != 0) return true;
        pos += got;
    }
    return false;
}
inline bool tier_handle_blank_or_tier(void* handle, uint64_t size, uint64_t mib_bytes) {
    HANDLE h = (HANDLE) handle;
    uint8_t head[kTierClaimBytes] = {};
    const DWORD want = (DWORD) std::min<uint64_t>(sizeof head, size);
    DWORD got = 0;
    if (want > 0 && (!SetFilePointerEx(h, LARGE_INTEGER{}, nullptr, FILE_BEGIN) || !ReadFile(h, head, want, &got, nullptr))) return false;
    if (tier_header_magic(head, got)) return true;
    return !tier_handle_has_data(handle, std::min<uint64_t>(size, mib_bytes));
}
#endif

#ifdef _WIN32
/// Windows: the length of an open file.  False with `why` when the handle is not a sized file.
inline bool tier_handle_size(void* handle, uint64_t& size, std::string& why) {
    LARGE_INTEGER li;
    li.QuadPart = 0;
    if (!GetFileSizeEx((HANDLE) handle, &li) || li.QuadPart <= 0) {
        why = "cannot get its size (the tier takes a regular file on Windows; a raw physical drive cannot be mapped)";
        return false;
    }
    size = (uint64_t) li.QuadPart;
    return true;
}
#else
/// POSIX: the capacity of an open regular file or block device.  `is_block` says which.  False with `why` for
/// anything else or an empty/unreadable size.
inline bool tier_fd_size(int fd, uint64_t& size, bool& is_block, std::string& why) {
    is_block = false;
    struct stat st {};
    if (::fstat(fd, &st) != 0) { why = "cannot stat it"; return false; }
    if (S_ISREG(st.st_mode)) {
        if (st.st_size <= 0) { why = "the file is empty"; return false; }
        size = (uint64_t) st.st_size;
        return true;
    }
    if (S_ISBLK(st.st_mode)) {
        is_block = true;
#if defined(__linux__) && defined(BLKGETSIZE64)
        uint64_t bytes = 0;
        if (::ioctl(fd, BLKGETSIZE64, &bytes) != 0 || bytes == 0) {
            why = "cannot get the block device's capacity (ioctl BLKGETSIZE64 failed)";
            return false;
        }
        size = bytes;
        return true;
#else
        why = "block devices are supported on Linux only; give a regular file here";
        return false;
#endif
    }
    why = "it is neither a regular file nor a block device";
    return false;
}

/// POSIX: opens a tier device read-write.  A Linux block device is opened O_EXCL, so the kernel refuses one the system is
/// using (EBUSY).  The opened file must be the kind that was looked at (a path swapped in between is refused).  -1 with `why`.
inline int tier_open_rw(const std::string& path, std::string& why) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) {
        why = "cannot open read-write (errno " + std::to_string(errno) + ": " + std::strerror(errno) + ")";
        return -1;
    }
    int flags = O_RDWR;
#if defined(O_CLOEXEC)
    flags |= O_CLOEXEC;
#endif
#if defined(__linux__)
    if (S_ISBLK(st.st_mode)) flags |= O_EXCL;
#endif
    const int fd = ::open(path.c_str(), flags);
    if (fd < 0) {
        if (errno == EBUSY && S_ISBLK(st.st_mode))
            why = "the block device is in use (mounted, used as swap, part of a RAID or LVM volume, or one of its partitions is): "
                  "Strata never writes over a device the system is using";
        else
            why = "cannot open read-write (errno " + std::to_string(errno) + ": " + std::strerror(errno) + ")";
        return -1;
    }
    struct stat st2 {};
    if (::fstat(fd, &st2) != 0 || S_ISBLK(st.st_mode) != S_ISBLK(st2.st_mode) || st.st_ino != st2.st_ino) {
        ::close(fd);
        why = "it changed while it was being opened";
        return -1;
    }
    return fd;
}
#endif

}  // namespace strata::core
