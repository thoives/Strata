"""The expert tier's `--expert-tier PATH,mib=N[,reserve=M][,verify=header|full]` entries, for setup and `strata launch`.

One grammar for everything that reads these entries: both engines (include/strata/core/expert_tier_spec.hpp), setup and the
launch. tests/core/expert_tier_spec_cases.txt holds the cases all of them are checked against. In plain words:

- PATH is everything before the first comma, kept exactly as typed. It may not be empty.
- every other field is key=value. A key is mib, reserve or verify, in any case. A field with no `=`, an empty field (two
  commas in a row, or a comma at the end) or a key given twice is refused.
- mib and reserve are whole numbers of MiB, ASCII digits only. mib is required and at least 1; reserve is less than mib.
- verify is header or full, in any case.

What setup and the launch write into a run config is the canonical form (`canonical`).

The checks made before a device is written (`problems`, `foreign`) are the engines' own (include/strata/core/expert_tier_dev.hpp),
made earlier so that a person hears about them before a download: two entries that are one device (by what the system says it
is, not by its path), a device that is one of the model's files or holds the file system they are on, a Linux block device the
system is using (mounted, swap, RAID or LVM: the kernel refuses an exclusive open), and a device too small for its mib.

Filling a tier writes over the device. A device that is neither blank nor already an expert tier holds someone's data: it is
only written after the person says yes (a question where one can be asked, `--expert-tier-overwrite` where none can). Saying yes
clears the first `mib` MiB the tier may write (`claim`), so the engine then finds it blank; the engine itself refuses a device that still
holds data.
"""

from __future__ import annotations

import contextlib
import errno
import os
import stat
import sys
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple

MAX_MIB = (2 ** 64 - 1) >> 20         # the largest mib whose size in bytes fits 64 bits (the engines' kTierMaxMib)
HEADER_AREA = 64 * 1024               # header A at 0, header B at 64 KiB: both engines
CLAIM_BYTES = 2 * HEADER_AREA         # the bytes a blank device has as zeros, and the bytes `claim` clears
MAGICS = (b"STRPTIER", b"STRPTDS4")   # the Qwen engine's header, the DeepSeek engine's
OVERWRITE_FLAG = "--expert-tier-overwrite"
GRAMMAR = "PATH,mib=N[,reserve=M][,verify=header|full]"
_BLKGETSIZE64 = 0x80081272            # _IOR(0x12, 114, size_t): a Linux block device's capacity, in bytes


class TierSpec(NamedTuple):
    path: str
    mib: int
    reserve_mib: int = 0
    verify_full: bool = False


def _mib(value: str) -> int | None:
    if not value or any(c not in "0123456789" for c in value):
        return None
    n = int(value)
    return n if n <= MAX_MIB else None


def parse(text: object) -> tuple[TierSpec | None, str]:
    """One entry -> (TierSpec, "") or (None, the reason in plain words)."""
    text = str(text)
    c0 = text.find(",")
    if c0 < 0:
        return None, f"expected {GRAMMAR}"
    path = text[:c0]
    if not path:
        return None, "the entry starts with the device path, and it is empty"
    mib = reserve = None
    verify = None
    for part in text[c0 + 1:].split(","):
        if not part:
            return None, f"{path}: an empty field (two commas in a row, or a comma at the end)"
        key, sep, value = part.partition("=")
        if not sep:
            return None, f"{path}: '{part}' is not key=value (mib=N, reserve=M or verify=header|full)"
        k = key.lower() if key.isascii() else key
        if k == "mib":
            if mib is not None:
                return None, f"{path}: mib is given twice"
            mib = _mib(value)
            if mib is None:
                return None, f"{path}: mib takes a whole number of MiB (digits only), not '{value}'"
        elif k == "reserve":
            if reserve is not None:
                return None, f"{path}: reserve is given twice"
            reserve = _mib(value)
            if reserve is None:
                return None, f"{path}: reserve takes a whole number of MiB (digits only), not '{value}'"
        elif k == "verify":
            if verify is not None:
                return None, f"{path}: verify is given twice"
            v = value.lower() if value.isascii() else value
            if v not in ("header", "full"):
                return None, f"{path}: verify takes header or full, not '{value}'"
            verify = v == "full"
        else:
            return None, f"{path}: unknown key '{key}' (mib, reserve or verify)"
    if mib is None:
        return None, f"{path}: mib=N is required (the MiB of the device the tier may use)"
    if mib == 0:
        return None, f"{path}: mib must be at least 1"
    reserve = reserve or 0
    if reserve >= mib:
        return None, f"{path}: reserve=M must be less than mib=N (the rest is what the tier holds)"
    return TierSpec(path, mib, reserve, bool(verify)), ""


def canonical(spec: TierSpec) -> str:
    """The spelling setup and the launch write: PATH,mib=N[,reserve=M][,verify=full]."""
    out = f"{spec.path},mib={spec.mib}"
    if spec.reserve_mib:
        out += f",reserve={spec.reserve_mib}"
    if spec.verify_full:
        out += ",verify=full"
    return out


def parse_all(entries: Iterable[object]) -> tuple[list[TierSpec], list[str]]:
    """Every entry, in order -> (the parsed ones, the refusals as "--expert-tier <entry>: <why>")."""
    specs, bad = [], []
    for raw in entries or []:
        spec, why = parse(raw)
        if spec is None:
            bad.append(f"--expert-tier {raw}: {why}")
        else:
            specs.append(spec)
    return specs, bad


def capacity_gb(specs: Iterable[TierSpec]) -> float:
    """The capacity after every device's reserve, in GiB."""
    return sum(s.mib - s.reserve_mib for s in specs) / 1024.0


# ---- what a device is ---------------------------------------------------------------------------------------------

def _expand(path: str) -> Path:
    return Path(path).expanduser()


def device_size(path: str) -> tuple[str | None, int | str]:
    """("file" | "block", bytes), or (None, the reason). A regular file by its length; a Linux block device by
    ioctl BLKGETSIZE64, as both engines ask. On Windows and macOS only a regular file is a tier device."""
    p = _expand(path)
    try:
        st = p.stat()
    except OSError as e:
        return None, f"cannot be opened read-write ({e})"
    if stat.S_ISREG(st.st_mode):
        return "file", st.st_size
    if stat.S_ISBLK(st.st_mode):
        if not sys.platform.startswith("linux"):
            return None, "a block device works as a tier device on Linux only; give a regular file here"
        import fcntl
        import struct
        try:
            fd = os.open(str(p), os.O_RDONLY)
        except OSError as e:
            return None, f"cannot be opened ({e})"
        try:
            size = struct.unpack("Q", fcntl.ioctl(fd, _BLKGETSIZE64, b"\0" * 8))[0]
        except OSError as e:
            return None, f"its capacity could not be read (ioctl BLKGETSIZE64: {e})"
        finally:
            os.close(fd)
        return "block", size
    return None, "it is neither a regular file nor a block device"


def file_id(path: str) -> tuple | None:
    """What the system says a path IS (symlinks followed): ("block", rdev) or ("file", st_dev, st_ino); None if missing."""
    try:
        st = os.stat(_expand(path))
    except OSError:
        return None
    if stat.S_ISBLK(st.st_mode):
        return ("block", st.st_rdev)
    return ("file", st.st_dev, st.st_ino)


def _whole_disk(dev: int) -> int:
    """Linux: the disk a partition belongs to (sysfs); the device itself otherwise."""
    if not sys.platform.startswith("linux"):
        return dev
    base = Path(f"/sys/dev/block/{os.major(dev)}:{os.minor(dev)}")
    try:
        if not (base / "partition").exists():
            return dev
        ma, mi = (base / ".." / "dev").read_text().strip().split(":")
        return os.makedev(int(ma), int(mi))
    except (OSError, ValueError):
        return dev


def same_device(a: tuple | None, b: tuple | None) -> bool:
    """One device: the same file, two nodes of one block device, or a partition and its disk."""
    if a is None or b is None or a[0] != b[0]:
        return False
    if a[0] == "block":
        return a[1] == b[1] or _whole_disk(a[1]) == b[1] or _whole_disk(b[1]) == a[1]
    return a == b


def holds(tier: tuple | None, f: tuple | None) -> bool:
    """A block device that carries the file system the file `f` lives on."""
    if tier is None or f is None or tier[0] != "block" or f[0] != "file":
        return False
    return f[1] == tier[1] or _whole_disk(f[1]) == tier[1]


def model_files(paths: Iterable[str | os.PathLike | None]) -> list[str]:
    """The files that must never be a tier device: each file given, and every regular file under each folder given."""
    out = []
    for raw in paths or []:
        if not raw:
            continue
        p = _expand(str(raw))
        if p.is_dir():
            for root, _dirs, files in os.walk(p):
                out += [str(Path(root) / f) for f in files]
        elif p.exists():
            out.append(str(p))
    return out


_MODEL_FLAGS = ("--pack", "--native", "--model", "--mtp", "--mmproj", "--expert-profile", "--expert-profile-save", "--model-dir")


def model_paths_from_args(args: Iterable[object]) -> list[str]:
    """The model's own files and folders an engine argument list names (--pack, --native, --model, the profile, ...). A GGUF
    split's sibling shards are named too (name-00001-of-00004.gguf -> every -0000N-of-00004.gguf beside it)."""
    import re
    args = [str(a) for a in args or []]
    out = []
    for i, a in enumerate(args[:-1]):
        if a not in _MODEL_FLAGS:
            continue
        v = args[i + 1]
        if not v or v.startswith("--") or v == "none":
            continue
        out.append(v)
        m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", v)
        if m:
            out += [f"{m.group(1)}-{n:05d}-of-{m.group(3)}.gguf" for n in range(1, int(m.group(3)) + 1)]
    return list(dict.fromkeys(out))


def problems(specs: list[TierSpec], protect: Iterable[str | os.PathLike | None] = (), devices: bool = True) -> list[str]:
    """Everything that stops these devices from being used, in plain words (empty: none). Nothing is written.
    `protect`: the model's files and folders (model_paths_from_args, --gguf-dir); a tier may not be one of them.
    `devices` False: only the identities (aliases, model files), not the size and use of each device."""
    out = []
    ids = []
    for i, s in enumerate(specs):
        if not devices:
            ids.append(file_id(s.path))
            continue
        kind, size = device_size(s.path)
        if kind is None:
            out.append(f"{s.path}: {size}")
            ids.append(None)
            continue
        if size < s.mib * 1024 * 1024:
            out.append(f"{s.path}: the device is {size / 1073741824:.2f} GiB, smaller than mib={s.mib} MiB - Strata does not "
                       "silently use less")
        if kind == "block" and not os.access(str(_expand(s.path)), os.W_OK):
            out.append(f"{s.path}: the block device is not writable by this user (the tier is written to it once)")
        try:                                      # held elsewhere, or not openable read-write: said now, nothing is written here
            with device_lock(s.path):
                pass
        except DeviceBusy as e:
            out.append(f"{s.path}: {e}")
        except OSError as e:
            out.append(f"{s.path}: cannot be opened read-write ({e.strerror or e})")
        ids.append(file_id(s.path))
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            if specs[i].path == specs[j].path or same_device(ids[i], ids[j]):
                out.append(f"{specs[i].path} and {specs[j].path} are the same device (a link, another name for the same file, "
                           "or a partition of the other); give each device once")
    files = model_files(protect)
    for i, s in enumerate(specs):
        for f in files:
            fid = file_id(f)
            if same_device(ids[i], fid):
                out.append(f"{s.path}: it is the model's file {f} (the same file under another name), or lies in the model's "
                           "folder: filling the tier would overwrite it. Put the tier outside the model's folder")
                break
            if holds(ids[i], fid):
                out.append(f"{s.path}: the model's file {f} is on this block device: filling the tier would overwrite it")
                break
    return out


def _read_nonzero(fd: int, start: int, end: int) -> bool:
    """True when any byte in [start, end) is not zero (read in 1 MiB pieces; the caller holds the device)."""
    pos = start
    while pos < end:
        os.lseek(fd, pos, os.SEEK_SET)
        chunk = os.read(fd, min(1 << 20, end - pos))
        if not chunk:
            return False                     # the device ends first: nothing more to hold data
        if any(chunk):
            return True
        pos += len(chunk)
    return False


def has_data(fd: int, end: int) -> bool:
    """True when any byte in [0, end) is not zero. Where the system says where its data is (SEEK_DATA / SEEK_HOLE, a sparse
    file), only those extents are read; elsewhere, and for a block device, every byte is read. The engines' own scan
    (expert_tier_layout.hpp, tier_blank_or_tier) answers the same question."""
    if end <= 0:
        return False
    if hasattr(os, "SEEK_DATA") and hasattr(os, "SEEK_HOLE"):
        pos = 0
        while pos < end:
            try:
                data = os.lseek(fd, pos, os.SEEK_DATA)
            except OSError as e:
                if e.errno == errno.ENXIO:
                    return False             # no data at or after pos
                break                        # not answered here: read it all below
            if data >= end:
                return False
            try:
                hole = min(os.lseek(fd, data, os.SEEK_HOLE), end)
            except OSError:
                hole = end
            if hole <= data:
                hole = end
            if _read_nonzero(fd, data, hole):
                return True
            pos = hole
        else:
            return False
    return _read_nonzero(fd, 0, end)


def blank_or_tier(fd: int, size: int, mib: int) -> bool:
    """True when a fill may write over the device behind `fd`: header A or B is either engine's tier header, or every byte the
    fill may write (the first `mib` MiB, not only the header areas: a sparse file may hold data past its first 128 KiB) is zero.
    False when it holds anything else (someone's data)."""
    head = os.pread(fd, CLAIM_BYTES, 0) if hasattr(os, "pread") else _head(fd)
    for off in (0, HEADER_AREA):
        if head[off:off + 8] in MAGICS:
            return True
    return not has_data(fd, min(size, mib * 1024 * 1024))


def _head(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    return os.read(fd, CLAIM_BYTES)


def foreign(specs: list[TierSpec]) -> list[str]:
    """The devices that hold data that is not an expert tier: a fill would overwrite it. A device another process holds
    is not judged here (problems() reports it as in use)."""
    out = []
    for s in specs:
        try:
            with device_lock(s.path) as (fd, size):
                if not blank_or_tier(fd, size, s.mib):
                    out.append(s.path)
        except OSError:                              # busy, missing or unreadable: problems() says which, and why
            continue
    return out


def overwrite_warning(specs: list[TierSpec]) -> str:
    """The one line said whenever devices are configured: a tier fill writes over them."""
    names = ", ".join(s.path for s in specs)
    return (f"expert tier: the first start writes over {'this device' if len(specs) == 1 else 'these devices'} ({names}), "
            "up to the mib given; whatever was stored there is gone")


def foreign_message(paths: list[str]) -> str:
    return (f"{', '.join(paths)} {'holds' if len(paths) == 1 else 'hold'} data that is not a Strata expert tier, and the "
            "first start would overwrite it")


def absolute(spec: TierSpec) -> TierSpec:
    """The entry with its path made absolute from THIS process's working directory (`~` expanded). Every later check, the
    setup and the engine get this one path, so a relative path means the same file in each of them."""
    return spec._replace(path=os.path.abspath(os.path.expanduser(spec.path)))


def absolute_args(args: Iterable[object]) -> list[str]:
    """An engine argument list with each `--expert-tier` value made absolute (an entry that does not parse stays as typed: the
    checks report it)."""
    out = list(args or [])
    for i in range(len(out) - 1):
        if out[i] == "--expert-tier":
            spec, _why = parse(str(out[i + 1]))
            if spec is not None:
                out[i + 1] = canonical(absolute(spec))
    return out


class DeviceBusy(OSError):
    """Another process holds the device (a running engine, another setup or launch): it is not touched."""


BUSY_FILE = ("another process holds this expert tier (a running Strata engine, setup or launch): one tier is used by one process "
             "at a time")
BUSY_BLOCK = ("the block device is in use (mounted, used as swap, part of a RAID or LVM volume, or one of its partitions is): "
              "Strata never writes over a device the system is using")


@contextlib.contextmanager
def device_lock(path: str) -> Iterator[tuple[int, int]]:
    """Opens a tier device read-write and holds it for the block: yields (fd, size in bytes). A regular file is locked
    exclusively (flock; on Windows a byte range past the end, LockFileEx's way): the engine takes the same lock when it opens the
    tier and keeps it until it closes, so a second process cannot claim or fill the tier meanwhile. A Linux block device is
    opened O_EXCL, which the kernel refuses while the system uses it or another exclusive open holds it. Raises DeviceBusy when
    it is held elsewhere, and the OSError of the open otherwise. The lock is released on every exit, the error ones too."""
    p = str(_expand(path))
    st = os.stat(p)                                   # FileNotFoundError: the caller's problem to say
    block = stat.S_ISBLK(st.st_mode)
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    if block and sys.platform.startswith("linux"):
        flags |= os.O_EXCL
    try:
        fd = os.open(p, flags)
    except OSError as e:
        if e.errno == errno.EBUSY and block:
            raise DeviceBusy(BUSY_BLOCK) from None
        raise
    try:
        size = os.fstat(fd).st_size if stat.S_ISREG(st.st_mode) else device_size(path)[1]
        if not isinstance(size, int):
            raise OSError(errno.EINVAL, f"{path}: its capacity could not be read ({size})")
        if not block:
            try:
                if os.name == "nt":
                    import msvcrt
                    os.lseek(fd, size, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise DeviceBusy(BUSY_FILE) from None
        yield fd, size
    finally:
        os.close(fd)                                  # (releases the flock and the exclusive open)


def _zero_range(fd: int, start: int, end: int) -> None:
    """Makes [start, end) zero. Linux regular files: a hole punch (quick, and the engine's scan then finds no data); else zeros
    written in 8 MiB pieces and flushed."""
    if sys.platform.startswith("linux") and end > start:
        try:
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            fallocate = libc.fallocate
            fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64]
            fallocate.restype = ctypes.c_int
            if fallocate(fd, 0x02 | 0x01, start, end - start) == 0:       # FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE
                return
        except Exception:                             # noqa: BLE001 - not a file system with holes: write zeros instead
            pass
    os.lseek(fd, start, os.SEEK_SET)
    left = end - start
    view = memoryview(b"\0" * min(8 << 20, left)) if left > 0 else memoryview(b"")
    while left > 0:
        n = min(len(view), left)
        left -= os.write(fd, view[:n])
    os.fsync(fd)


def claim(path: str, mib: int) -> None:
    """After the person said yes: makes the first `mib` MiB of the device zero, exactly what the fill may write over, so the
    engine finds it blank (the engine's own check covers the same range). Holds the device's lock while it does. Raises OSError,
    DeviceBusy."""
    with device_lock(path) as (fd, size):
        end = min(size, mib * 1024 * 1024)
        _zero_range(fd, 0, end)
