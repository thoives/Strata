"""pack_gguf's "whole" check: a metadata-only GGUF (no tensors, such as Unsloth's UD-Q8_K_XL shard 1) ends right after its header.
The packer's data start is padded to the alignment, but a file with no tensor data has no padding after its header, so the check must
measure against the header's end. Before this, the official shard 1 (5,780,589 bytes) was reported as "5780589 of 5780608 bytes" and
refused as not whole. A file whose tensors really are cut short is still refused."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dsv41"))
import pack_gguf as PG  # noqa: E402
from tests.terminal.test_pack_gguf_engram import tiny_gguf  # noqa: E402
import struct  # noqa: E402


def header_only(path: Path, kv: dict) -> Path:
    """A metadata-only GGUF as the official shard 1 is: the header, and the file ends there (no padding to the alignment)."""
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)))
    for k, v in kv.items():
        out += struct.pack("<Q", len(k)) + k.encode() + struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v.encode()
    path.write_bytes(bytes(out))
    return path


class Wholeness(unittest.TestCase):
    def test_a_metadata_only_file_is_whole_at_its_header_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            # a header that does not end on the 32-byte alignment, no tensors: the file ends where the header does
            for n in range(30, 60):
                p = header_only(Path(tmp) / f"shard1-{n}.gguf", {"general.architecture": "deepseek41", "deepseek41.note": "x" * n})
                if p.stat().st_size % 32:
                    break
            g = PG.Gguf(p)
            self.assertEqual(g.tensors, [])
            self.assertNotEqual(g.data_start, p.stat().st_size, "the fixture must be unaligned for this test to mean anything")
            self.assertIsNone(PG.short(g))

    def test_a_file_whose_tensor_data_is_cut_short_is_still_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = tiny_gguf(Path(tmp) / "cut.gguf", "deepseek41", {}, tensors=("blk.0.ffn_gate_exps.weight",))  # (its tensor data is missing)
            g = PG.Gguf(p)
            self.assertIsNotNone(PG.short(g))


if __name__ == "__main__":
    unittest.main()
