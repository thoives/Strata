"""The DSpark drafter's geometry travels from DeepSeek's config.json to the pack: drafter_pack.py writes it into dspark.json, and
pack_gguf.py puts it in the GGUF's kv the way a dflash GGUF states it (block size, target layers, noise token, markov rank, expert
counts). Without it the engine loads no drafter, silently.

The fixture holds the text_config values of DeepSeek-V4.1-Flash-1017's config.json. The conversion test writes stand-in safetensors
shards (the real ones are 8 GB) with the same tensor names and dtypes, and runs drafter_pack.main() on them.
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dsv41"))
import drafter_pack as DP  # noqa: E402
import pack_gguf as PG  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "dsv41" / "deepseek-v41-dspark-config.json"
EXPECTED = {"block_size": 5, "noise_token_id": 128799, "markov_rank": 256, "expert_count": 128, "expert_used_count": 3,
            "target_layers": [37, 38, 39]}


def stand_in_shards(folder: Path) -> None:
    """Three safetensors files: the first holds a drafter block's FP8 matrix (with its scale), an FP4 expert (I8, with its scale) and a
    BF16 norm; the other two hold nothing, as the stand-in needs no more."""
    rng = np.random.default_rng(1)
    tensors = [("mtp.0.attn.wq_a.weight", "F8_E4M3", [32, 32], rng.integers(0, 0x7E, 32 * 32, dtype=np.uint8).tobytes()),
               ("mtp.0.attn.wq_a.scale", "U8", [1, 1], bytes([120])),
               ("mtp.0.ffn.experts.0.w1.weight", "I8", [2, 32], rng.integers(0, 255, 64, dtype=np.uint8).tobytes()),
               ("mtp.0.ffn.experts.0.w1.scale", "U8", [2, 2], bytes([120, 121, 122, 123])),
               ("mtp.0.norm.weight", "BF16", [8], rng.integers(0, 255, 16, dtype=np.uint8).tobytes())]
    head, data, off = {}, b"", 0
    for name, dt, shape, raw in tensors:
        head[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
        data += raw
        off += len(raw)
    blob = json.dumps(head).encode()
    (folder / "model-00044-of-00048.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + data)
    for i in (45, 46):
        blob = b"{}"
        (folder / f"model-000{i}-of-00048.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)


class DrafterGeometry(unittest.TestCase):
    def test_the_checkpoint_config_gives_the_drafters_geometry(self):
        config = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(DP.drafter_geometry(config), EXPECTED)

    def test_a_config_without_one_of_them_is_a_clear_error(self):
        config = json.loads(FIXTURE.read_text(encoding="utf-8"))
        del config["text_config"]["dspark_markov_rank"]
        with self.assertRaises(SystemExit) as ctx:
            DP.drafter_geometry(config)
        self.assertIn("dspark_markov_rank", str(ctx.exception))

    def test_the_packer_writes_it_as_the_gguf_states_it_and_the_engine_reads_it(self):
        kv = {"general.architecture": "deepseek41"}
        PG.apply_drafter_geometry(kv, EXPECTED)
        self.assertEqual(kv["deepseek41.dspark.block_size"], 5)
        self.assertEqual(kv["deepseek41.dspark.target_layers"], [37, 38, 39])
        geo = PG.geometry(kv, 129280)   # the packer's own reader: what ds4.txt gets
        self.assertEqual(geo["dspark_block_size"], 5)
        self.assertEqual(geo["dspark_noise_token_id"], 128799)
        self.assertEqual(geo["dspark_markov_rank"], 256)
        self.assertEqual(geo["dspark_n_routed_experts"], 128)
        self.assertEqual(geo["dspark_num_experts_per_tok"], 3)
        self.assertEqual(geo["dspark_target_layer_ids"], [37, 38, 39])

    def test_a_value_the_gguf_already_states_must_agree(self):
        kv = {"general.architecture": "deepseek41", "deepseek41.dspark.block_size": 4}
        with self.assertRaises(SystemExit) as ctx:
            PG.apply_drafter_geometry(kv, EXPECTED)
        self.assertIn("disagrees", str(ctx.exception))
        PG.apply_drafter_geometry({"general.architecture": "deepseek41", "deepseek41.dspark.block_size": 5}, EXPECTED)

    def test_the_conversion_writes_the_geometry_into_dspark_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            stand_in_shards(folder)
            old = sys.argv
            try:
                sys.argv = ["drafter_pack.py", "--dir", str(folder), "--config", str(FIXTURE)]
                self.assertEqual(DP.main(), 0)
            finally:
                sys.argv = old
            out = json.loads((folder / "dspark.json").read_text())
            self.assertEqual(out["geometry"], EXPECTED)
            self.assertEqual(len(out["tensors"]), 3)

    def test_without_a_config_the_json_has_no_geometry_and_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            stand_in_shards(folder)
            old = sys.argv
            try:
                sys.argv = ["drafter_pack.py", "--dir", str(folder)]
                self.assertEqual(DP.main(), 0)
            finally:
                sys.argv = old
            self.assertNotIn("geometry", json.loads((folder / "dspark.json").read_text()))


if __name__ == "__main__":
    unittest.main()
