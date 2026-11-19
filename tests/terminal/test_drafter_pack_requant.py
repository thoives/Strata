"""drafter_pack --requant: which tensors take the quant, and which keep their own type.

The 4-bit-heavy DSpark drafter is `--requant <name> --experts keep --dense Q4_K`: the routed experts stay MXFP4 and the large dense
matrices go to Q4_K. A dense matrix whose rows are not whole Q4_K blocks (256 columns) used to abort the whole run; it now keeps its
type. Norms, gates and heads are never requantized. No libggml is needed: requant_target() is the whole decision.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dsv41"))
import drafter_pack as DP  # noqa: E402

EXPERT = "mtp.0.ffn.experts.3.w1.weight"


class RequantTarget(unittest.TestCase):
    def test_routed_experts_stay_mxfp4_with_keep(self):
        self.assertEqual(DP.requant_target(EXPERT, "MXFP4", [2048, 4096], "keep", "Q4_K"), "MXFP4")

    def test_a_dense_matrix_fitting_q4k_blocks_takes_q4k(self):
        self.assertEqual(DP.requant_target("mtp.0.attn.wq_b.weight", "BF16", [1024, 4096], "keep", "Q4_K"), "Q4_K")

    def test_a_dense_matrix_not_whole_q4k_blocks_keeps_bf16_instead_of_aborting(self):
        self.assertEqual(DP.requant_target("mtp.0.attn.wq_a.weight", "BF16", [1024, 32 * 5], "keep", "Q4_K"), "BF16")

    def test_norms_gates_and_heads_are_never_requantized(self):
        for name in ("mtp.0.attn_norm.weight", "mtp.0.ffn.gate.weight", "mtp.0.confidence_head.proj.weight",
                     "mtp.0.markov_head.embed.weight", "mtp.0.attn_sink"):
            self.assertEqual(DP.requant_target(name, "BF16", [4096, 4096], "keep", "Q4_K"), "BF16", name)

    def test_a_non_bf16_dense_tensor_keeps_its_type(self):
        self.assertEqual(DP.requant_target("mtp.0.attn.wq_b.weight", "F32", [1024, 4096], "keep", "Q4_K"), "F32")

    def test_a_one_dimensional_tensor_keeps_its_type(self):
        self.assertEqual(DP.requant_target("mtp.0.attn.bias", "BF16", [4096], "keep", "Q4_K"), "BF16")

    def test_a_dense_type_equal_to_the_source_is_a_plain_copy(self):
        self.assertEqual(DP.requant_target("mtp.0.attn.wq_b.weight", "BF16", [1024, 4096], "keep", "BF16"), "BF16")

    def test_experts_requantized_only_when_asked(self):
        self.assertEqual(DP.requant_target(EXPERT, "MXFP4", [2048, 4096], "Q4_K", "keep"), "Q4_K")
        self.assertEqual(DP.requant_target(EXPERT, "MXFP4", [2048, 4096], "keep", "keep"), "MXFP4")


if __name__ == "__main__":
    unittest.main()
