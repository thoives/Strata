"""--dspark-expert-cache-mib: in the Terminal's table, the engine's usage, the desktop app's value-flag list, and passed through unchanged.

The engine's own checks (needs --dspark on; a GGUF pack's drafter; no Mac) run in ds4_main.cpp and ds4_model.cu, which need a GPU to
run; this file only pins that the flag is documented and reaches the engine. The desktop filter's npm test is not run here.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from strata_cli.launch import params, settings as S

ROOT = Path(__file__).resolve().parents[2]
FLAG = "--dspark-expert-cache-mib"


class DsparkExpertCacheFlag(unittest.TestCase):
    def test_the_terminal_table_documents_it_as_a_number_of_mib(self):
        meta, desc = params.DSV41_FLAGS[FLAG]
        self.assertEqual(meta, "N")
        self.assertIn("MiB", desc)
        self.assertIn("RAM", desc)

    def test_the_setting_reaches_the_engine_argv_as_typed(self):
        pairs = S.settings_from_pairs(["dspark-expert-cache-mib=512"])
        self.assertEqual(pairs, {FLAG: "512"})
        args, env = S.as_argv_changes(pairs, frozenset())
        self.assertEqual(args, {FLAG: "512"})
        self.assertEqual(env, {})

    def test_a_zero_budget_is_a_value_not_a_removed_flag(self):
        # 0 is the hybrid placement with the drafter's experts all in RAM (the zero-slot arm of the benchmark), so it must reach the engine
        args, _ = S.as_argv_changes({FLAG: "0"}, frozenset())
        self.assertEqual(args[FLAG], "0")

    def test_the_engine_usage_and_the_refusal_are_in_the_source(self):
        src = (ROOT / "src" / "dsv41" / "ds4_main.cpp").read_text(encoding="utf-8")
        usage = src[src.index("void usage()"):src.index("void usage()") + 1600]
        self.assertIn("[--dspark-expert-cache-mib N]", usage)
        self.assertIn("--dspark-expert-cache-mib needs --dspark on", src)

    def test_the_desktop_filter_takes_it_as_a_value_flag(self):
        text = (ROOT / "apps" / "host" / "server" / "native-ds-args.mjs").read_text(encoding="utf-8")
        self.assertIn(f"'{FLAG}'", text)


if __name__ == "__main__":
    unittest.main()
