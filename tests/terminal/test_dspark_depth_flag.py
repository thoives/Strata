"""--dspark-depth: documented in the Terminal's table and the engine's usage, and passed through the launcher unchanged.

The depth is the most guessed words the model checks in one round (ds4_main.cpp keeps at most g_spec_depth drafts for verification).
It limits the checks only; the drafter still runs its whole block, so the help text says that. The desktop app's filter lists the
flag as a value flag (apps/host/server/native-ds-args.mjs); its own test runs under npm in apps/host, which this check does not replace.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from strata_cli.launch import params, settings as S

ROOT = Path(__file__).resolve().parents[2]


class DsparkDepthFlag(unittest.TestCase):
    def test_the_terminal_table_documents_it_with_a_value_and_says_what_it_limits(self):
        meta, desc = params.DSV41_FLAGS["--dspark-depth"]
        self.assertEqual(meta, "N")
        self.assertIn("checks", desc)
        self.assertIn("still guesses", desc)

    def test_the_setting_reaches_the_engine_argv_as_typed(self):
        pairs = S.settings_from_pairs(["dspark-depth=3"])
        self.assertEqual(pairs, {"--dspark-depth": "3"})
        args, env = S.as_argv_changes(pairs, frozenset())
        self.assertEqual(args, {"--dspark-depth": "3"})
        self.assertEqual(env, {})

    def test_it_is_not_a_word_flag_or_a_bare_flag(self):
        # a value is kept as the number; it is not mistaken for on/off, and an empty value is no change (as for every value flag)
        args, _ = S.as_argv_changes({"--dspark-depth": "2"}, frozenset())
        self.assertEqual(args["--dspark-depth"], "2")
        args, _ = S.as_argv_changes({"--dspark-depth": ""}, frozenset())
        self.assertNotIn("--dspark-depth", args)

    def test_the_engine_usage_lists_it(self):
        src = (ROOT / "src" / "dsv41" / "ds4_main.cpp").read_text(encoding="utf-8")
        usage = src[src.index("void usage()"):src.index("void usage()") + 1400]
        self.assertIn("[--dspark-depth N]", usage)

    def test_the_desktop_filter_passes_it_as_a_value_flag(self):
        text = (ROOT / "apps" / "host" / "server" / "native-ds-args.mjs").read_text(encoding="utf-8")
        self.assertIn("'--dspark-depth'", text)


if __name__ == "__main__":
    unittest.main()
