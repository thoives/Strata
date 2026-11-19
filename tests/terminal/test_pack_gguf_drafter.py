"""pack_gguf picks ONE DSpark drafter source, and an explicit choice is never silently overridden by a discovered file.

Before this, a dspark-*.gguf beside the model always won, so `--drafter` (drafter_pack.py's dspark.json) was ignored with no
message. The decision is dspark_gguf_choice(); these cases pin it: discovery stays the default when no choice is given.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dsv41"))
import pack_gguf as PG  # noqa: E402

FOUND = Path("/models/dspark-q8.gguf")


class DrafterChoice(unittest.TestCase):
    def test_discovery_is_the_default(self):
        self.assertEqual(PG.dspark_gguf_choice(None, None, FOUND), FOUND)

    def test_nothing_found_and_nothing_named_means_no_gguf(self):
        self.assertIsNone(PG.dspark_gguf_choice(None, None, None))

    def test_an_explicit_drafter_json_beats_a_discovered_gguf(self):
        self.assertIsNone(PG.dspark_gguf_choice(None, "/d/dspark.json", FOUND))

    def test_an_explicit_gguf_is_used_and_resolved(self):
        got = PG.dspark_gguf_choice("~/other/dspark-x.gguf", None, FOUND)
        self.assertEqual(got, Path("~/other/dspark-x.gguf").expanduser().resolve())

    def test_naming_both_is_an_error_not_a_silent_pick(self):
        with self.assertRaises(ValueError):
            PG.dspark_gguf_choice("/d/a.gguf", "/d/dspark.json", FOUND)

    def test_naming_both_with_nothing_discovered_is_still_an_error(self):
        with self.assertRaises(ValueError):
            PG.dspark_gguf_choice("/d/a.gguf", "/d/dspark.json", None)


if __name__ == "__main__":
    unittest.main()
