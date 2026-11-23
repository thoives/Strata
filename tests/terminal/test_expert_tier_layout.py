"""The expert tier's layout (docs/EXPERT_TIER.md, "The layout") in the Terminal: the rules the engines follow
(include/strata/core/expert_tier_spec.hpp), the header reader, the capacity estimate, the engine arguments, and the plan.
No GPU, no downloads; the plan reads the devices it is given and writes nothing (checked below)."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import derive_config, tier_layout_pairs
from strata_control import expert_tier as et
from tests.terminal.launch_support import HW_12, Home, run_cli


def header_bytes(version: int, reserved0: int = 0, fill_state: int = 2, magic: bytes = b"STRPTIER") -> bytes:
    """The fixed fields of a Qwen tier header (expert_tier_layout.hpp, 160 bytes); the CRC is not checked by the reader."""
    b = bytearray(160)
    b[0:8] = magic
    struct.pack_into("<I", b, 8, version)
    struct.pack_into("<I", b, 84, reserved0)
    struct.pack_into("<I", b, 152, fill_state)
    return bytes(b)


class LayoutRules(unittest.TestCase):
    def test_the_layout_names_in_any_case(self):
        self.assertEqual(et.parse_layout("Striped"), ("striped", ""))
        self.assertEqual(et.parse_layout("whole"), ("whole", ""))
        layout, why = et.parse_layout("stripes")
        self.assertIsNone(layout)
        self.assertIn("whole or striped", why)

    def test_the_stripe_sizes(self):
        self.assertEqual(et.parse_stripe_kib("256"), (256, ""))
        self.assertEqual(et.parse_stripe_kib("4")[0], 4)
        self.assertEqual(et.parse_stripe_kib("16384")[0], 16384)
        for bad in ("0", "6", "16388", "-4", "1e3", "", "0x100", "2.5"):
            self.assertIsNone(et.parse_stripe_kib(bad)[0], bad)
            self.assertIn("KiB", et.parse_stripe_kib(bad)[1])

    def test_the_layout_options_need_their_devices(self):
        ok = ""
        self.assertEqual(et.layout_problem(None, None, 0), ok)              # nothing given: whole, as before
        self.assertEqual(et.layout_problem(None, None, 2), ok)
        self.assertNotEqual(et.layout_problem("whole", None, 0), ok)        # a layout with no tier does nothing: refused
        self.assertNotEqual(et.layout_problem("striped", None, 0), ok)
        self.assertNotEqual(et.layout_problem("striped", None, 1), ok)      # striped needs two devices
        self.assertEqual(et.layout_problem("striped", None, 2), ok)
        self.assertEqual(et.layout_problem("striped", None, et.MAX_STRIPED_DEVICES), ok)
        self.assertNotEqual(et.layout_problem("striped", None, et.MAX_STRIPED_DEVICES + 1), ok)
        self.assertNotEqual(et.layout_problem(None, 64, 2), ok)             # a stripe size with the whole layout is refused
        self.assertNotEqual(et.layout_problem("whole", 64, 2), ok)
        self.assertEqual(et.layout_problem("striped", 64, 2), ok)

    def test_the_default_stripe_is_256_kib_for_striped_only(self):
        self.assertEqual(et.striped_stripe_kib("striped", None), 256)
        self.assertEqual(et.striped_stripe_kib("striped", 64), 64)
        self.assertIsNone(et.striped_stripe_kib("whole", None))
        self.assertIsNone(et.striped_stripe_kib(None, None))


class CapacityEstimate(unittest.TestCase):
    def test_striped_is_the_smallest_device_times_the_devices(self):
        specs = [et.TierSpec("/a", 40960, 0, False), et.TierSpec("/b", 81920, 0, False)]   # 40 GiB and 80 GiB
        e = et.capacity_estimate(specs, "striped", None)
        self.assertEqual(e["stripe_kib"], 256)
        self.assertAlmostEqual(e["usable_gib"], 80.0)      # 2 x the smallest
        self.assertAlmostEqual(e["unused_gib"], 40.0)      # the larger device's rest
        self.assertEqual(e["per_device_gib"], [40.0, 80.0])

    def test_whole_adds_the_devices_up(self):
        specs = [et.TierSpec("/a", 40960, 4096, False), et.TierSpec("/b", 81920, 0, False)]   # a 4 GiB reserve on the first
        e = et.capacity_estimate(specs, None, None)
        self.assertAlmostEqual(e["usable_gib"], 116.0)
        self.assertEqual(e["unused_gib"], 0.0)


class HeaderReader(unittest.TestCase):
    def write(self, *chunks: bytes) -> str:
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".tier")
        f.write(b"".join(chunks))
        f.close()
        self.addCleanup(Path(f.name).unlink, missing_ok=True)
        return f.name

    def test_a_striped_header_reports_its_stripe_and_state(self):
        path = self.write(header_bytes(3, reserved0=65536, fill_state=2))
        self.assertEqual(et.header_layout(path), {"layout": "striped", "stripe_kib": 64, "complete": True})

    def test_a_whole_header_and_an_unfinished_one(self):
        self.assertEqual(et.header_layout(self.write(header_bytes(2))), {"layout": "whole", "stripe_kib": None, "complete": True})
        self.assertFalse(et.header_layout(self.write(header_bytes(2, fill_state=1)))["complete"])

    def test_header_b_is_read_when_a_does_not_hold_a_tier(self):
        # header A (the first 64 KiB) holds no tier; header B, at 64 KiB, is a striped tier with 4 KiB stripes
        self.assertEqual(et.header_layout(self.write(b"\0" * (64 * 1024), header_bytes(3, reserved0=4096)))["stripe_kib"], 4)

    def test_no_tier_header_gives_none(self):
        self.assertIsNone(et.header_layout(self.write(b"\0" * 300)))          # blank
        self.assertIsNone(et.header_layout(self.write(header_bytes(9))))     # a version this reader does not know
        self.assertIsNone(et.header_layout(self.write(header_bytes(2, magic=b"STRPTDS4"))))   # the DeepSeek engine's tier
        self.assertIsNone(et.header_layout("/nonexistent/tier"))

    def test_the_note_says_reuse_or_refill(self):
        path = self.write(header_bytes(3, reserved0=65536))
        self.assertIn("refills", et.layout_note(path, "whole", None))
        self.assertIn("different stripe size", et.layout_note(path, "striped", 256))
        self.assertIn("reuses it", et.layout_note(path, "striped", 64))
        unfinished = self.write(header_bytes(3, reserved0=65536, fill_state=1))
        self.assertIn("not finished", et.layout_note(unfinished, "striped", 64))


class SettingsAndEngineArgs(unittest.TestCase):
    GEN = ["--native", "x.gguf", "--expert-cache", "auto", "--max-context", "32768"]

    def test_the_whole_layout_is_stored_as_nothing(self):
        self.assertIsNone(S.LaunchSettings.from_dict({"expert_tier_layout": "whole"}).expert_tier_layout)
        self.assertIsNone(S.LaunchSettings.from_dict({}).expert_tier_layout)
        s = S.LaunchSettings.from_dict({"expert_tier_layout": "STRIPED", "expert_tier_stripe_kib": 64})
        self.assertEqual((s.expert_tier_layout, s.expert_tier_stripe_kib), ("striped", 64))
        self.assertIsNone(S.LaunchSettings.from_dict({"expert_tier_stripe_kib": -4}).expert_tier_stripe_kib)

    def test_the_settings_round_trip(self):
        s = S.LaunchSettings(expert_tiers=["/a,mib=4", "/b,mib=8"], expert_tier_layout="striped", expert_tier_stripe_kib=64)
        back = S.LaunchSettings.from_dict(s.to_dict())
        self.assertEqual((back.expert_tier_layout, back.expert_tier_stripe_kib, back.expert_tiers),
                         ("striped", 64, ["/a,mib=4", "/b,mib=8"]))

    def test_the_copyable_line_says_the_devices_and_the_layout(self):
        line = S.flags_text(S.LaunchSettings(expert_tiers=["/a,mib=4", "/b,mib=8"], expert_tier_layout="striped"))
        self.assertIn('--expert-tier "/a,mib=4" --expert-tier "/b,mib=8"', line)
        self.assertIn("--expert-tier-layout striped --expert-tier-stripe-kib 256", line)
        whole = S.flags_text(S.LaunchSettings(expert_tiers=["/a,mib=4", "/b,mib=8"]))
        self.assertNotIn("--expert-tier-layout", whole)                   # the default is not written

    def test_striped_args_follow_the_devices_and_replace_old_ones(self):
        base = {"exe": "component:strata-engine", "port": 8095, "model_name": "m", "env": {},
                "args": [*self.GEN, "--expert-tier-layout", "whole", "--expert-tier", "old.bin,mib=1"],
                "user_args": ["--expert-tier-stripe-kib", "128", "--vram-reserve-mib", "0"]}
        cfg = derive_config(base, S.LaunchSettings(expert_tiers=["/a,mib=4", "/b,mib=8"], expert_tier_layout="striped"),
                            bool_flags=frozenset())
        args = cfg["args"]
        self.assertEqual(args[args.index("--expert-tier-layout") + 1], "striped")
        self.assertEqual(args[args.index("--expert-tier-stripe-kib") + 1], "256")
        self.assertEqual(args.count("--expert-tier-layout"), 1)                  # the old whole flag is gone
        self.assertNotIn("old.bin,mib=1", args)
        self.assertEqual(cfg["user_args"], ["--vram-reserve-mib", "0"])          # the stale layout left user_args too

    def test_a_whole_launch_writes_no_layout_flag_and_keeps_its_args(self):
        base = {"exe": "component:strata-engine", "port": 8095, "model_name": "m", "env": {}, "args": list(self.GEN)}
        cfg = derive_config(base, S.LaunchSettings(), bool_flags=frozenset())
        self.assertEqual(cfg["args"], self.GEN)                                  # byte-identical without a tier
        self.assertEqual(tier_layout_pairs(S.LaunchSettings(expert_tier_layout="striped")), ["--expert-tier-layout", "striped",
                                                                                              "--expert-tier-stripe-kib", "256"])
        self.assertEqual(tier_layout_pairs(S.LaunchSettings()), [])

    def test_a_layout_left_in_the_config_goes_when_the_run_has_no_tier(self):
        base = {"exe": "component:strata-engine", "port": 8095, "model_name": "m", "env": {},
                "args": [*self.GEN, "--expert-tier-layout", "striped", "--expert-tier-stripe-kib", "256"]}
        cfg = derive_config(base, S.LaunchSettings(), bool_flags=frozenset())
        self.assertEqual(cfg["args"], self.GEN)     # the engine would refuse a layout with no device

    def test_the_server_starts_with_the_striped_flags(self):
        from serve.server import engine_args
        cfg = derive_config({"exe": "component:strata-engine", "port": 8095, "model_name": "m", "env": {}, "args": list(self.GEN)},
                            S.LaunchSettings(expert_tiers=["/a,mib=4", "/b,mib=8"], expert_tier_layout="striped",
                                             expert_tier_stripe_kib=64), bool_flags=frozenset())
        argv = engine_args(cfg)
        self.assertEqual(argv[argv.index("--expert-tier-layout") + 1], "striped")
        self.assertEqual(argv[argv.index("--expert-tier-stripe-kib") + 1], "64")


class TheLaunchCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.devs = []
        for name in ("a.tier", "b.tier"):
            p = Path(self.tmp.name) / name
            p.write_bytes(b"\0" * (1 << 20))              # one MiB each: mib=1 fits; nothing is written by a plan
            self.devs += ["--expert-tier", f"{p},mib=1"]

    def plan(self, extra: list[str], family: str = "qwen", model: str = "q2_0"):
        home = Home()
        home.__enter__()
        self.addCleanup(home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)
        return run_cli(["launch", "--plan", "--output", "json", "--family", family, "--model", model, *self.devs, *extra])

    def test_a_striped_plan_shows_the_layout_and_the_estimate(self):
        code, out, err = self.plan(["--expert-tier-layout", "striped"])
        self.assertEqual(code, 0, err or out)
        plan = json.loads(out)["plan"]
        self.assertEqual(plan["settings"]["expert_tier_layout"], "striped")
        self.assertEqual(plan["settings"]["expert_tier_stripe_kib"], 256)
        self.assertEqual(plan["expert_tier"]["layout"], "striped")
        self.assertEqual(len(plan["expert_tier"]["devices"]), 2)
        self.assertIn("usable_gib", plan["expert_tier"]["estimate"])

    def test_the_plan_writes_nothing_to_the_devices(self):
        before = [Path(a.split(",")[0]).read_bytes() for a in self.devs[1::2]]
        self.plan(["--expert-tier-layout", "striped"])
        self.assertEqual([Path(a.split(",")[0]).read_bytes() for a in self.devs[1::2]], before)

    def test_refusals_say_why(self):
        cases = [
            (["--expert-tier-layout", "striped", "--expert-tier-stripe-kib", "6"], "multiple of 4"),
            (["--expert-tier-layout", "stripes"], "whole or striped"),
            (["--expert-tier-stripe-kib", "64"], "for --expert-tier-layout striped"),
        ]
        for extra, want in cases:
            code, out, err = self.plan(extra)
            self.assertNotEqual(code, 0, extra)
            self.assertIn(want, (out + err), extra)

    def test_a_layout_with_no_device_is_refused(self):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--family", "qwen", "--model", "q2_0",
                                  "--expert-tier-layout", "striped"])
        self.assertNotEqual(code, 0)
        self.assertIn("at least one --expert-tier", out + err)

    def test_the_deepseek_plan_accepts_striped(self):
        # the DeepSeek engine has the striped layout: the plan keeps the setting and names no refusal
        code, out, err = self.plan(["--expert-tier-layout", "striped"], family="deepseek41", model="ist-das-2.00bpw")
        self.assertEqual(code, 0, err or out)
        plan = json.loads(out)["plan"]
        self.assertEqual(plan["settings"]["expert_tier_layout"], "striped")
        self.assertNotIn("no striped expert tier", " ".join(plan["warnings"]))

if __name__ == "__main__":
    unittest.main()
