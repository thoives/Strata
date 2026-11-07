"""§12.8 (docs/EXPERT_TIER_PLAN.md): `--expert-tier` reaches the engine args, and a repeated flag keeps its order
across a reconfigure (the user_args mechanism).  No GPU, no downloads, nothing started."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.runner import derive_config
from strata_control import user_args
from strata_cli.launch.catalog import Catalog
from tests.terminal.launch_support import HW_12, Home, run_cli


class EngineArgs(unittest.TestCase):
    """derive_config is what a launch writes over the installed config: the tier flags must land in its args,
    repeated, in order, replacing any tier groups the earlier config had."""

    BASE = {"exe": "component:strata-engine",
            "args": ["--native", "x.gguf", "--expert-cache", "auto", "--expert-tier", "old1.bin,mib=1", "--max-context", "32768",
                     "--expert-tier", "old2.bin,mib=2"],
            "port": 8095, "model_name": "m", "env": {}}

    def test_the_tier_flags_reach_the_args_in_order(self):
        cfg = derive_config(self.BASE, S.LaunchSettings(
            expert_tiers=["/dev/t1,mib=40960", "/dev/t2,mib=81920,verify=full"]), bool_flags=frozenset())
        args = cfg["args"]
        pos = [i for i, x in enumerate(args) if x == "--expert-tier"]
        self.assertEqual([args[i + 1] for i in pos], ["/dev/t1,mib=40960", "/dev/t2,mib=81920,verify=full"])
        self.assertEqual(args.count("--expert-tier"), 2)                 # the config's own old groups were replaced
        self.assertNotIn("old1.bin,mib=1", args)

    def test_without_the_flag_the_args_are_byte_identical(self):
        base = {**self.BASE, "args": ["--native", "x.gguf", "--expert-cache", "auto", "--max-context", "32768"]}
        cfg = derive_config(base, S.LaunchSettings(), bool_flags=frozenset())
        self.assertEqual(cfg["args"], base["args"])                        # no tier in this launch: nothing touched


class SettingsRoundTrip(unittest.TestCase):
    def test_the_preset_file_keeps_the_list_in_order(self):
        s = S.LaunchSettings(expert_tiers=["a.bin,mib=4", "b.bin,mib=8"])
        d = s.to_dict()
        self.assertEqual(d["expert_tiers"], ["a.bin,mib=4", "b.bin,mib=8"])
        self.assertEqual(S.LaunchSettings.from_dict(d).expert_tiers, ["a.bin,mib=4", "b.bin,mib=8"])
        self.assertEqual(S.LaunchSettings.from_dict({}).expert_tiers, [])   # no tier by default

    def test_the_parser_repeats_the_flag_in_order(self):
        with Home() as home:
            p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
            p.start()
            try:
                code, out, err = run_cli(["launch", "--plan", "--output", "json", "--family", "qwen", "--model", "q2_0",
                                          "--expert-tier", "/dev/t1,mib=40960", "--expert-tier", "/dev/t2,mib=81920"])
                self.assertEqual(code, 0, err or out)
                plan = json.loads(out)["plan"]
                self.assertEqual(plan["settings"]["expert_tiers"], ["/dev/t1,mib=40960", "/dev/t2,mib=81920"])
            finally:
                p.stop()


class ReconfigureOrder(unittest.TestCase):
    """§12.8: the person's repeated --expert-tier (hand-edited into args, or kept user_args) survives a setup run
    again: capture keeps every group, merge puts them all back over the new generated args, in order."""

    GEN = ["--native", "x.gguf", "--expert-cache", "auto", "--max-context", "32768"]

    def test_capture_and_merge_keep_both_devices(self):
        old = {"args": [*self.GEN, "--expert-tier", "a.bin,mib=40960", "--expert-tier", "b.bin,mib=81920"],
               "args_written": list(self.GEN)}                  # setup wrote none: the person added both by hand
        mine = user_args.capture(old, frozenset())
        self.assertEqual(mine, ["--expert-tier", "a.bin,mib=40960", "--expert-tier", "b.bin,mib=81920"])
        merged = user_args.merge([*self.GEN, "--expert-tier", "generated.bin,mib=1"], mine)
        pos = [i for i, x in enumerate(merged) if x == "--expert-tier"]
        self.assertEqual([merged[i + 1] for i in pos], ["a.bin,mib=40960", "b.bin,mib=81920"])
        self.assertNotIn("--expert-tier", self.GEN)                       # the person's groups replace the setup's own

    def test_the_user_key_itself_round_trips(self):
        cfg = {"args": list(self.GEN), user_args.USER_KEY: ["--expert-tier", "a,mib=1", "--expert-tier", "b,mib=2"]}
        applied = user_args.apply(cfg)
        self.assertEqual(applied, ["--expert-tier", "--expert-tier"])
        pos = [i for i, x in enumerate(cfg["args"]) if x == "--expert-tier"]
        self.assertEqual([cfg["args"][i + 1] for i in pos], ["a,mib=1", "b,mib=2"])


class ThroughTheServersMerge(unittest.TestCase):
    """The config a launch writes is what serve/server.py reads at start: engine_args() puts "user_args" over "args".  A
    --expert-tier left in the earlier config's user_args must not win over the devices this launch asked for."""

    def cfg(self, **extra):
        return {"exe": "component:strata-engine", "port": 8095, "model_name": "m", "env": {},
                "args": ["--native", "x.gguf", "--expert-cache", "auto", "--expert-tier", "old1.bin,mib=1", "--max-context", "32768"],
                "user_args": ["--vram-reserve-mib", "0", "--expert-tier", "old2.bin,mib=2", "--expert-tier", "old3.bin,mib=3"],
                **extra}

    @staticmethod
    def tiers(args):
        return [args[i + 1] for i, x in enumerate(args) if x == "--expert-tier"]

    def test_requested_devices_win_over_the_configs_user_args(self):
        from serve.server import engine_args
        want = ["/dev/t1,mib=40960", "/dev/t2,mib=81920,verify=full"]
        cfg = derive_config(self.cfg(), S.LaunchSettings(expert_tiers=want), bool_flags=frozenset())
        self.assertEqual(self.tiers(cfg["args"]), want)
        self.assertEqual(self.tiers(cfg["user_args"]) if "user_args" in cfg else [], [])   # the old groups left user_args
        self.assertEqual(self.tiers(engine_args(cfg)), want)                    # what the engine is started with
        self.assertIn("--vram-reserve-mib", engine_args(cfg))                      # the person's other options stay
        self.assertEqual(cfg["user_args"], ["--vram-reserve-mib", "0"])

    def test_user_args_with_only_tier_groups_is_dropped(self):
        from serve.server import engine_args
        cfg = derive_config(self.cfg(user_args=["--expert-tier", "old2.bin,mib=2"]), S.LaunchSettings(expert_tiers=["/dev/t1,mib=9"]),
                            bool_flags=frozenset())
        self.assertNotIn("user_args", cfg)
        self.assertEqual(self.tiers(engine_args(cfg)), ["/dev/t1,mib=9"])

    def test_no_tier_flag_leaves_both_lists_alone(self):
        from serve.server import engine_args
        base = self.cfg()
        cfg = derive_config(base, S.LaunchSettings(), bool_flags=frozenset())
        self.assertEqual(cfg["args"], base["args"])
        self.assertEqual(cfg["user_args"], base["user_args"])
        self.assertEqual(self.tiers(engine_args(cfg)), ["old2.bin,mib=2", "old3.bin,mib=3"])   # the config's own choice, as before


class DeepSeekNeedsAStoreSize(unittest.TestCase):
    """The DeepSeek engine stops at "--expert-tier" with the RAM store at auto: a launch catches that before starting it."""

    def cfg(self, **kw):
        return {"arch": "deepseek41", "args": ["--pack", "p", "--expert-tier", "/dev/t1,mib=4096"], "env": {}, **kw}

    def test_a_tier_with_no_store_size_is_named_before_the_engine_starts(self):
        from strata_cli.launch.runner import dsv41_tier_store_problem
        self.assertIn("RAM store", dsv41_tier_store_problem(self.cfg()))

    def test_every_way_of_giving_the_size_clears_it(self):
        from strata_cli.launch.runner import dsv41_tier_store_problem
        self.assertEqual(dsv41_tier_store_problem(self.cfg(args=[*self.cfg()["args"], "--ssd-cache", "40"])), "")
        self.assertEqual(dsv41_tier_store_problem(self.cfg(user_args=["--ssd-cache", "40"])), "")      # the person's own, merged at the start
        self.assertEqual(dsv41_tier_store_problem(self.cfg(env={"DS4_STORE_GB": "40"})), "")
        # through --set, as the launch applies it
        cfg = derive_config(self.cfg(), S.LaunchSettings(extra={"--ssd-cache": "40"}), bool_flags=frozenset())
        self.assertEqual(dsv41_tier_store_problem(cfg), "")

    def test_other_families_and_tier_free_runs_are_not_touched(self):
        from strata_cli.launch.runner import dsv41_tier_store_problem
        self.assertEqual(dsv41_tier_store_problem(self.cfg(arch="qwen")), "")
        self.assertEqual(dsv41_tier_store_problem({"arch": "deepseek41", "args": ["--pack", "p"], "env": {}}), "")


class ToSetupAndChecked(unittest.TestCase):
    """A launch that installs runs setup.py: its --expert-tier flags reach setup in order (so setup works the DeepSeek store size
    out from them), and the devices are checked - and confirmed when they hold data - before anything is downloaded."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = __import__("pathlib").Path(self.tmp.name)

    def dev(self, name, head=b"", mib=4):
        p = self.d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(head)
            fh.truncate(mib << 20)
        return p

    def runner(self, tiers, **kw):
        from strata_cli.launch.runner import Runner
        from tests.terminal.launch_support import catalog
        s = S.LaunchSettings(family="qwen", model="q2_0", expert_tiers=list(tiers))
        events = []
        r = Runner(s, catalog=catalog(), emit=events.append, **kw)
        r.events = events
        return r

    def test_setup_is_given_every_device_in_order(self):
        with Home():
            a = self.runner(["/t2,mib=8", "/t1,mib=4,verify=full"]).setup_args()
            pos = [i for i, x in enumerate(a) if x == "--expert-tier"]
            self.assertEqual([a[i + 1] for i in pos], ["/t2,mib=8", "/t1,mib=4,verify=full"])
            self.assertNotIn("--expert-tier-overwrite", a)
            a = self.runner(["/t2,mib=8"], tier_overwrite=True).setup_args()
            self.assertIn("--expert-tier-overwrite", a)
            a = self.runner([]).setup_args()
            self.assertNotIn("--expert-tier", a)
            self.assertNotIn("--expert-tier-overwrite", a)

    def test_the_command_line_writes_the_canonical_entry_and_refuses_a_bad_one(self):
        with Home():
            p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
            p.start()
            try:
                code, out, err = run_cli(["launch", "--plan", "--output", "json", "--family", "qwen", "--model", "q2_0",
                                          "--expert-tier", "/dev/t1,MIB=0040960,Verify=FULL"])
                self.assertEqual(code, 0, err or out)
                self.assertEqual(json.loads(out)["plan"]["settings"]["expert_tiers"], ["/dev/t1,mib=40960,verify=full"])
                code, out, err = run_cli(["launch", "--plan", "--family", "qwen", "--model", "q2_0", "--expert-tier", "/dev/t1,mib=-5"])
                self.assertNotEqual(code, 0)
                self.assertIn("mib takes a whole number", out + err)
            finally:
                p.stop()

    def test_a_blank_device_needs_no_question(self):
        with Home():
            dev = self.dev("t.tier")
            asked = []
            r = self.runner([f"{dev},mib=4"], confirm_overwrite=lambda t, x: asked.append(x) or True)
            r.check_expert_tiers()
            self.assertEqual(asked, [])
            self.assertTrue(any("writes over" in e.text for e in r.events))        # the warning is said every time

    def test_data_on_the_device_is_never_written_without_a_yes(self):
        from strata_cli.launch.runner import LaunchError
        with Home():
            dev = self.dev("photos.img", head=b"\0" * 100 + b"my photos")
            for kw in ({}, {"confirm_overwrite": lambda t, x: False}, {"assume_yes": True}):
                r = self.runner([f"{dev},mib=4"], **kw)
                with self.assertRaises(LaunchError) as e:
                    r.check_expert_tiers()
                self.assertIn("--expert-tier-overwrite", e.exception.next)
                self.assertIn(b"my photos", dev.read_bytes()[:200])                  # nothing written
            asked = []
            r = self.runner([f"{dev},mib=4"], confirm_overwrite=lambda t, x: asked.append(x) or True)
            r.check_expert_tiers()
            self.assertEqual(len(asked), 1)
            self.assertIn(b"my photos", dev.read_bytes()[:200])                      # the check itself writes nothing
            r.clear_expert_tiers({"args": []})                                       # after the model's files are known
            self.assertFalse(any(dev.read_bytes()[:4 << 20]))                       # cleared for the mib, after the yes
            dev2 = self.dev("photos2.img", head=b"x")
            r2 = self.runner([f"{dev2},mib=4"], tier_overwrite=True)
            r2.check_expert_tiers()
            r2.clear_expert_tiers({"args": []})
            self.assertFalse(any(dev2.read_bytes()[:4 << 20]))

    def test_a_device_that_was_not_confirmed_is_never_cleared(self):
        from strata_cli.launch.runner import LaunchError
        with Home():
            dev = self.dev("late.img")                              # blank when checked: nothing to confirm
            r = self.runner([f"{dev},mib=4"])
            r.check_expert_tiers()
            with dev.open("r+b") as fh:                              # then someone writes into it (still 4 MiB long)
                fh.seek(100)
                fh.write(b"someone wrote here after the check")
            with self.assertRaises(LaunchError) as e:
                r.clear_expert_tiers({"args": []})
            self.assertIn("Nothing was written", e.exception.text)
            self.assertIn(b"someone wrote here", dev.read_bytes()[100:200])

    def test_a_model_file_named_by_the_final_args_is_refused_and_nothing_is_cleared(self):
        from strata_cli.launch.runner import LaunchError
        with Home():
            dev = self.dev("pack_like.img", head=b"\0" * 10 + b"precious")
            r = self.runner([f"{dev},mib=4"], tier_overwrite=True)
            r.check_expert_tiers()
            with self.assertRaises(LaunchError):
                r.clear_expert_tiers({"args": ["--pack", str(dev)]})         # the installed/carried args name the device as a model file
            self.assertIn(b"precious", dev.read_bytes()[:64])

    def test_two_names_of_one_device_and_a_model_file_are_refused(self):
        import os
        from strata_cli.launch.runner import LaunchError
        with Home():
            dev = self.dev("t.tier")
            os.link(dev, self.d / "t.hard")
            with self.assertRaises(LaunchError) as e:
                self.runner([f"{dev},mib=4", f"{self.d / 't.hard'},mib=4"]).check_expert_tiers()
            self.assertIn("same device", e.exception.text)
            pack = self.d / "pack"
            model = self.dev("pack/experts.bin")
            r = self.runner([f"{model},mib=4"])
            probs = r.expert_tier_problems(["--pack", str(pack)])
            self.assertTrue(any("model's file" in p for p in probs), probs)


class FreshLaunchToEngineArgs(unittest.TestCase):
    """The whole way a fresh launch takes: the launch's setup arguments -> setup's own tier check and its config write (the
    user-args channel) -> the launch's derived config -> serve/server.py engine_args(): the argv the engine starts with."""

    def test_the_engine_gets_the_canonical_devices_in_order(self):
        import argparse
        import tempfile
        from pathlib import Path
        import setup
        from serve.server import engine_args
        from strata_cli.launch.runner import Runner
        from tests.terminal.launch_support import catalog
        with Home(), tempfile.TemporaryDirectory() as d:
            t1, t2 = Path(d) / "t1.tier", Path(d) / "t2.tier"
            for t in (t1, t2):
                with open(t, "wb") as fh:
                    fh.truncate(8 << 20)
            typed = [f"{t2},MIB=8,verify=FULL", f"{t1},mib=4,reserve=1"]
            from strata_control import expert_tier
            canon = [expert_tier.canonical(expert_tier.parse(x)[0]) for x in typed]
            s = S.LaunchSettings(family="qwen", model="q2_0", expert_tiers=canon)
            sargs = Runner(s, catalog=catalog()).setup_args()
            given = [sargs[i + 1] for i, x in enumerate(sargs) if x == "--expert-tier"]
            ns = argparse.Namespace(yes=True, expert_tier_overwrite="--expert-tier-overwrite" in sargs, vram_reserve_mib=None)
            out = __import__("io").StringIO()
            with mock.patch("sys.stdout", out):
                ns._expert_tiers = setup.check_expert_tiers(given, ns)
            self.assertIn("writes over", out.getvalue())
            cfg_path = Path(d) / "strata-q2_0.json"
            generated = {"exe": "component:strata-engine", "port": 8095, "model_name": "m", "env": {},
                         "args": ["--pack", str(Path(d) / "pack"), "--expert-cache", "auto", "--max-context", "32768"]}
            with mock.patch("sys.stdout", out):
                setup.write_setup_config(cfg_path, generated, None, setup.explicit_args(ns))
            written = json.loads(cfg_path.read_text())
            cfg = derive_config(written, s, bool_flags=frozenset())
            argv = engine_args(cfg)
            self.assertEqual([argv[i + 1] for i, x in enumerate(argv) if x == "--expert-tier"], canon)
            self.assertEqual(canon, [f"{t2},mib=8,verify=full", f"{t1},mib=4,reserve=1"])
            # a second launch with no --expert-tier keeps the devices the config holds, in order
            argv2 = engine_args(derive_config(written, S.LaunchSettings(family="qwen", model="q2_0"), bool_flags=frozenset()))
            self.assertEqual([argv2[i + 1] for i, x in enumerate(argv2) if x == "--expert-tier"], canon)


if __name__ == "__main__":
    unittest.main()


class PlanAndTuiWarnings(unittest.TestCase):
    """Item 6: the warnings Runner.plan() makes reach both plan outputs, and the screens carry the overwrite answer."""

    def test_plan_json_and_text_keep_the_tier_warnings(self):
        import tempfile
        from pathlib import Path
        with Home(), tempfile.TemporaryDirectory() as td:
            dev = Path(td) / "photos.img"
            dev.write_bytes(b"\0" * 10 + b"my photos")
            with dev.open("ab") as fh:
                fh.truncate(4 << 20)
            p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
            p.start()
            try:
                args = ["launch", "--plan", "--family", "qwen", "--model", "q2_0", "--expert-tier", f"{dev},mib=4"]
                code, out, err = run_cli([*args[:1], "--output", "json", *args[1:]])
                self.assertEqual(code, 0, err or out)
                warnings = json.loads(out)["plan"]["warnings"]
                self.assertTrue(any("not a Strata expert tier" in w for w in warnings), warnings)
                self.assertTrue(any("first start writes over" in w for w in warnings), warnings)
                code, out, err = run_cli(args)
                self.assertEqual(code, 0, err or out)
                self.assertIn("[!] ", out)
                self.assertIn("not a Strata expert tier", out)
                self.assertIn("Nothing was changed.", out)
                self.assertEqual(dev.read_bytes()[10:19], b"my photos")          # a plan writes nothing
            finally:
                p.stop()

    def test_the_interactive_runner_gets_the_overwrite_answer_and_a_question(self):
        from strata_cli.launch import app as launch_app
        seen = {}

        class FakeRunner:
            def __init__(self, s, **kw):
                seen.update(kw)

            def download_gb(self):
                return 0.0

        ctx = mock.Mock(settings=S.LaunchSettings(family="qwen", model="q2_0", expert_tiers=["a.img,mib=4"]), catalog=None,
                        feed=None, move_data=False, retune=False, restart=False, tier_overwrite=True, pending_save=None)
        self_ = mock.Mock(ctx=ctx, cancel=None)
        self_._ask_overwrite = lambda *a, **k: None
        with mock.patch.object(launch_app, "Runner", FakeRunner), mock.patch.object(launch_app, "RunScreen", mock.Mock()), \
                mock.patch.object(launch_app.threading, "Thread", mock.Mock()):
            launch_app.LaunchApp.run_launch(self_)
        self.assertIs(seen["tier_overwrite"], True)
        self.assertIsNotNone(seen["confirm_overwrite"])

    def test_the_overwrite_question_defaults_to_stop(self):
        from strata_cli.launch import app as launch_app
        asked = {}

        def fake_ask(self_, title, text, options=None, default=True):
            asked.update(title=title, options=options, default=default)
            return default

        import types
        fake = types.SimpleNamespace(_ask_from_run=lambda title, text, options=None, default=True: fake_ask(None, title, text, options, default))
        ok = launch_app.LaunchApp._ask_overwrite(fake, "Overwrite?", "data")
        self.assertIs(ok, False)
        self.assertIs(asked["default"], False)
        self.assertEqual([c for c, _l in asked["options"]], ["go", "no"])


class RelativePathsAcrossFolders(unittest.TestCase):
    """Item 2: a tier path given relative (or with ~) names the same device whatever folder the launch, setup and engine run in."""

    def test_the_runner_makes_the_devices_absolute_before_anything_reads_them(self):
        import os
        import tempfile
        from pathlib import Path
        from tests.terminal.launch_support import catalog
        from strata_cli.launch.runner import Runner
        here = Path.cwd()
        with tempfile.TemporaryDirectory() as td, Home():
            base = Path(td)
            (base / "tiers").mkdir()
            with (base / "tiers" / "rel.tier").open("wb") as fh:
                fh.truncate(4 << 20)
            try:
                os.chdir(base)
                s = S.LaunchSettings(family="qwen", model="q2_0", expert_tiers=["tiers/rel.tier,mib=4", "missing name.tier,mib=8"])
                r = Runner(s, catalog=catalog(), emit=lambda e: None)
                r.absolute_tiers()
                first, second = s.expert_tiers
                self.assertEqual(first, f"{base / 'tiers' / 'rel.tier'},mib=4")
                self.assertEqual(second, f"{base / 'missing name.tier'},mib=8")   # a name with a space stays one name
                os.chdir(base / "tiers")                                       # the folder the engine might run in
                probs = r.expert_tier_problems()                               # the same file is found from there too
                self.assertFalse(any("rel.tier" in p for p in probs), probs)   # (the missing one is reported, as it should be)
                self.assertEqual(len(probs), 1, probs)
                self.assertTrue(Path(first.split(",")[0]).is_file())
            finally:
                os.chdir(here)
