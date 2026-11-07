"""Tests for setup.py's low-RAM messages and the paging question (#250): the one-GPU warning says why and shows the
RAM math, and an explicit --low-ram off installs with --yes instead of stopping at "Install it anyway?" (default
n).  Mocked input - no GPU, no downloads.

    python -m unittest tools.test_setup_lowram
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def run(fn, *args, stdin=None):
    """-> (exit code or None, printed text); `stdin` answers input() (None: never asked)."""
    out = io.StringIO()
    asked = []

    def fake_input(prompt=""):
        asked.append(prompt)
        if stdin is None:
            raise AssertionError(f"asked {prompt!r} with --yes")
        return stdin

    code = None
    with contextlib.redirect_stdout(out), mock.patch("builtins.input", fake_input):
        try:
            fn(*args)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), asked


class OneGpuWhy(unittest.TestCase):
    def test_auto_shows_the_ram_math(self):
        lines = setup.low_ram_one_gpu_why("IQ3_XXS", 46, "auto")       # the #250 report: 46 GB, IQ3_XXS
        text = "\n".join(lines)
        arena = setup.MODELS["IQ3_XXS"]["arena_gb"]
        need = arena + setup.LOW_RAM_HEADROOM_GB
        self.assertIn(f"{arena:.0f} + {setup.LOW_RAM_HEADROOM_GB} = {need:.0f} GB", text)
        self.assertIn("this PC has 46 GB", text)
        self.assertIn("layer split", text)                            # why one GPU is recommended
        self.assertIn("--gpus 0,1", text)                             # #364 #384: and the way to use them all
        self.assertIn("OS file cache", text)
        self.assertNotIn("--low-ram off", text)                       # (that pages the experts: not the way)

    def test_an_explicit_choice_says_so(self):
        text = "\n".join(setup.low_ram_one_gpu_why("Q2_0", 64, "resident"))
        self.assertIn("you chose the low-RAM mode (--low-ram resident)", text)


class PagingQuestion(unittest.TestCase):
    MODEL, RAM = "IQ3_XXS", 46                                       # 60 GB wanted, 46 here

    def test_yes_with_low_ram_off_installs(self):
        code, out, asked = run(setup.confirm_paging, self.MODEL, self.RAM, "off", True)
        self.assertIsNone(code, out)
        self.assertEqual(asked, [])
        self.assertIn("installing IQ3_XXS with 46 GB of RAM, as you chose (--low-ram off)", out)

    def test_yes_with_auto_still_stops(self):
        code, out, asked = run(setup.confirm_paging, self.MODEL, self.RAM, "auto", True)
        self.assertEqual(code, 1)
        self.assertIn("IQ3_XXS needs about 60 GB of RAM; this PC has 46 GB", out)
        self.assertIn("--low-ram off --yes", out)                    # the way to insist

    def test_interactive(self):
        for choice, answer, want in (("off", "", None), ("auto", "", 1), ("auto", "y", None), ("off", "n", 1)):
            with self.subTest(choice=choice, answer=answer):
                code, out, asked = run(setup.confirm_paging, self.MODEL, self.RAM, choice, False, stdin=answer)
                self.assertEqual(code, want, out)
                self.assertEqual(len(asked), 1)
                self.assertIn("[y]" if choice == "off" else "[n]", asked[0])


class DraftLayerCountsInTheFit(unittest.TestCase):
    """The draft layer (MTP) takes about 1 GB of graphics memory out of the expert cache: every fit check counts it while it is on."""

    def test_the_card_holds_one_gb_less_with_mtp_on(self):
        on = setup.low_ram_gpu_gb("IQ3_XXS", 12.0)
        off = setup.low_ram_gpu_gb("IQ3_XXS", 12.0, mtp=False)
        self.assertAlmostEqual(off - on, setup.MTP_VRAM_GB)
        self.assertEqual(setup.MTP_VRAM_GB, 1.0)
        self.assertAlmostEqual(setup.low_ram_gpu_share("IQ3_XXS", 12.0, mtp=False) - setup.low_ram_gpu_share("IQ3_XXS", 12.0),
                               1.0 / setup.MODELS["IQ3_XXS"]["arena_gb"])

    def test_the_edge_of_what_fits_moves_by_the_draft_layer(self):
        arena = setup.MODELS["IQ1_M"]["arena_gb"]
        ram = arena + 6 - 7.0 + 0.5            # a card of 12 GB holds 7 GB beside the dense part; the draft layer needs one of them
        self.assertFalse(setup.low_ram_fits("IQ1_M", ram, 12.0))
        self.assertTrue(setup.low_ram_fits("IQ1_M", ram, 12.0, mtp=False))


class ExpertTierFit(unittest.TestCase):
    """§12.7 (docs/EXPERT_TIER_PLAN.md §9.1/§9.2/§9.4): the fit rules with tier_gb, the parser and the device check.
    Defaults unchanged: every old call answers exactly as before."""

    def test_defaults_answer_as_before(self):
        for m in ("IQ3_XXS", "IQ1_M", "UD-Q4_K_XL"):
            for ram in (31.9, 46.0, 63.7, 127.8):
                for vram in (8.0, 11.9, 23.5):
                    self.assertEqual(setup.low_ram_fits(m, ram, vram),
                                     setup.low_ram_fits(m, ram, vram, tier_gb=0.0))

    def test_defaults_match_the_expression_before_the_tier(self):
        # the pre-tier body of low_ram_fits, written out here: the same inputs must give the same answers
        def before(model, ram, vram_gb, mtp=True):
            return ram - 6 + max(0.0, vram_gb - 5 - (setup.MTP_VRAM_GB if mtp else 0.0)) >= setup.MODELS[model]["arena_gb"]
        for m in setup.MODELS:
            for ram in (0.0, 4.0, 15.9, 31.9, 46.0, 63.7, 127.8):
                for vram in (0.0, 5.5, 6.0, 8.0, 11.9, 23.5, 48.0, 96.0):
                    for mtp in (True, False):
                        got = setup.low_ram_fits(m, ram, vram, mtp=mtp)
                        self.assertIsInstance(got, bool)
                        self.assertEqual(got, before(m, ram, vram, mtp), (m, ram, vram, mtp))

    def test_tier_fit_edge_moves_by_exactly_the_tier(self):
        arena = setup.MODELS["IQ3_XXS"]["arena_gb"]
        gpu = max(0.0, 12.0 - 5 - setup.MTP_VRAM_GB)           # what a 12 GB card holds beside the dense part, MTP on
        edge = arena - gpu + 6                                    # the RAM at which the old rule just fits
        self.assertTrue(setup.low_ram_fits("IQ3_XXS", edge, 12.0))
        self.assertFalse(setup.low_ram_fits("IQ3_XXS", edge - 0.01, 12.0))
        self.assertTrue(setup.low_ram_fits("IQ3_XXS", edge - 2.0, 12.0, tier_gb=2.0))
        self.assertFalse(setup.low_ram_fits("IQ3_XXS", edge - 2.01, 12.0, tier_gb=2.0))
        self.assertIsInstance(setup.low_ram_fits("IQ3_XXS", edge, 12.0, tier_gb=2.0), bool)

    def test_a_big_card_on_a_small_ram_keeps_its_old_answer(self):
        # Compatibility only, not advice: the rule has no floor on the RAM, so a 96 GB card makes "Q2_0 fits" on a 4 GB PC.
        # This test pins that old answer so a change to it is deliberate; such a PC is not a recommended setup.
        self.assertTrue(setup.low_ram_fits("Q2_0", 4.0, 96.0))
        self.assertFalse(setup.low_ram_fits("Q2_0", 4.0, 12.0))

    def test_a_tier_makes_a_tight_pc_fit(self):
        arena = setup.MODELS["IQ3_XXS"]["arena_gb"]
        gpu = setup.low_ram_gpu_gb("IQ3_XXS", 12.0)
        ram = arena - 1.5                                     # 1.5 GB short of what the RAM alone has to hold
        self.assertFalse(setup.low_ram_fits("IQ3_XXS", ram, 12.0))
        self.assertTrue(setup.low_ram_fits("IQ3_XXS", ram, 12.0, tier_gb=2.0))   # the tier carries the shortfall
        # resident (--resident-experts) copies EVERY expert no card holds into RAM, whatever a tier holds: a tier never makes
        # that copy fit, so the resident decision has no tier term at all (the engine has no budget a tier could shrink)
        rest = arena - gpu
        self.assertEqual(setup.low_ram_resident("IQ3_XXS", ram, 12.0), ram >= rest + setup.LOW_RAM_HEADROOM_GB)
        import inspect
        self.assertNotIn("tier_gb", inspect.signature(setup.low_ram_resident).parameters)

    def test_the_recommendation_number(self):
        # §9.2: needed = arena - GPU share - RAM copy budget; never negative
        arena = setup.MODELS["IQ3_XXS"]["arena_gb"]
        gpu = setup.low_ram_gpu_gb("IQ3_XXS", 12.0)
        self.assertAlmostEqual(setup.low_ram_tier_needed("IQ3_XXS", 46.0, 12.0), max(0.0, arena - gpu - 40.0))
        self.assertEqual(setup.low_ram_tier_needed("IQ3_XXS", 512.0, 12.0), 0.0)   # plenty of RAM: nothing needed

    def test_the_parser(self):
        good = setup.parse_expert_tier("/dev/disk10/tier.bin,mib=40960")
        self.assertEqual(good[0], ("/dev/disk10/tier.bin", 40960, 0, False))
        self.assertEqual(setup.parse_expert_tier("a.bin,mib=100,reserve=4,verify=full")[0], ("a.bin", 100, 4, True))
        self.assertIsNone(setup.parse_expert_tier(",mib=10")[0])
        self.assertIsNone(setup.parse_expert_tier("a.bin")[0])            # mib is required
        self.assertIsNone(setup.parse_expert_tier("a.bin,mib=10,verify=half")[0])
        self.assertIsNone(setup.parse_expert_tier("a.bin,mib=10,reserve=20")[0])   # reserve must stay under mib

    def test_the_device_check(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tiny = Path(d) / "tiny.bin"
            tiny.write_bytes(b"" * 1024)                  # 1 KiB
            ok, why, note = setup.expert_tier_device_check(str(tiny), 64)
            self.assertFalse(ok)
            self.assertIn("smaller than mib=64 MiB", why)
            ok, why, _ = setup.expert_tier_device_check(str(Path(d) / "nope.bin"), 64)
            self.assertFalse(ok)
            self.assertIn("cannot be opened", why)
            big = Path(d) / "big.bin"
            big.write_bytes(b"\x01" * 160 + b"\0" * (64 * 1024 * 1024 - 160))   # sparse, not all zero
            ok, why, note = setup.expert_tier_device_check(str(big), 64)
            self.assertTrue(ok)
            self.assertIn("no tier header yet", note)
            import struct
            head = bytearray(160)
            head[0:8] = b"STRPTIER"
            struct.pack_into("<I", head, 8, 2)             # layout version 2: the crc covers fill_state too
            struct.pack_into("<I", head, 152, 1)            # fill_state 1: a fill that did not finish
            struct.pack_into("<I", head, 156, (0xFFFFFFFF ^ setup._tier_crc32c(bytes(head[:156]), 0xFFFFFFFF)) & 0xFFFFFFFF)
            big.write_bytes(bytes(head) + b"\0" * (64 * 1024 * 1024 - 160))
            ok, why, note = setup.expert_tier_device_check(str(big), 64)
            self.assertTrue(ok)
            self.assertIn("fill_state 1", note)
            self.assertIn("-> refill", note)
            # a version 1 header (its crc over 152 bytes, fill_state left out) is not read: it refills once
            struct.pack_into("<I", head, 8, 1)
            struct.pack_into("<I", head, 152, 2)
            struct.pack_into("<I", head, 156, (0xFFFFFFFF ^ setup._tier_crc32c(bytes(head[:152]), 0xFFFFFFFF)) & 0xFFFFFFFF)
            big.write_bytes(bytes(head) + b"\0" * (64 * 1024 * 1024 - 160))
            ok, why, note = setup.expert_tier_device_check(str(big), 64)
            self.assertIn("layout version 1", note)
            self.assertIn("-> refill", note)
            # a version 2 header whose fill_state was flipped to 2 without its crc: the checksum says so
            struct.pack_into("<I", head, 8, 2)
            struct.pack_into("<I", head, 152, 1)
            struct.pack_into("<I", head, 156, (0xFFFFFFFF ^ setup._tier_crc32c(bytes(head[:156]), 0xFFFFFFFF)) & 0xFFFFFFFF)
            struct.pack_into("<I", head, 152, 2)
            big.write_bytes(bytes(head) + b"\0" * (64 * 1024 * 1024 - 160))
            ok, why, note = setup.expert_tier_device_check(str(big), 64)
            self.assertIn("checksum does not match", note)

    def test_the_deepseek_header_is_read_with_its_own_layout(self):
        import struct
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            dev = Path(d) / "ds4.img"
            head = bytearray(176)
            head[0:8] = b"STRPTDS4"
            struct.pack_into("<I", head, 8, 1)
            struct.pack_into("<I", head, 88, 2)            # fill_state 2 at byte 88 (the Qwen header keeps it at 152)
            struct.pack_into("<I", head, 172, (0xFFFFFFFF ^ setup._tier_crc32c(bytes(head[:172]), 0xFFFFFFFF)) & 0xFFFFFFFF)
            dev.write_bytes(bytes(head) + b"\0" * (8 * 1024 * 1024 - 176))
            ok, why, note = setup.expert_tier_device_check(str(dev), 8)
            self.assertTrue(ok, why)
            self.assertIn("header is complete", note)      # not "no tier header yet": setup knows the DeepSeek layout
            self.assertNotIn("refill", note)
            struct.pack_into("<I", head, 88, 1)            # an unfinished fill (the DeepSeek checksum covers fill_state: the engine rewrites it with the byte)
            struct.pack_into("<I", head, 172, (0xFFFFFFFF ^ setup._tier_crc32c(bytes(head[:172]), 0xFFFFFFFF)) & 0xFFFFFFFF)
            dev.write_bytes(bytes(head) + b"\0" * (8 * 1024 * 1024 - 176))
            ok, why, note = setup.expert_tier_device_check(str(dev), 8)
            self.assertIn("fill_state 1", note)

    def test_the_pack_identity_is_the_engines_to_judge(self):
        """The engine's pack identity hashes every file's name, size and time at the file system's own resolution; setup does not
        guess it any more (an aggregate it computed differently reported a valid header as another model's)."""
        self.assertFalse(hasattr(setup, "expert_tier_pack_id"))

    def test_a_block_device_is_sized_by_ioctl_and_other_kinds_are_refused(self):
        import stat
        import struct
        from unittest import mock
        fake = mock.Mock(st_mode=stat.S_IFBLK | 0o660, st_size=0)          # stat() of a block device: size 0, the mode says block
        with mock.patch.object(Path, "stat", lambda self, *a, **k: fake), mock.patch("os.open", return_value=99), \
                mock.patch("os.close"), mock.patch("fcntl.ioctl", return_value=struct.pack("Q", 3 << 30)) as ioc, \
                mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(setup.expert_tier_device_size("/dev/fake"), ("block", 3 << 30))
            self.assertEqual(ioc.call_args[0][1], setup.expert_tier._BLKGETSIZE64)
        with mock.patch.object(Path, "stat", lambda self, *a, **k: fake), mock.patch.object(sys, "platform", "win32"):
            kind, why = setup.expert_tier_device_size("\\\\.\\PhysicalDrive1")
            self.assertIsNone(kind)
            self.assertIn("Linux only", why)
        char = mock.Mock(st_mode=stat.S_IFCHR | 0o666, st_size=0)
        with mock.patch.object(Path, "stat", lambda self, *a, **k: char):
            self.assertEqual(setup.expert_tier_device_size("/dev/null")[0], None)
        ok, why, _ = setup.expert_tier_device_check("/dev/null", 1)
        self.assertFalse(ok)
        self.assertIn("neither a regular file nor a block device", why)

    def test_setup_checks_the_devices_and_asks_before_writing_over_data(self):
        """§5.4: setup's --expert-tier check is the engines': the canonical entry is what it keeps; two names of one device and a
        model file are refused; a device holding data is written only after a yes - a typed yes in a terminal, else
        --expert-tier-overwrite (--yes alone is not a yes to losing data) - and the yes clears the first mib MiB the tier may write."""
        import argparse
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            def dev(name, head=b""):
                p = Path(d) / name
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "wb") as fh:
                    fh.write(head)
                    fh.truncate(4 << 20)
                return p
            ns = lambda **k: argparse.Namespace(**{"yes": True, "expert_tier_overwrite": False, **k})   # noqa: E731
            a = dev("a.tier")
            code, out, _ = run(setup.check_expert_tiers, [f"{a},MIB=4,Verify=Full"], ns())
            self.assertIsNone(code, out)
            self.assertIn("writes over", out)
            with contextlib.redirect_stdout(io.StringIO()):
                got = setup.check_expert_tiers([f"{a},MIB=4,Verify=Full"], ns())
            self.assertEqual([raw for raw, _t in got], [f"{a},mib=4,verify=full"])
            os.link(a, Path(d) / "a.hard")
            code, out, _ = run(setup.check_expert_tiers, [f"{a},mib=4", f"{Path(d) / 'a.hard'},mib=4"], ns())
            self.assertEqual(code, 1)
            self.assertIn("same device", out)
            model = dev("pack/experts.bin")
            code, out, _ = run(lambda e, n: setup.check_expert_tiers(e, n, protect=[Path(d) / "pack"]), [f"{model},mib=4"], ns())
            self.assertEqual(code, 1)
            self.assertIn("model's file", out)
            photos = dev("photos.img", b"\0" * 99 + b"my photos")
            code, out, _ = run(setup.check_expert_tiers, [f"{photos},mib=4"], ns())        # --yes, no terminal: stops
            self.assertEqual(code, 1)
            self.assertIn("--expert-tier-overwrite", out)
            self.assertIn(b"my photos", photos.read_bytes()[:200])
            with mock.patch.object(sys.stdin, "isatty", lambda: True), mock.patch.object(setup, "flush_typed_ahead", lambda: None):
                code, out, _ = run(setup.check_expert_tiers, [f"{photos},mib=4"], ns(yes=False), stdin="no")
                self.assertEqual(code, 1)
                self.assertIn(b"my photos", photos.read_bytes()[:200])
                code, out, _ = run(setup.check_expert_tiers, [f"{photos},mib=4"], ns(yes=False), stdin="yes")
                self.assertIsNone(code, out)
            self.assertIn(b"my photos", photos.read_bytes()[:200])     # the check writes nothing: the model's files are not known yet
            cfg = {"args": ["--native", "m.gguf", "--expert-tier", f"{photos},mib=4"]}
            code, out, _ = run(setup.check_config_expert_tiers, cfg)    # the settings are written: the confirmed yes clears the mib
            self.assertIsNone(code, out)
            self.assertFalse(any(photos.read_bytes()[:4 << 20]))
            photos2 = dev("photos2.img", b"x")
            code, out, _ = run(setup.check_expert_tiers, [f"{photos2},mib=4"], ns(expert_tier_overwrite=True))
            self.assertIsNone(code, out)
            self.assertEqual(photos2.read_bytes()[:1], b"x")            # (still unwritten at the check)
            code, out, _ = run(setup.check_config_expert_tiers, {"args": ["--expert-tier", f"{photos2},mib=4"]})
            self.assertIsNone(code, out)
            self.assertFalse(any(photos2.read_bytes()[:4 << 20]))
            # a device that was never confirmed is never cleared, even when the config names it
            photos3 = dev("photos3.img", b"\0" * 10 + b"kept")
            code, out, _ = run(setup.check_config_expert_tiers, {"args": ["--expert-tier", f"{photos3},mib=4"]})
            self.assertEqual(code, 1)
            self.assertIn(b"kept", photos3.read_bytes()[:64])

    def test_a_config_whose_tier_is_a_model_file_is_not_written(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            pack = Path(d) / "pack"
            pack.mkdir()
            model = pack / "experts.bin"
            with open(model, "wb") as fh:
                fh.truncate(4 << 20)
            os.symlink(model, Path(d) / "tier.link")
            cfg = {"args": ["--pack", str(pack), "--expert-tier", f"{Path(d) / 'tier.link'},mib=4"]}
            code, out, _ = run(setup.check_config_expert_tiers, cfg)
            self.assertEqual(code, 1)
            self.assertIn("model's file", out)
            code, out, _ = run(setup.check_config_expert_tiers, {"args": ["--pack", str(pack)]})   # no tier: nothing to check
            self.assertIsNone(code)

    def test_a_regular_file_keeps_its_length_as_its_size(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "t.img"
            f.write_bytes(b"\0" * 4096)
            self.assertEqual(setup.expert_tier_device_size(str(f)), ("file", 4096))
            self.assertEqual(setup.expert_tier_device_size(str(Path(d) / "gone"))[0], None)
            (Path(d) / "empty").write_bytes(b"")
            self.assertEqual(setup.expert_tier_device_size(str(Path(d) / "empty")), ("file", 0))   # (the check then says "smaller than mib")

    @unittest.skipUnless(sys.platform.startswith("linux") and hasattr(os, "geteuid") and os.geteuid() == 0 and shutil.which("losetup"),
                         "a loop-backed block device needs root and losetup")
    def test_a_real_loop_device(self):
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / "loop.img"
            with open(img, "wb") as fh:
                fh.truncate(32 * 1024 * 1024)
            try:
                dev = subprocess.run(["losetup", "--find", "--show", str(img)], capture_output=True, text=True, check=True).stdout.strip()
            except (OSError, subprocess.CalledProcessError) as e:
                self.skipTest(f"losetup unavailable here: {e}")
            try:
                self.assertEqual(setup.expert_tier_device_size(dev), ("block", 32 * 1024 * 1024))
                ok, why, note = setup.expert_tier_device_check(dev, 16)
                self.assertTrue(ok, why)
                ok, why, _ = setup.expert_tier_device_check(dev, 64)
                self.assertFalse(ok)
                self.assertIn("smaller than mib=64 MiB", why)
            finally:
                subprocess.run(["losetup", "-d", dev], check=False)


if __name__ == "__main__":
    unittest.main()
