"""strata_control/expert_tier.py: the one --expert-tier grammar (the same cases as the engines' tests/core/expert_tier_spec_test.cpp)
and the checks made before a device is written: aliases by identity, the model's files, data that is not a tier, and the
clearing that follows a yes.  No GPU, no download; a block device only where this process may make one (root, losetup)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from strata_control import expert_tier as T

CASES = Path(__file__).resolve().parents[1] / "core" / "expert_tier_spec_cases.txt"


def cases():
    for line in CASES.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        yield line.split("\t")


class Grammar(unittest.TestCase):
    def test_every_shared_case(self):
        n = 0
        for c in cases():
            n += 1
            spec, why = T.parse(c[1])
            if c[0] == "ok":
                self.assertIsNotNone(spec, f"{c[1]!r} refused: {why}")
                self.assertEqual(T.canonical(spec), c[2], c[1])
                self.assertEqual(T.canonical(T.parse(c[2])[0]), c[2])          # the canonical form round-trips
            else:
                self.assertIsNone(spec, f"{c[1]!r} accepted as {spec}")
                self.assertTrue(why, c[1])
        self.assertGreaterEqual(n, 20)

    def test_the_tuple_shape_setup_used(self):
        self.assertEqual(T.parse("a.bin,mib=100,reserve=4,verify=full")[0], ("a.bin", 100, 4, True))

    def test_parse_all_keeps_the_order_and_names_the_bad_entry(self):
        specs, bad = T.parse_all(["b,mib=2", "a,mib=1"])
        self.assertEqual([s.path for s in specs], ["b", "a"])
        self.assertEqual(bad, [])
        _specs, bad = T.parse_all(["a,mib=1", "b,mib=-1"])
        self.assertEqual(len(bad), 1)
        self.assertIn("--expert-tier b,mib=-1", bad[0])

    def test_model_paths_from_args_names_every_shard(self):
        args = ["--native", "/m/Q-00001-of-00003.gguf", "--pack", "/p", "--expert-profile", "none", "--max-context", "8192"]
        got = T.model_paths_from_args(args)
        self.assertEqual(got, ["/m/Q-00001-of-00003.gguf", "/m/Q-00002-of-00003.gguf", "/m/Q-00003-of-00003.gguf", "/p"])


class Destinations(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)

    def dev(self, name, mib=4, head=b""):
        p = self.d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(head)
            fh.truncate(mib * 1024 * 1024)
        return p

    def spec(self, p, mib=4):
        return T.parse(f"{p},mib={mib}")[0]

    def test_a_good_device_has_no_problem(self):
        self.assertEqual(T.problems([self.spec(self.dev("a.tier"))]), [])

    def test_too_small_and_missing(self):
        probs = T.problems([self.spec(self.dev("s.tier", mib=1), mib=4), self.spec(self.d / "gone.tier")])
        self.assertTrue(any("smaller than mib=4" in p for p in probs), probs)
        self.assertTrue(any("cannot be opened" in p for p in probs), probs)

    @unittest.skipIf(sys.platform == "win32", "symlinks need extra rights on Windows")
    def test_two_names_of_one_device(self):
        a = self.dev("a.tier")
        (self.d / "a.link").symlink_to(a)
        os.link(a, self.d / "a.hard")
        (self.d / "x").mkdir()
        for other in (self.d / "a.link", self.d / "a.hard", self.d / "x" / ".." / "a.tier"):
            probs = T.problems([self.spec(a), self.spec(other)])
            self.assertTrue(any("same device" in p for p in probs), (other, probs))
        self.assertTrue(any("same device" in p for p in T.problems([self.spec(a), self.spec(a)])))

    @unittest.skipIf(sys.platform == "win32", "symlinks need extra rights on Windows")
    def test_a_model_file_or_a_tier_in_the_model_folder(self):
        pack = self.d / "pack"
        model = self.dev("pack/experts.bin")
        (self.d / "model.link").symlink_to(model)
        os.link(model, self.d / "model.hard")
        inside = self.dev("pack/sub/tier.bin")
        for p in (model, self.d / "model.link", self.d / "model.hard", inside):
            probs = T.problems([self.spec(p)], [pack])
            self.assertTrue(any("model's file" in x for x in probs), (p, probs))
        outside = self.dev("tiers/tier.bin")
        self.assertEqual(T.problems([self.spec(outside)], [pack, self.d / "nowhere"]), [])
        # the args a run config carries name the model's files
        probs = T.problems([self.spec(self.d / "model.link")], T.model_paths_from_args(["--pack", str(pack)]), devices=False)
        self.assertTrue(probs)

    def test_blank_tier_and_foreign(self):
        blank = self.dev("blank.img")
        qwen = self.dev("qwen.img", head=b"STRPTIER" + b"\x01" * 100)
        ds4b = self.dev("ds4.img", head=b"\0" * T.HEADER_AREA + b"STRPTDS4")   # (header A torn: B still says it is a tier)
        late = self.dev("late.img", head=b"\0" * T.CLAIM_BYTES + b"data")       # data past the header areas, inside the mib: foreign
        photos = self.dev("photos.img", head=b"\0" * 5000 + b"my photos")
        specs = [self.spec(p) for p in (blank, qwen, ds4b, late, photos)]
        self.assertEqual(T.foreign(specs), [str(late), str(photos)])

    def test_a_sparse_file_with_data_past_its_prefix_is_not_blank(self):
        # the first 128 KiB are zeros; the data sits 3 MiB in, a hole in between: a prefix check would call this blank
        p = self.d / "sparse.img"
        with open(p, "wb") as fh:
            fh.seek(3 * 1024 * 1024)
            fh.write(b"someone's data")
            fh.truncate(4 * 1024 * 1024)
        self.assertEqual(T.foreign([self.spec(p)]), [str(p)])
        # the same data past the device's mib is never written over by a fill of that mib: the device is blank for it
        q = self.d / "beyond.img"
        with open(q, "wb") as fh:
            fh.seek(6 * 1024 * 1024)
            fh.write(b"outside the tier")
            fh.truncate(8 * 1024 * 1024)
        self.assertEqual(T.foreign([self.spec(q, mib=4)]), [])

    def test_a_zeroed_or_truncated_file_is_blank_without_a_question(self):
        z = self.d / "zeros.img"
        with open(z, "wb") as fh:
            fh.write(b"\0" * (4 * 1024 * 1024))
        self.assertEqual(T.foreign([self.spec(z)]), [])
        t = self.d / "trunc.img"
        with open(t, "wb") as fh:
            fh.truncate(4 * 1024 * 1024)
        self.assertEqual(T.foreign([self.spec(t)]), [])

    def test_claim_clears_the_mib_it_may_write_and_nothing_past_it(self):
        p = self.d / "photos.img"
        with open(p, "wb") as fh:
            fh.write(b"\x07" * (T.CLAIM_BYTES + 10))
            fh.seek(3 * 1024 * 1024)
            fh.write(b"deep data")
            fh.seek(5 * 1024 * 1024)
            fh.write(b"past the mib")
            fh.truncate(6 * 1024 * 1024)
        T.claim(str(p), 4)
        data = p.read_bytes()
        self.assertFalse(any(data[:4 * 1024 * 1024]))             # the whole mib is zero now: the engine finds it blank
        self.assertEqual(data[5 * 1024 * 1024:5 * 1024 * 1024 + 12], b"past the mib")
        self.assertEqual(len(data), 6 * 1024 * 1024)
        self.assertEqual(T.foreign([self.spec(p)]), [])

    def test_the_lock_is_held_while_a_device_is_open(self):
        import subprocess
        p = self.dev("locked.img")
        code = ("import sys, time\n"
                "from strata_control import expert_tier as T\n"
                "with T.device_lock(sys.argv[1]):\n"
                "    print('held', flush=True)\n"
                "    time.sleep(30)\n")
        proc = subprocess.Popen([sys.executable, "-c", code, str(p)], stdout=subprocess.PIPE, text=True,
                                cwd=str(Path(__file__).resolve().parents[2]))
        try:
            self.assertEqual(proc.stdout.readline().strip(), "held")
            probs = T.problems([self.spec(p)])
            self.assertTrue(any("another process holds" in x for x in probs), probs)
            with self.assertRaises(T.DeviceBusy):
                T.claim(str(p), 4)
            self.assertEqual(T.foreign([self.spec(p)]), [])        # not judged while another process holds it
            with self.assertRaises(T.DeviceBusy):
                with T.device_lock(str(p)):
                    pass
        finally:
            proc.kill()
            proc.wait()
        self.assertEqual(T.problems([self.spec(p)]), [])            # released when the holder is gone

    def test_the_lock_is_released_on_an_error_inside_the_block(self):
        p = self.dev("err.img")
        with self.assertRaises(RuntimeError):
            with T.device_lock(str(p)):
                raise RuntimeError("boom")
        self.assertEqual(T.problems([self.spec(p)]), [])

    def test_absolute_paths_do_not_depend_on_the_working_directory(self):
        import os as _os
        here = Path.cwd()
        try:
            _os.chdir(self.d)
            (self.d / "rel dir").mkdir()
            spec = T.parse("rel dir/t.tier,mib=4")[0]
            a = T.absolute(spec)
            self.assertEqual(a.path, str(self.d / "rel dir" / "t.tier"))
            self.assertEqual(a.mib, 4)
            _os.chdir(self.tmp.name + "/..")                      # somewhere else entirely
            self.assertEqual(T.absolute(T.parse(f"{a.path},mib=4")[0]).path, a.path)
        finally:
            _os.chdir(here)

    def test_tilde_expands_to_the_home_folder(self):
        import os as _os
        from unittest import mock as _mock
        with _mock.patch.dict(_os.environ, {"HOME": str(self.d), "USERPROFILE": str(self.d)}):
            a = T.absolute(T.parse("~/tiers/t.tier,mib=8")[0])
        self.assertEqual(a.path, str(self.d / "tiers" / "t.tier"))

    def test_absolute_args_rewrite_only_the_tier_values(self):
        out = T.absolute_args(["--native", "m.gguf", "--max-context", "8192", "--expert-tier", "rel.tier,mib=8",
                               "--expert-tier", "bad,mib=-1"])
        self.assertEqual(out[:4], ["--native", "m.gguf", "--max-context", "8192"])
        self.assertTrue(out[5].endswith("rel.tier,mib=8") and Path(out[5].split(",")[0]).is_absolute(), out)
        self.assertEqual(out[7], "bad,mib=-1")                    # an entry that does not parse is left for the checks

    def test_the_warning_names_the_devices(self):
        w = T.overwrite_warning([self.spec("/a"), self.spec("/b")])
        self.assertIn("/a, /b", w)
        self.assertIn("writes over", w)

    @unittest.skipUnless(sys.platform.startswith("linux") and hasattr(os, "geteuid") and os.geteuid() == 0 and shutil.which("losetup"),
                         "a loop-backed block device needs root and losetup")
    def test_a_block_device_under_two_names_and_in_use(self):
        img = self.dev("loop.img", mib=16)
        dev = subprocess.run(["losetup", "--find", "--show", str(img)], capture_output=True, text=True).stdout.strip()
        if not dev:
            self.skipTest("losetup made no device")
        self.addCleanup(subprocess.run, ["losetup", "-d", dev])
        (self.d / "loop.link").symlink_to(dev)
        probs = T.problems([self.spec(dev, 8), self.spec(self.d / "loop.link", 8)])
        self.assertTrue(any("same device" in p for p in probs), probs)
        fd = os.open(dev, os.O_RDONLY | os.O_EXCL)               # held exclusively, as a mount holds it
        try:
            self.assertTrue(any("in use" in p for p in T.problems([self.spec(dev, 8)])))
        finally:
            os.close(fd)
        self.assertEqual(T.problems([self.spec(dev, 8)]), [])


if __name__ == "__main__":
    unittest.main()
