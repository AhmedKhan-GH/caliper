"""Unit tests of python/caliper/loader_host.py, run by ctest with the test
interpreter in isolated mode:

    python -I test_loader_host.py <loader_host.py> <scratch folder>

The host is imported by its path, as Caliper runs it, and every file the tests
write lies under the scratch folder, which is inside the build tree. Only the
standard library, numpy and, when installed, pyarrow are used.
"""
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import types
import unittest

HOST_PATH = None
SCRATCH = None
host = None
np = None


def load_host(path):
    spec = importlib.util.spec_from_file_location("loader_host_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def has_pyarrow():
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError:
        return False
    return True


class Folder:
    """A fresh empty folder under the scratch folder."""

    count = 0

    @classmethod
    def make(cls, name):
        cls.count += 1
        path = os.path.join(SCRATCH, "%s-%d" % (name, cls.count))
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
        return path


def read_manifest(folder):
    with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


class MakeIdTest(unittest.TestCase):
    def test_matches_the_library_rule(self):
        self.assertEqual(host.make_id("layers.0.attn"), "layers_0_attn")
        self.assertEqual(host.make_id("Weights-2"), "weights-2")
        self.assertEqual(host.make_id("Weights-2", underscores_only=True), "weights_2")
        self.assertEqual(host.make_id("0abc"), "f_0abc")
        self.assertEqual(host.make_id(""), "f_")
        self.assertEqual(host.make_id("é"), "f___")
        long_name = host.make_id("x" * 100)
        self.assertEqual(len(long_name), 64)
        self.assertTrue(long_name.startswith("x" * 55 + "-"))
        self.assertTrue(host.valid_id(long_name))
        self.assertNotEqual(host.make_id("x" * 100), host.make_id("x" * 101))

    def test_valid_id(self):
        for good in ["a", "a_b-c", "z" * 64]:
            self.assertTrue(host.valid_id(good), good)
        for bad in ["", "A", "1a", "a.b", "a" * 65, None, 3, "a b"]:
            self.assertFalse(host.valid_id(bad), bad)


class TensorTest(unittest.TestCase):
    def setUp(self):
        self.folder = Folder.make("tensor")
        self.out = host.Writer(self.folder, np, parquet=False)

    def load(self, name):
        return np.load(os.path.join(self.folder, name + ".npy"), allow_pickle=False)

    def test_each_catalog_dtype_round_trips(self):
        cases = {
            "bool": np.array([True, False]), "u8": np.array([0, 255], np.uint8),
            "u16": np.array([1, 65535], ">u2"), "u32": np.array([0, 2**32 - 1], np.uint32),
            "u64": np.array([0, 2**64 - 1], np.uint64), "i8": np.array([-128, 127], np.int8),
            "i16": np.array([-32768, 32767], np.int16), "i32": np.array([-2**31, 2**31 - 1], ">i4"),
            "i64": np.array([-2**63, 2**63 - 1], np.int64), "f16": np.array([1.5, -2], np.float16),
            "f32": np.array([0.25], ">f4"), "f64": np.array([math.nan, math.inf, -0.0]),
        }
        for dtype, array in cases.items():
            name = "t_" + dtype
            self.out.tensor(name, array)
            back = self.load(name)
            self.assertEqual(back.dtype.byteorder in "<|=", True, dtype)
            self.assertTrue(back.flags.c_contiguous)
            np.testing.assert_array_equal(back, array)
            self.assertEqual(host.catalog_dtype(back.dtype), dtype)
        entries = {e["name"]: e for e in self.out.manifest()["outputs"]}
        self.assertEqual(entries["t_u16"]["dtype"], "u16")
        self.assertEqual(entries["t_f64"]["shape"], [3])
        for e in entries.values():
            self.assertEqual(e["kind"], "tensor")
            self.assertEqual(e["bytes"], os.path.getsize(os.path.join(self.folder, e["file"])))

    def test_layout_is_row_major_little_endian(self):
        f = np.asfortranarray(np.arange(6, dtype=">i4").reshape(2, 3))
        self.out.tensor("f", f)
        with open(os.path.join(self.folder, "f.npy"), "rb") as h:
            head = h.read(128).decode("latin1")
        self.assertIn("'<i4'", head)
        self.assertIn("'fortran_order': False", head)
        np.testing.assert_array_equal(self.load("f"), f)
        self.out.tensor("scalar", np.float32(2.5))
        self.assertEqual(self.load("scalar").shape, ())
        self.out.tensor("empty", np.zeros((0, 4)))
        self.assertEqual(self.load("empty").shape, (0, 4))
        self.out.tensor("nested", [[1, 2], [3, 4]])
        self.assertEqual(self.out.manifest()["outputs"][-1]["shape"], [2, 2])

    def test_refusals(self):
        with self.assertRaisesRegex(TypeError, "object dtype"):
            self.out.tensor("o", np.array([{}, None], dtype=object))
        with self.assertRaises(TypeError):
            self.out.tensor("s", np.array(["text"]))
        with self.assertRaises(TypeError):
            self.out.tensor("c", np.array([1 + 2j]))
        with self.assertRaises(TypeError):
            self.out.tensor("r", np.zeros(2, dtype=[("a", "i4"), ("b", "f4")]))
        with self.assertRaises(TypeError):
            self.out.tensor("d", np.array(["2026-01-01"], dtype="datetime64[D]"))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.out.tensor("deep", np.zeros((1,) * 9))
        for bad in ["W", "1w", "a.b", "../x", "", "a" * 65]:
            with self.assertRaisesRegex(ValueError, "does not match"):
                self.out.tensor(bad, [1])
        self.out.tensor("w", [1])
        with self.assertRaisesRegex(ValueError, "two tensors"):
            self.out.tensor("w", [2])
        # A file the script planted under an output's name is never overwritten.
        with open(os.path.join(self.folder, "planted.npy"), "wb") as f:
            f.write(b"x")
        with self.assertRaises(FileExistsError):
            self.out.tensor("planted", [1])
        self.assertEqual(len(self.out.manifest()["outputs"]), 1)

    def test_no_pickle_is_ever_written(self):
        self.out.tensor("w", np.arange(3))
        with open(os.path.join(self.folder, "w.npy"), "rb") as f:
            self.assertNotIn(b"\x80\x04", f.read(16))
        np.load(os.path.join(self.folder, "w.npy"), allow_pickle=False)


class CsvTableTest(unittest.TestCase):
    def setUp(self):
        self.folder = Folder.make("csv")
        self.out = host.Writer(self.folder, np, parquet=False)

    def text(self, name):
        with open(os.path.join(self.folder, name + ".csv"), encoding="utf-8", newline="") as f:
            return f.read()

    def test_typed_header_and_values(self):
        self.out.table("rows", [
            {"i": 1, "x": 0.5, "flag": True, "s": 'a,"b"'},
            {"i": 2, "x": math.nan, "flag": False, "s": ""},
            {"i": None, "x": -1, "flag": None, "s": "line\nbreak", "late": "z"},
        ])
        self.assertEqual(self.text("rows"),
                         '"i:BIGINT","x:DOUBLE","flag:BOOLEAN","s:VARCHAR","late:VARCHAR"\n'
                         '1,0.5,true,"a,""b""",\n'
                         '2,nan,false,"",\n'
                         ',-1.0,,"line\nbreak","z"\n')
        entry = self.out.manifest()["outputs"][0]
        self.assertEqual(entry["kind"], "table")
        self.assertEqual(entry["file"], "rows.csv")
        self.assertNotIn("dtype", entry)

    def test_dict_of_columns_and_numpy(self):
        self.out.table("cols", {"step": np.arange(3), "loss": np.array([0.5, np.inf, -np.inf]),
                                "ok": np.array([True, False, True]), "none": [None, None, None]})
        self.assertEqual(self.text("cols"),
                         '"step:BIGINT","loss:DOUBLE","ok:BOOLEAN","none:VARCHAR"\n'
                         '0,0.5,true,\n1,inf,false,\n2,-inf,true,\n')

    def test_refusals(self):
        with self.assertRaisesRegex(TypeError, "mixes"):
            self.out.table("a", {"m": [1, "x"]})
        with self.assertRaisesRegex(TypeError, "mixes"):
            self.out.table("b", {"m": [True, 1]})
        with self.assertRaisesRegex(TypeError, "without pyarrow"):
            self.out.table("c", {"m": [b"bytes"]})
        with self.assertRaisesRegex(ValueError, "64-bit"):
            self.out.table("d", {"m": [2**70]})
        with self.assertRaisesRegex(ValueError, "different lengths"):
            self.out.table("e", {"a": [1], "b": [1, 2]})
        with self.assertRaisesRegex(ValueError, "at least one column"):
            self.out.table("f", {})
        with self.assertRaisesRegex(ValueError, "control character"):
            self.out.table("g", {"a\nb": [1]})
        with self.assertRaises(TypeError):
            self.out.table("h", [1, 2])
        with self.assertRaises(TypeError):
            self.out.table("i", "rows")
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            self.out.table("j", {"m": np.zeros((2, 2))})
        with self.assertRaisesRegex(ValueError, "NUL"):
            self.out.table("k", {"m": ["a\0b"]})


@unittest.skipUnless(has_pyarrow(), "pyarrow is not installed")
class ParquetTableTest(unittest.TestCase):
    def test_round_trip(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        folder = Folder.make("parquet")
        out = host.Writer(folder, np)
        out.table("rows", [{"a": 1, "b": "x"}, {"a": 2, "c": 0.5}])
        out.table("cols", {"step": np.arange(3), "s": ["a", "b", None]})
        out.table("arrow", pa.table({"v": [1.5, 2.5]}))
        self.assertEqual(pq.read_table(os.path.join(folder, "rows.parquet")).to_pylist(),
                         [{"a": 1, "b": "x", "c": None}, {"a": 2, "b": None, "c": 0.5}])
        self.assertEqual(pq.read_table(os.path.join(folder, "cols.parquet")).column("s").to_pylist(),
                         ["a", "b", None])
        self.assertEqual([e["file"] for e in out.manifest()["outputs"]],
                         ["rows.parquet", "cols.parquet", "arrow.parquet"])
        with self.assertRaisesRegex(ValueError, "no columns"):
            out.table("empty", pa.table({}))


class FakeTensor:
    def __init__(self, array, dtype):
        self.array = array
        self.dtype = dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return FakeTensor(self.array.astype(np.float32), "float32")

    def numpy(self):
        if self.dtype in ("bfloat16", "complex32"):
            raise TypeError("Got unsupported ScalarType %s" % self.dtype)
        return self.array


class TorchWeightsTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        fake = types.ModuleType("torch")
        fake.Tensor = FakeTensor
        fake.bfloat16 = "bfloat16"
        state = {
            "layers.0.w": FakeTensor(np.arange(4, dtype=np.float32).reshape(2, 2), "float32"),
            "layers.0.b": FakeTensor(np.array([0.5, 1.5], np.float32), "bfloat16"),
            "opt": {"state": [FakeTensor(np.array([7]), "int64"), 3]},
            "layers_0_w": FakeTensor(np.array([1.0]), "float64"),
            "z": FakeTensor(np.array([1.0]), "complex32"),
            "epoch": 3,
        }

        def load(f, **kwargs):
            self.calls.append((f, kwargs))
            return state

        fake.load = load
        self.saved = sys.modules.get("torch")
        sys.modules["torch"] = fake

    def tearDown(self):
        if self.saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self.saved

    def test_weights_only_and_names(self):
        folder = Folder.make("torch")
        out = host.Writer(folder, np, parquet=False)
        names = out.torch_weights("model.pt")
        self.assertEqual(self.calls, [("model.pt", {"map_location": "cpu", "weights_only": True})])
        self.assertEqual(names, ["layers_0_w", "layers_0_b", "opt_state_0", "layers_0_w-2"])
        manifest = out.manifest()
        entries = {e["name"]: e for e in manifest["outputs"]}
        self.assertEqual(entries["layers_0_w"]["title"], "layers.0.w")
        self.assertEqual(entries["layers_0_w"]["warnings"], [host.PICKLE_WARNING])
        self.assertEqual(entries["layers_0_b"]["warnings"], [host.PICKLE_WARNING, "converted from bf16 to f32"])
        self.assertEqual(entries["layers_0_b"]["dtype"], "f32")
        self.assertEqual(entries["opt_state_0"]["title"], "opt.state.0")
        self.assertEqual(entries["layers_0_w-2"]["title"], "layers_0_w")
        self.assertEqual(len(manifest["warnings"]), 2)
        self.assertIn("z was skipped", manifest["warnings"][0])
        self.assertEqual(manifest["warnings"][1], "2 values of the checkpoint are not tensors and were skipped.")


class WarnTest(unittest.TestCase):
    def test_warnings_are_cleaned_and_bounded(self):
        out = host.Writer(Folder.make("warn"), np, parquet=False)
        out.warn("two\nlines\x00")
        out.warn("x" * 5000)
        for i in range(100):
            out.warn("w%d" % i)
        warnings = out.manifest()["warnings"]
        self.assertEqual(warnings[0], "two lines ")
        self.assertEqual(len(warnings[1]), host.MAX_WARNING_CHARS)
        self.assertEqual(len(warnings), host.MAX_WARNINGS)
        self.assertEqual(warnings[-1], "39 more warnings were dropped.")


class MainTest(unittest.TestCase):
    def run_host(self, script_text, extra_files=None, output=None):
        folder = Folder.make("main")
        script = os.path.join(folder, "script.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(script_text)
        for name, text in (extra_files or {}).items():
            with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
                f.write(text)
        out = output or os.path.join(folder, "out")
        if output is None:
            os.makedirs(out)
        data = os.path.join(folder, "in.bin")
        with open(data, "wb") as f:
            f.write(b"\x00\x01")
        proc = subprocess.run([sys.executable, "-I", HOST_PATH, "--script", script, "--input", data,
                               "--output", out], cwd=out if os.path.isdir(out) else folder,
                              capture_output=True, text=True, timeout=120)
        return proc, out, folder

    def test_a_script_writes_outputs_and_a_manifest(self):
        proc, out, folder = self.run_host(
            "import helper\n\ndef load(path, out):\n"
            "    with open(path, 'rb') as f:\n        out.tensor('raw', list(f.read()))\n"
            "    out.table('t', {'v': [helper.VALUE]})\n", {"helper.py": "VALUE = 7\n"})
        # The script's folder is first on sys.path, so a module beside it imports.
        self.assertEqual(proc.returncode, 0, proc.stderr)
        manifest = read_manifest(out)
        self.assertEqual(manifest["caliper_python_loader"], 0)
        self.assertEqual([e["name"] for e in manifest["outputs"]], ["raw", "t"])
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(out), "__pycache__")))

    def test_helper_module_beside_the_script(self):
        proc, out, _ = self.run_host("import helper\n\ndef load(path, out):\n    out.tensor('v', [helper.VALUE])\n",
                                     {"helper.py": "VALUE = 42\n"})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(int(np.load(os.path.join(out, "v.npy"), allow_pickle=False)[0]), 42)

    def test_a_failing_script_writes_error_json(self):
        proc, out, _ = self.run_host("def load(path, out):\n    raise ValueError('broken on purpose')\n")
        self.assertEqual(proc.returncode, host.EXIT_SCRIPT_FAILED)
        self.assertIn("ValueError: broken on purpose", proc.stderr)
        with open(os.path.join(out, "error.json"), encoding="utf-8") as f:
            error = json.load(f)["error"]
        self.assertEqual(error["type"], "ValueError")
        self.assertEqual(error["message"], "ValueError: broken on purpose")
        self.assertFalse(os.path.exists(os.path.join(out, "manifest.json")))

    def test_no_load_function_and_exit_inside_load(self):
        proc, out, _ = self.run_host("x = 1\n")
        self.assertEqual(proc.returncode, host.EXIT_SCRIPT_FAILED)
        self.assertIn("load(path, out)", proc.stderr)
        proc, out, _ = self.run_host("import sys\n\ndef load(path, out):\n    sys.exit(0)\n")
        self.assertEqual(proc.returncode, host.EXIT_SCRIPT_FAILED)

    def test_a_planted_manifest_is_not_replaced(self):
        proc, out, _ = self.run_host("def load(path, out):\n    open('manifest.json', 'w').write('{}')\n"
                                     "    out.tensor('w', [1])\n")
        self.assertEqual(proc.returncode, host.EXIT_SCRIPT_FAILED)
        self.assertIn("FileExistsError", proc.stderr)

    def test_the_output_folder_must_be_empty(self):
        folder = Folder.make("busy")
        with open(os.path.join(folder, "x"), "w") as f:
            f.write("x")
        proc, _, _ = self.run_host("def load(path, out):\n    pass\n", output=folder)
        self.assertEqual(proc.returncode, host.EXIT_USAGE)
        proc = subprocess.run([sys.executable, "-I", HOST_PATH, "--script", "x.py"], capture_output=True,
                              text=True, timeout=120)
        self.assertEqual(proc.returncode, host.EXIT_USAGE)

    def test_main_in_process(self):
        folder = Folder.make("inproc")
        script = os.path.join(folder, "s.py")
        with open(script, "w") as f:
            f.write("def load(path, out):\n    out.tensor('p', [len(path)])\n")
        out = os.path.join(folder, "out")
        os.makedirs(out)
        saved_path = list(sys.path)
        try:
            self.assertEqual(host.main(["--script", script, "--input", "abc", "--output", out]), host.EXIT_OK)
        finally:
            sys.path[:] = saved_path
        self.assertEqual(read_manifest(out)["outputs"][0]["shape"], [1])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: test_loader_host.py <loader_host.py> <scratch folder>", file=sys.stderr)
        sys.exit(2)
    HOST_PATH = os.path.abspath(sys.argv[1])
    SCRATCH = os.path.abspath(sys.argv[2])
    shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(SCRATCH)
    import numpy as np  # noqa: E402  (the host needs numpy; so do the tests)

    host = load_host(HOST_PATH)
    sys.dont_write_bytecode = True
    result = unittest.main(argv=[sys.argv[0], "-v"], exit=False).result
    sys.exit(0 if result.wasSuccessful() else 1)
