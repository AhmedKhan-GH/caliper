"""The host of a Caliper Python loader.

Caliper runs a Python loader out of its own process with this file:

    python -I loader_host.py --script <script.py> --input <file> --output <empty folder>

and, when Caliper LIB is installed, the same works as
``python -m caliper.loader_host``. The file is self-contained: it needs only
the standard library and numpy, so it runs by path without Caliper installed.

The user's script defines one function and imports nothing from Caliper:

    def load(path, out):
        out.tensor("weights", array)     # anything numpy.asarray accepts
        out.table("metrics", rows)       # a list of dicts, a dict of columns,
                                         # or a pandas or pyarrow table
        out.warn("text")                 # a warning on every source of the run
        out.torch_weights(path)          # a torch checkpoint, weights-only

The host writes each tensor as an .npy file (never with the object dtype),
each table as Parquet when pyarrow can be imported and otherwise as CSV with a
typed header, and a manifest.json that lists the outputs. Caliper then checks
the manifest as untrusted input and reads the files with its own readers.

The typed CSV header: every header cell is "<column>:<type>", where the type
is BIGINT, DOUBLE, BOOLEAN or VARCHAR. Every header cell and every VARCHAR
value is quoted, with a quote inside doubled; NULL is an empty unquoted field;
booleans are true and false; doubles use Python's shortest repr, with nan, inf
and -inf. Fields are separated by commas and rows end with a line feed.

Exit codes: 0 when the outputs and manifest.json are written; 1 when the script
fails (error.json in the output folder says why); 2 when the arguments are
wrong; 3 when numpy cannot be imported.
"""
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import math
import os
import sys
import traceback

FORMAT_VERSION = 0
EXIT_OK = 0
EXIT_SCRIPT_FAILED = 1
EXIT_USAGE = 2
EXIT_NO_NUMPY = 3

MAX_OUTPUTS = 10000
MAX_RANK = 8
MAX_WARNINGS = 64
MAX_WARNING_CHARS = 1024
MAX_TITLE_CHARS = 256
MAX_COLUMNS = 10000
MAX_DEPTH = 32

PICKLE_WARNING = "loaded weights-only from a pickle file"

# (numpy kind, item size) to the catalog's dtype names.
_DTYPES = {
    ("b", 1): "bool",
    ("u", 1): "u8", ("u", 2): "u16", ("u", 4): "u32", ("u", 8): "u64",
    ("i", 1): "i8", ("i", 2): "i16", ("i", 4): "i32", ("i", 8): "i64",
    ("f", 2): "f16", ("f", 4): "f32", ("f", 8): "f64",
}

_INT64_MIN = -(2 ** 63)
_INT64_MAX = 2 ** 63 - 1


def valid_id(name):
    """Whether a name matches the id grammar [a-z][a-z0-9_-]{0,63}."""
    if not isinstance(name, str) or not 1 <= len(name) <= 64 or not "a" <= name[0] <= "z":
        return False
    return all("a" <= c <= "z" or "0" <= c <= "9" or c in "_-" for c in name)


def _fnv1a(data):
    h = 2166136261
    for b in data:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def make_id(text, underscores_only=False):
    """Makes an id from any text, with the same rule as Caliper's make_id():
    lower case, every byte outside [a-z0-9_-] replaced by "_", "f_" in front
    when it does not start with a letter, and at most 64 characters (a longer
    one keeps 55 and ends with "-" and 8 hexadecimal digits of a hash)."""
    raw = text.encode("utf-8")
    out = []
    for b in raw:
        if 65 <= b <= 90:
            out.append(chr(b + 32))
        elif 97 <= b <= 122 or 48 <= b <= 57 or b == 95:
            out.append(chr(b))
        elif b == 45 and not underscores_only:
            out.append("-")
        else:
            out.append("_")
    result = "".join(out)
    if not result or not "a" <= result[0] <= "z":
        result = "f_" + result
    if len(result) > 64:
        result = result[:55] + ("_" if underscores_only else "-") + "%08x" % _fnv1a(raw)
    return result


def _clean_text(text, limit):
    """One line of plain text: control characters become spaces, and the text
    is cut to `limit` characters."""
    cleaned = "".join(" " if ord(c) < 0x20 or ord(c) == 0x7F else c for c in str(text))
    return cleaned[:limit]


def catalog_dtype(dtype):
    """The catalog's name for a numpy dtype; raises TypeError for any dtype the
    catalog does not have, and always for the object dtype."""
    if dtype.hasobject:
        raise TypeError("an array with the object dtype holds Python objects, which Caliper does not load; "
                        "convert it to numbers first")
    if dtype.fields is not None or dtype.subdtype is not None:
        raise TypeError("a structured dtype (%s) is not a catalog dtype; write each field as its own tensor"
                        % dtype)
    name = _DTYPES.get((dtype.kind, dtype.itemsize))
    if name is None:
        raise TypeError("the dtype %s is not a catalog dtype (bool, u8 to u64, i8 to i64, f16, f32, f64)" % dtype)
    return name


class Writer:
    """The `out` object that a script's load() receives. Each method writes one
    file into the output folder at once, so a script that fails half way has
    still not touched anything outside that folder."""

    def __init__(self, folder, numpy, parquet=None):
        self._folder = folder
        self._np = numpy
        self._parquet = parquet  # None: Parquet when pyarrow imports; True or False to choose
        self._pyarrow = None
        self._outputs = []
        self._names = {"tensor": set(), "table": set()}
        self._warnings = []
        self._dropped_warnings = 0

    # -- the script's interface -------------------------------------------------

    def tensor(self, name, array):
        """Writes `array` (anything numpy.asarray accepts) as the tensor `name`."""
        self._tensor(name, array, None, [])

    def table(self, name, rows):
        """Writes `rows` as the table `name`: a list of dicts, a dict of columns,
        or a pandas DataFrame or pyarrow Table when those are installed."""
        self._check_name("table", name)
        pa = self._arrow()
        if pa is not None:
            table = _to_arrow(pa, self._np, rows)
            if table.num_columns == 0:
                raise ValueError("the table %s has no columns" % name)
            if table.num_columns > MAX_COLUMNS:
                raise ValueError("the table %s has more than %d columns" % (name, MAX_COLUMNS))
            file = name + ".parquet"
            with open(os.path.join(self._folder, file), "xb") as f:
                pa.parquet.write_table(table, f)
        else:
            columns = _to_columns(self._np, rows)
            file = name + ".csv"
            _write_typed_csv(os.path.join(self._folder, file), columns)
        self._add("table", name, file, None, [], None, None)

    def warn(self, text):
        """Adds a warning that every source of this run carries in the catalog."""
        if len(self._warnings) < MAX_WARNINGS - 1:
            self._warnings.append(_clean_text(text, MAX_WARNING_CHARS))
        else:
            self._dropped_warnings += 1

    def torch_weights(self, file):
        """Loads a torch checkpoint weights-only and writes each of its tensors.

        The only call is torch.load(file, map_location="cpu", weights_only=True),
        which refuses any pickled object that is not a tensor or a plain
        container. Nested dicts and lists are walked, and each tensor's key path
        ("layers.0.attn.weight") becomes its title, with the output name made
        from it as an id. bf16 tensors become f32. Every tensor carries the
        warning "loaded weights-only from a pickle file". Returns the names
        written."""
        import torch

        state = torch.load(file, map_location="cpu", weights_only=True)
        written = []
        skipped = [0]

        def walk(value, key, depth):
            if depth > MAX_DEPTH:
                raise ValueError("the checkpoint is nested more than %d levels deep" % MAX_DEPTH)
            if isinstance(value, torch.Tensor):
                name = self._torch_tensor(torch, value, key)
                if name is not None:
                    written.append(name)
            elif isinstance(value, dict):
                for k, v in value.items():
                    walk(v, "%s.%s" % (key, k) if key else str(k), depth + 1)
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    walk(v, "%s.%d" % (key, i) if key else str(i), depth + 1)
            else:
                skipped[0] += 1

        walk(state, "", 0)
        n = skipped[0]
        if n:
            self.warn("%d %s of the checkpoint %s and %s skipped." % (
                n, "value" if n == 1 else "values", "is not a tensor" if n == 1 else "are not tensors",
                "was" if n == 1 else "were"))
        return written

    # -- the host's side --------------------------------------------------------

    def manifest(self):
        warnings = list(self._warnings)
        if self._dropped_warnings:
            warnings.append("%d more warnings were dropped." % self._dropped_warnings)
        return {"caliper_python_loader": FORMAT_VERSION, "outputs": list(self._outputs), "warnings": warnings}

    def finish(self):
        """Writes manifest.json, which must not exist yet."""
        text = json.dumps(self.manifest(), indent=1, allow_nan=False)
        with open(os.path.join(self._folder, "manifest.json"), "x", encoding="utf-8") as f:
            f.write(text)

    def _check_name(self, kind, name):
        if not valid_id(name):
            raise ValueError("the %s name %r does not match [a-z][a-z0-9_-]{0,63}" % (kind, name))
        if name in self._names[kind]:
            raise ValueError("the script wrote two %ss named %s" % (kind, name))
        if len(self._outputs) >= MAX_OUTPUTS:
            raise ValueError("a Python loader writes at most %d outputs" % MAX_OUTPUTS)

    def _tensor(self, name, array, title, warnings):
        self._check_name("tensor", name)
        np = self._np
        a = np.asarray(array)
        dtype = catalog_dtype(a.dtype)
        if a.ndim > MAX_RANK:
            raise ValueError("the tensor %s has %d dimensions; at most %d are allowed" % (name, a.ndim, MAX_RANK))
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        if not a.flags.c_contiguous:
            a = a.copy(order="C")
        file = name + ".npy"
        with open(os.path.join(self._folder, file), "xb") as f:
            np.save(f, a, allow_pickle=False)
        self._add("tensor", name, file, title, warnings, dtype, [int(d) for d in a.shape])

    def _torch_tensor(self, torch, value, key):
        warnings = [PICKLE_WARNING]
        t = value.detach()
        if t.dtype == torch.bfloat16:
            t = t.float()
            warnings.append("converted from bf16 to f32")
        try:
            array = t.cpu().numpy()
            catalog_dtype(array.dtype)
        except (TypeError, RuntimeError) as e:
            self.warn("The tensor %s was skipped: %s." % (key or "(top level)", e))
            return None
        base = make_id(key) if key else "weights"
        name, n = base, 2
        while name in self._names["tensor"]:
            name = make_id("%s-%d" % (key or "weights", n))
            n += 1
        self._tensor(name, array, key or None, warnings)
        return name

    def _arrow(self):
        if self._parquet is False:
            return None
        if self._pyarrow is None:
            try:
                import pyarrow
                import pyarrow.parquet  # noqa: F401  (pyarrow.parquet is used through pyarrow)
            except ImportError:
                if self._parquet:
                    raise
                self._parquet = False
                return None
            self._pyarrow = pyarrow
        return self._pyarrow

    def _add(self, kind, name, file, title, warnings, dtype, shape):
        self._names[kind].add(name)
        size = os.lstat(os.path.join(self._folder, file)).st_size
        entry = {"name": name, "kind": kind, "file": file, "bytes": size}
        if title:
            entry["title"] = _clean_text(title, MAX_TITLE_CHARS)
        if warnings:
            entry["warnings"] = [_clean_text(w, MAX_WARNING_CHARS) for w in warnings[:MAX_WARNINGS]]
        if kind == "tensor":
            entry["dtype"] = dtype
            entry["shape"] = shape
        self._outputs.append(entry)


def _is_pandas_frame(value):
    t = type(value)
    return t.__module__.split(".")[0] == "pandas" and t.__name__ == "DataFrame"


def _to_arrow(pa, np, rows):
    if isinstance(rows, pa.Table):
        return rows
    if isinstance(rows, pa.RecordBatch):
        return pa.Table.from_batches([rows])
    if _is_pandas_frame(rows):
        return pa.Table.from_pandas(rows, preserve_index=False)
    if isinstance(rows, dict):
        for k in rows:
            if not isinstance(k, str):
                raise TypeError("a table's column names are strings, not %s" % type(k).__name__)
        return pa.table({k: (v if isinstance(v, np.ndarray) else list(v)) for k, v in rows.items()})
    if isinstance(rows, (list, tuple)):
        return pa.table(_row_columns(rows))
    raise TypeError("a table is a list of dicts, a dict of columns, or a pandas or pyarrow table, not %s"
                    % type(rows).__name__)


def _row_columns(rows):
    """A list of dicts as columns: every key of any row, in the order first
    seen, with None where a row lacks the key."""
    names = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("a table given as a list holds one dict per row, not %s" % type(row).__name__)
        for k in row:
            if not isinstance(k, str):
                raise TypeError("a table's column names are strings, not %s" % type(k).__name__)
            names.setdefault(k, None)
    return {k: [row.get(k) for row in rows] for k in names}


def _plain(np, value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_columns(np, rows):
    """A table as an ordered dict of column name to a list of plain values."""
    if _is_pandas_frame(rows):
        rows = rows.to_dict(orient="list")
    if isinstance(rows, dict):
        columns = {}
        for k, v in rows.items():
            if not isinstance(k, str):
                raise TypeError("a table's column names are strings, not %s" % type(k).__name__)
            if isinstance(v, np.ndarray):
                if v.ndim != 1:
                    raise ValueError("the column %r is not one-dimensional" % k)
                v = v.tolist()
            elif not isinstance(v, (list, tuple)):
                raise TypeError("the column %r is a list or a one-dimensional array, not %s" % (k, type(v).__name__))
            columns[k] = [_plain(np, x) for x in v]
    elif isinstance(rows, (list, tuple)):
        columns = {k: [_plain(np, x) for x in v] for k, v in _row_columns(rows).items()}
    else:
        raise TypeError("a table is a list of dicts, a dict of columns, or a pandas table, not %s"
                        % type(rows).__name__)
    if not columns:
        raise ValueError("a table needs at least one column")
    if len(columns) > MAX_COLUMNS:
        raise ValueError("a table has at most %d columns" % MAX_COLUMNS)
    lengths = {len(v) for v in columns.values()}
    if len(lengths) != 1:
        raise ValueError("the columns of a table have different lengths")
    for k in columns:
        if not k or any(ord(c) < 0x20 or ord(c) == 0x7F for c in k):
            raise ValueError("the column name %r is empty or holds a control character" % k)
    return columns


def _column_type(name, values):
    kinds = set()
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            kinds.add("bool")
        elif isinstance(v, int):
            if not _INT64_MIN <= v <= _INT64_MAX:
                raise ValueError("the column %r holds an integer outside the 64-bit range" % name)
            kinds.add("int")
        elif isinstance(v, float):
            kinds.add("float")
        elif isinstance(v, str):
            if "\0" in v:
                raise ValueError("the column %r holds a NUL character" % name)
            kinds.add("str")
        else:
            raise TypeError("the column %r holds a %s value; without pyarrow a table holds only bool, int, "
                            "float, str and None values" % (name, type(v).__name__))
    if not kinds or kinds == {"str"}:
        return "VARCHAR"
    if kinds == {"bool"}:
        return "BOOLEAN"
    if kinds == {"int"}:
        return "BIGINT"
    if kinds <= {"int", "float"}:
        return "DOUBLE"
    raise TypeError("the column %r mixes %s values" % (name, " and ".join(sorted(kinds))))


def _quote(text):
    return '"' + text.replace('"', '""') + '"'


def _cell(sql_type, value):
    if value is None:
        return ""
    if sql_type == "VARCHAR":
        return _quote(value)
    if sql_type == "BOOLEAN":
        return "true" if value else "false"
    if sql_type == "BIGINT":
        return str(value)
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return repr(value)


def _write_typed_csv(path, columns):
    names = list(columns)
    types = [_column_type(n, columns[n]) for n in names]
    count = len(columns[names[0]])
    with open(path, "x", encoding="utf-8", newline="") as f:
        f.write(",".join(_quote("%s:%s" % (n, t)) for n, t in zip(names, types)) + "\n")
        for i in range(count):
            f.write(",".join(_cell(t, columns[n][i]) for n, t in zip(names, types)) + "\n")


def _load_script(path):
    """Runs the script as a module named caliper_loader_script, with its folder
    first on sys.path so that it can import modules beside it."""
    path = os.path.abspath(path)
    sys.path.insert(0, os.path.dirname(path))
    loader = importlib.machinery.SourceFileLoader("caliper_loader_script", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _write_error(folder, kind, message, tb=""):
    record = {
        "caliper_python_loader": FORMAT_VERSION,
        "error": {"type": kind, "message": _clean_text(message, 2000), "traceback": tb[-8000:]},
    }
    try:
        with open(os.path.join(folder, "error.json"), "w", encoding="utf-8") as f:
            json.dump(record, f)
    except OSError:
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="loader_host",
        description="Runs a Caliper Python loader script and writes its outputs and manifest.json.")
    parser.add_argument("--script", required=True, help="the script that defines load(path, out)")
    parser.add_argument("--input", required=True, help="the file the script loads")
    parser.add_argument("--output", required=True, help="an empty folder for the outputs")
    args = parser.parse_args(argv)
    folder = os.path.abspath(args.output)
    if not os.path.isdir(folder) or os.listdir(folder):
        print("loader_host: --output must name an empty folder", file=sys.stderr)
        return EXIT_USAGE
    sys.dont_write_bytecode = True
    try:
        import numpy
    except ImportError as e:
        _write_error(folder, "numpy_missing", "numpy is not installed for %s (%s)" % (sys.executable, e))
        print("loader_host: numpy is not installed for %s" % sys.executable, file=sys.stderr)
        return EXIT_NO_NUMPY
    try:
        module = _load_script(args.script)
        load = getattr(module, "load", None)
        if not callable(load):
            raise TypeError("the script defines no function load(path, out)")
        writer = Writer(folder, numpy)
        load(args.input, writer)
        writer.finish()
    except BaseException as e:  # noqa: BLE001  (the host reports every failure of the script)
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        message = "%s: %s" % (type(e).__name__, e) if str(e) else type(e).__name__
        _write_error(folder, type(e).__name__, message, tb)
        return EXIT_SCRIPT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
