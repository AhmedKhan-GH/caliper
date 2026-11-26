"""Durable tensor and JSON writes for dictionary evaluation."""
from __future__ import annotations

import json
import os
from pathlib import Path

from safetensors.torch import save_file
from caliper.interp._dictionary import dictionary_fingerprint


def _save_tensors(values, path):
    save_file(values, path)
    # Windows file flush requires a handle with write access.
    with Path(path).open("r+b") as source:
        os.fsync(source.fileno())


def write_json(path, value):
    """Replace a JSON file only after the complete new file reaches disk."""
    path = Path(path)
    temporary = path.with_name(path.name + ".new")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
