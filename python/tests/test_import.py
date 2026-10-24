import subprocess
import sys


def test_import_and_views():
    import caliper
    assert caliper.HAVE_VIEWS is True, "built without the embed v1.2 views"
    assert caliper.EMBED_API_VERSION >= 1
    assert caliper.VIEW_FLAG_WAIT_GPU and caliper.VIEW_FLAG_RAW


def test_import_is_lazy():
    # Importing the package must not load the native module (GLFW/AppKit/Metal).
    code = "import sys, caliper; print('caliper._native' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_torch_coexists():
    # The point of the torch-free libcaliper: Python's torch and the module share a process.
    import torch
    import caliper
    assert caliper._load_native() is not None
    assert torch.__version__
