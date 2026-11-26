"""Private native inference entry through the managed run environment."""
from __future__ import annotations

import argparse
import os


def main(argv=None) -> None:
    prompt = os.environ.pop("CALIPER_RUN_PROMPT", None)
    parser = argparse.ArgumentParser(description="Native Inference Run")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dictionary", required=True)
    args = parser.parse_args(argv)
    if not prompt:
        parser.error("the native runner did not supply a prompt")
    from ._hosted import main as hosted_main
    hosted_main(["--model=" + args.model, "--dictionary=" + args.dictionary, "--prompt=" + prompt])


if __name__ == "__main__":
    main()
