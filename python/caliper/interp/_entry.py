"""Private native inference entry through the managed run environment."""

import argparse
import os


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native Inference Run")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dictionary", required=True)
    args = parser.parse_args(argv)
    prompt = os.environ.pop("CALIPER_RUN_PROMPT", None)
    if not prompt:
        parser.error("the native runner did not supply a prompt")

    # Keep the prompt out of process arguments and descendant environments.
    # A separate value preserves checkpoint and parameter override precedence.
    from ._hosted import main as hosted_main
    hosted_main(["--model=" + args.model, "--dictionary=" + args.dictionary,
                 "--prompt=" + prompt])


if __name__ == "__main__":
    main()
