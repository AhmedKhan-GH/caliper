"""Private native inference entry through the managed run environment."""

import argparse
import os


def main(argv=None):
    # Remove the prompt even when argument validation rejects this request.
    prompt = os.environ.pop("CALIPER_RUN_PROMPT", None)
    parser = argparse.ArgumentParser(description="Native Inference Run")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--replication-profile", choices=("published_dallas_v1",))
    parser.add_argument("--private-output-directory")
    args = parser.parse_args(argv)
    if (args.replication_profile is None) != (args.private_output_directory is None):
        parser.error("replication-profile and private-output-directory must appear together")
    if not prompt:
        parser.error("the native runner did not supply a prompt")

    # Keep the prompt out of process arguments and descendant environments.
    # A separate value preserves checkpoint and parameter override precedence.
    from ._hosted import main as hosted_main
    forwarded = ["--model=" + args.model, "--dictionary=" + args.dictionary, "--prompt=" + prompt]
    if args.replication_profile is not None:
        forwarded.extend(["--replication-profile=" + args.replication_profile,
                          "--private-output-directory=" + args.private_output_directory])
    hosted_main(forwarded)


if __name__ == "__main__":
    main()
