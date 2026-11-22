"""The acceptance test of instrument building (design/instrument-interface.md, Acceptance Test).

The hidden app with no script, and caliper-mcp as a child: an agent builds the training debugger's instrument
through the MCP tools on a managed run of train.py. Not written yet: the test exits 77, which ctest reports as
skipped.
"""

import sys

if __name__ == "__main__":
    sys.exit(77)
