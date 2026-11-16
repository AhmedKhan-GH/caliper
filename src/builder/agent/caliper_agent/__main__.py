"""`python -I -m caliper_agent` runs the sidecar. Use -I (or -P) so that the
working folder, which is the user's workspace, is never put on the import path."""

from caliper_agent.cli import main

raise SystemExit(main())
