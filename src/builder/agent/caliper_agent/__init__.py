"""caliper-agent: the BYOK backend of Caliper's agent pane.

The sidecar speaks the agent side of the Agent Client Protocol (ACP) version 1
on stdio, runs its own loop against the Claude API with the user's key, and
acts on Caliper only through Caliper API on loopback. The design is in
docs/design/byok-agent.md.
"""

__version__ = "0.1.0"
