"""caliper-agent: the widget agent of Caliper.

The sidecar speaks the agent side of the Agent Client Protocol (ACP) version 1
on stdio, runs its agent loop on Pydantic AI, against the Claude API with the
user's key or any OpenAI-compatible model server, and acts on Caliper only
through Caliper API on loopback. The design is in design/byok-agent.md.
"""

__version__ = "0.1.0"
