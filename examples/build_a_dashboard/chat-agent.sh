#!/bin/sh
# chat-agent.sh: caliper-agent, Caliper's own ACP agent, on a model that Ollama
# serves on this machine, for the side chat. In the chat's agent picker choose
# Custom command and give this script's absolute path as the command line; no
# account or key is used, and the script reads none.
#
# It runs the agent from this repository's source (src/builder/agent), so
# nothing is installed, with:
#
#   CALIPER_AGENT_PYTHON  the Python that has caliper-agent's dependencies
#                         (src/builder/agent/pyproject.toml); by default the
#                         repository's .venv/bin/python when there is one,
#                         else python3 on PATH, unless that is the stub of a
#                         Mac without the Command Line Tools
#   CALIPER_CHAT_MODEL    the model (default qwen3.8:27b)
#   CALIPER_CHAT_CONTEXT  the context length in tokens (default: the Ollama
#                         preset's own for the chat, 32,768)
#   CALIPER_CHAT_BASE_URL Ollama's address (default http://127.0.0.1:11434/v1)
#
# An app started from the Finder gives its agents a PATH of only the system's
# folders, so set the variables in the command line itself when you need them:
#
#   /usr/bin/env CALIPER_AGENT_PYTHON=/path/to/python /path/to/chat-agent.sh
#
# Its own arguments are passed on to caliper-agent. It works from any working
# folder; the side chat starts it in the workspace.

here=$(cd "$(dirname "$0")" && pwd -P) || exit 2
repo=$(cd "$here/../.." && pwd -P) || exit 2
source="$repo/src/builder/agent"
if [ ! -f "$source/caliper_agent/__main__.py" ]; then
    echo "chat-agent.sh: caliper-agent's source is not at $source" >&2
    exit 2
fi

python=${CALIPER_AGENT_PYTHON:-}
if [ -z "$python" ]; then
    if [ -x "$repo/.venv/bin/python" ]; then
        python="$repo/.venv/bin/python"
    else
        python=$(command -v python3 2> /dev/null)
        # On a Mac without the Command Line Tools, /usr/bin/python3 is a stub that opens an install
        # dialog; say what to do instead of starting it.
        if [ "$(uname)" = Darwin ] && [ "$python" = /usr/bin/python3 ] && ! /usr/bin/xcode-select -p > /dev/null 2>&1; then
            echo "chat-agent.sh: no Python for caliper-agent; set CALIPER_AGENT_PYTHON to one with its dependencies" >&2
            exit 2
        fi
        python=${python:-python3}
    fi
fi
if ! command -v "$python" > /dev/null 2>&1; then
    echo "chat-agent.sh: no Python at $python; set CALIPER_AGENT_PYTHON to one with caliper-agent's dependencies" >&2
    exit 2
fi
if ! PYTHONPATH="$source" "$python" -P -c "import caliper_agent, pydantic_ai, openai, anthropic" > /dev/null 2>&1; then
    echo "chat-agent.sh: $python lacks caliper-agent's dependencies (src/builder/agent/pyproject.toml); set CALIPER_AGENT_PYTHON" >&2
    exit 2
fi

set -- --preset ollama --base-url "${CALIPER_CHAT_BASE_URL:-http://127.0.0.1:11434/v1}" \
    --model "${CALIPER_CHAT_MODEL:-qwen3.8:27b}" ${CALIPER_CHAT_CONTEXT:+--context-length "$CALIPER_CHAT_CONTEXT"} "$@"
export PYTHONPATH="$source" PYTHONDONTWRITEBYTECODE=1
exec "$python" -P -m caliper_agent "$@"
