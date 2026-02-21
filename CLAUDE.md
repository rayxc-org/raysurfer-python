# Raysurfer Python SDK

Drop-in replacement for Claude Agent SDK with automatic code caching.

## Quick Start

Two modes: high-level drop-in replacement for Claude Agent SDK, and low-level API for snippet pull/upload/vote.

```python
from raysurfer import RaysurferClient
from claude_agent_sdk import ClaudeAgentOptions
```

Set the `RAYSURFER_API_KEY` environment variable to enable caching.

## How It Works

1. **On query**: Automatically retrieves relevant cached code and injects it into the system prompt
2. **On success**: Automatically uploads generated code to the cache for future reuse
3. **No caching?**: If `RAYSURFER_API_KEY` isn't set, behaves exactly like the original SDK

## Example

```python
import os
from raysurfer import RaysurferClient
from claude_agent_sdk import ClaudeAgentOptions

os.environ["RAYSURFER_API_KEY"] = "rs_..."

options = ClaudeAgentOptions(
    allowed_tools=["Read", "Write", "Bash"],
    system_prompt="You are a helpful assistant.",
)

async with RaysurferClient(options) as client:
    await client.query("Fetch data from GitHub API")
    async for msg in client.response():
        print(msg)
```

## Direct API Access

For advanced use cases, use the low-level client directly:

```python
from raysurfer import AsyncRaySurfer
from raysurfer.types import FileWritten

async with AsyncRaySurfer(api_key="rs_...") as client:
    # Search: retrieve cached code by user query
    result = await client.search(task="Fetch GitHub user data")

    # Upload: store a code file + logs + query (voting triggered by default)
    await client.upload(
        task="Fetch GitHub user data",
        file_written=FileWritten(path="fetcher.py", content="def fetch(): ..."),
        succeeded=True,
        execution_logs="Fetched user data successfully",
    )
```

## Package Managers

Use `uv` for all Python operations.

## Before Completing Tasks

Run `uv run ruff check .` before marking a task as completed or pushing to git/PyPI.

## Documentation Sync

When making changes to this package, check `docs/` at the repo root to see if documentation needs updating.
