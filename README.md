# RaySurfer Python SDK

[Website](https://www.raysurfer.com) · [Docs](https://docs.raysurfer.com) · [Dashboard](https://www.raysurfer.com/dashboard/api-keys)

<!-- Old: LLM output caching for AI agents. Retrieve proven code instead of regenerating it. -->
<!-- Old: Code reputation layer for AI agents. Let agents re-use generated code vs running 30 serial tools or generating code per execution. -->
AI Maintained Skills for Vertical Agents. Re-use verified code from prior runs rather than serial tool calls or generating code per execution.

## Installation

```bash
pip install raysurfer
```

## Setup

Set your API key:

```bash
export RAYSURFER_API_KEY=your_api_key_here
```

Get your key from the [dashboard](https://www.raysurfer.com/dashboard/api-keys).

## Low-Level API

For custom integrations, use the `RaySurfer` client directly with any LLM provider.

### Complete Example with Anthropic API

```python
import anthropic
from raysurfer import RaySurfer
from raysurfer.types import FileWritten, LogFile

client = RaySurfer(api_key="your_raysurfer_api_key")
task = "Fetch GitHub trending repos"

# 1. Search for cached code matching a task
result = client.search(
    task=task,
    top_k=5,
    min_verdict_score=0.3,
)

for match in result.matches:
    print(f"{match.code_block.name}: {match.combined_score}")
    print(f"  Source: {match.code_block.source[:80]}...")

# 2. Upload a new code file after execution
file = FileWritten(path="fetch_repos.py", content="def fetch(): ...")
client.upload_new_code_snip(
    task=task,
    file_written=file,
    succeeded=True,
    execution_logs="Fetched 10 trending repos successfully",
    dependencies={"httpx": "0.27.0", "pydantic": "2.5.0"},
)

# 2b. Bulk upload prompts/logs/code for sandboxed grading
logs = [LogFile(path="logs/run.log", content="Task completed", encoding="utf-8")]
client.upload_bulk_code_snips(
    prompts=["Build a CLI tool", "Add CSV support"],
    files_written=[FileWritten(path="cli.py", content="def main(): ...")],
    log_files=logs,
)

# 3. Vote on whether a cached snippet was useful
client.vote_code_snip(
    task=task,
    code_block_id=result.matches[0].code_block.id,
    code_block_name=result.matches[0].code_block.name,
    code_block_description=result.matches[0].code_block.description,
    succeeded=True,
)
```

### Async Version

```python
import anthropic
from raysurfer import AsyncRaySurfer
from raysurfer.types import FileWritten

async with AsyncRaySurfer(api_key="your_api_key") as client:
    # 1. Search for cached code
    result = await client.search(task="Fetch GitHub trending repos")

    for match in result.matches:
        print(f"{match.code_block.name}: {match.combined_score}")

    # 2. Upload a new code file after execution
    file = FileWritten(path="fetch_repos.py", content="def fetch(): ...")
    await client.upload_new_code_snip(
        task="Fetch GitHub trending repos",
        file_written=file,
        succeeded=True,
        execution_logs="Fetched 10 trending repos successfully",
    )

    # 3. Vote on snippet manually
    await client.vote_code_snip(
        task="Fetch GitHub trending repos",
        code_block_id=result.matches[0].code_block.id,
        code_block_name=result.matches[0].code_block.name,
        code_block_description=result.matches[0].code_block.description,
        succeeded=True,
    )
```

### Client Options

```python
client = RaySurfer(
    api_key="your_api_key",
    base_url="https://api.raysurfer.com",  # optional
    timeout=30,                             # optional, in seconds
    organization_id="org_xxx",              # optional, for team namespacing
    workspace_id="ws_xxx",                  # optional, for enterprise namespacing
    snips_desired="company",                # optional, snippet scope
    public_snips=True,                      # optional, include community snippets
)
```

### Response Fields

The `search()` response includes:

| Field | Type | Description |
|-------|------|-------------|
| `matches` | `list[SearchMatch]` | Matching code blocks with scoring |
| `total_found` | `int` | Total matches found |
| `cache_hit` | `bool` | Whether results were from cache |

Each `SearchMatch` contains `code_block` (with `id`, `name`,
`source`, `description`, `entrypoint`, `language`, `dependencies`),
`combined_score`, `vector_score`, `verdict_score`, `thumbs_up`,
`thumbs_down`, `filename`, and `entrypoint`.

### Store a Code Block with Full Metadata

```python
result = client.store_code_block(
    name="GitHub User Fetcher",
    source="def fetch_user(username): ...",
    entrypoint="fetch_user",
    language="python",
    description="Fetches user data from GitHub API",
    tags=["github", "api", "user"],
    dependencies={"httpx": "0.27.0", "pydantic": "2.5.0"},
)
```

### Retrieve Few-Shot Examples

```python
examples = client.get_few_shot_examples(task="Parse CSV files", k=3)

for ex in examples:
    print(f"Task: {ex.task}")
    print(f"Code: {ex.code_snippet}")
```

### Retrieve Task Patterns

```python
patterns = client.get_task_patterns(
    task="API integration",
    min_thumbs_up=5,
    top_k=20,
)

for p in patterns:
    print(f"{p.task_pattern} -> {p.code_block_name}")
```

### User-Provided Votes

Instead of relying on AI voting, provide your own votes:

```python
# Single upload with your own vote (AI voting is skipped)
client.upload_new_code_snip(
    task="Fetch GitHub trending repos",
    file_written=file,
    succeeded=True,
    user_vote=1,  # 1 = thumbs up, -1 = thumbs down
)

# Bulk upload with per-file votes (AI grading is skipped)
client.upload_bulk_code_snips(
    prompts=["Build a CLI tool", "Add CSV support"],
    files_written=files,
    log_files=logs,
    user_votes={
        "app.py": 1,     # thumbs up
        "utils.py": -1,  # thumbs down
    },
)
```

### Method Reference

| Method | Description |
|--------|-------------|
| `search(task, top_k, min_verdict_score, prefer_complete, input_schema)` | Search for cached code snippets |
| `get_code_snips(task, top_k, min_verdict_score)` | Retrieve cached code snippets by semantic search |
| `retrieve_best(task, top_k, min_verdict_score)` | Retrieve the single best match |
| `get_few_shot_examples(task, k)` | Retrieve few-shot examples for code generation prompting |
| `get_task_patterns(task, min_thumbs_up, top_k)` | Retrieve proven task-to-code mappings |
| `store_code_block(name, source, entrypoint, language, description, tags, dependencies, ...)` | Store a code block with full metadata |
| `upload_new_code_snip(task, file_written, succeeded, use_raysurfer_ai_voting, user_vote, execution_logs, dependencies)` | Store a single code file with optional dependency versions |
| `upload_bulk_code_snips(prompts, files_written, log_files, use_raysurfer_ai_voting, user_votes)` | Bulk upload for grading (AI votes by default, or provide per-file votes) |
| `vote_code_snip(task, code_block_id, name, description, succeeded)` | Vote on snippet usefulness |

### Exceptions

Both sync and async clients include built-in retry logic with exponential backoff for transient failures (429, 5xx, network errors).

| Exception | Description |
|-----------|-------------|
| `RaySurferError` | Base exception for all Raysurfer errors |
| `APIError` | API returned an error response (includes `status_code`) |
| `AuthenticationError` | API key is invalid or missing |
| `CacheUnavailableError` | Cache backend is unreachable |
| `RateLimitError` | Rate limit exceeded after retries (includes `retry_after`) |
| `ValidationError` | Request validation failed (includes `field`) |

```python
from raysurfer import RaySurfer
from raysurfer.exceptions import RateLimitError

client = RaySurfer(api_key="your_api_key")

try:
    result = client.get_code_snips(task="Fetch GitHub repos")
except RateLimitError as e:
    print(f"Rate limited after retries: {e}")
    if e.retry_after:
        print(f"Try again in {e.retry_after}s")
```

---

## Claude Agent SDK Drop-in

Swap your client class and method names. Options come directly from `claude_agent_sdk`:

```python
# Before
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

# After
from raysurfer import RaysurferClient
from claude_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(
    allowed_tools=["Read", "Write", "Bash"],
    system_prompt="You are a helpful assistant.",
)

async with RaysurferClient(options) as client:
    await client.query("Generate quarterly report")
    async for msg in client.response():
        print(msg)
```

### Method Mapping

| Claude SDK | Raysurfer |
|------------|-----------|
| `ClaudeSDKClient(options)` | `RaysurferClient(options)` |
| `await client.query(prompt)` | `await client.query(prompt)` |
| `client.receive_response()` | `client.response()` |

### Full Example

```python
import asyncio
import os
from raysurfer import RaysurferClient
from claude_agent_sdk import ClaudeAgentOptions

os.environ["RAYSURFER_API_KEY"] = "your_api_key"

async def main():
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Bash"],
        system_prompt="You are a helpful assistant.",
    )

    async with RaysurferClient(options) as client:
        # First run: generates and caches code
        await client.query("Fetch GitHub trending repos")
        async for msg in client.response():
            print(msg)

        # Second run: retrieves from cache (instant)
        await client.query("Fetch GitHub trending repos")
        async for msg in client.response():
            print(msg)

asyncio.run(main())
```

### Without Caching

If `RAYSURFER_API_KEY` is not set, `RaysurferClient` behaves exactly like `ClaudeSDKClient` — no caching, just a pass-through wrapper.

## Snippet Retrieval Scope

Control which cached snippets are retrieved using `snips_desired`:

```python
from raysurfer import RaysurferClient
from claude_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(
    allowed_tools=["Read", "Write", "Bash"],
)

# Include company-level snippets
client = RaysurferClient(
    options,
    snips_desired="company",  # Company-level snippets (Team/Enterprise)
)

# Enterprise: Retrieve client-specific snippets only
client = RaysurferClient(
    options,
    snips_desired="client",   # Client workspace snippets (Enterprise only)
)
```

| Configuration | Required Tier |
|--------------|---------------|
| `snips_desired="company"` | TEAM or ENTERPRISE |
| `snips_desired="client"` | ENTERPRISE only |

## Public Snippets

Include community public snippets (crawled from GitHub) in
retrieval results alongside your private snippets:

```python
# High-level
client = RaysurferClient(options, public_snips=True)

# Low-level
client = RaySurfer(api_key="...", public_snips=True)
```

## Programmatic Tool Calling

Register local tools, then either:
1) pass in `user_code` (primary mode), or
2) generate code inside the sandbox with your own provider key + prompt (optional mode).

```python
import asyncio
from raysurfer import AsyncRaySurfer

async def main():
    rs = AsyncRaySurfer(api_key="your_api_key")

    @rs.tool
    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    @rs.tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers together."""
        return a * b

    user_code = """
intermediate = add(5, 3)
final = multiply(intermediate, 2)
print(final)
"""
    result = await rs.execute(
        "Add 5 and 3, then multiply the result by 2",
        user_code=user_code,
    )
    print(result.result)       # "16"
    print(result.tool_calls)   # [ToolCallRecord(tool_name='add', ...), ToolCallRecord(tool_name='multiply', ...)]
    print(result.cache_hit)    # False (reserved field for execute)

asyncio.run(main())
```

The `@rs.tool` decorator introspects your function signature to build a JSON schema. Both sync and async callbacks are supported.

### How It Works

1. SDK connects a WebSocket to the server for tool call routing
2. Your app sends either `user_code` (primary mode) or `codegen_*` inputs (optional mode) to `/api/execute/run`
3. Code runs in a sandboxed environment — tool calls are routed back to your local functions via WebSocket
4. Results are returned with full tool call history

### Execute Options

```python
result = await rs.execute(
    "Your task description",
    user_code="print(add(1, 2))",  # Primary mode
    timeout=300,                    # Max execution time in seconds (default 300)
)

# Optional mode: generate code in sandbox using your own key + prompt
result = await rs.execute(
    "Your task description",
    codegen_api_key="your_anthropic_key",
    codegen_prompt="Write Python code that uses add(a, b) and prints the result for 2 + 3.",
    codegen_model="claude-opus-4-6",
)
```

### ExecuteResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `execution_id` | `str` | Unique execution identifier |
| `result` | `str \| None` | Stdout output from the script |
| `exit_code` | `int` | Process exit code (0 = success) |
| `duration_ms` | `int` | Total execution time |
| `cache_hit` | `bool` | Reserved field (currently always `False` for execute) |
| `error` | `str \| None` | Error message if exit_code != 0 |
| `tool_calls` | `list[ToolCallRecord]` | All tool calls made during execution |

## License

MIT
