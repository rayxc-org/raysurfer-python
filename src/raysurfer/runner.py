"""High-level Agent runner for batch query execution with retroactive voting."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from types import TracebackType

from raysurfer.client import DEFAULT_BASE_URL, AsyncRaySurfer
from raysurfer.sdk_client import RaysurferClient
from raysurfer.types import SnipsDesired

try:
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage
except ImportError:
    ClaudeAgentOptions = None  # type: ignore[assignment,misc]
    ResultMessage = None  # type: ignore[assignment,misc]


@dataclass
class RunResult:
    """Result of a single query execution with metadata for retroactive voting."""

    run_id: str
    query: str
    succeeded: bool
    messages: list[object] = field(default_factory=list)
    code_used: list[dict[str, str]] = field(default_factory=list)


class Agent:
    """
    Batch query runner with automatic code persistence and retroactive voting.

    Wraps raysurfer.search() and raysurfer.upload() into a single run() call.
    For each query, searches for proven cached code, executes via Claude with
    that code injected, and stores any new code generated. Tracks which cached
    snippets contributed to each result so user feedback can retroactively
    promote or demote them.

    Usage:
        from raysurfer import Agent

        async with Agent(org_id="acme-corp") as agent:
            results = await agent.run(
                ["Generate quarterly report", "Summarize sales data"],
                user_id="user_123",
            )

            # User liked the first result, disliked the second
            await agent.feedback(results[0].run_id, satisfied=True)
            await agent.feedback(results[1].run_id, satisfied=False)
    """

    def __init__(
        self,
        *,
        org_id: str | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        agent_id: str | None = None,
        allowed_tools: list[str] | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> None:
        """
        Initialize an Agent runner.

        Args:
            org_id: Organization ID for shared code library across the team.
            api_key: RaySurfer API key (or set RAYSURFER_API_KEY env var).
            base_url: API base URL.
            agent_id: Optional agent identifier for agent-scoped isolation.
            allowed_tools: Tools the agent can use (default: Read, Write, Bash).
            system_prompt: System prompt for the underlying Claude agent.
            model: Model to use (default: claude-opus-4-6).
        """
        self._org_id = org_id
        self._api_key = api_key
        self._base_url = base_url
        self._agent_id = agent_id
        self._allowed_tools = allowed_tools or ["Read", "Write", "Bash"]
        self._system_prompt = system_prompt or "You are a helpful assistant."
        self._model = model
        self._run_log: dict[str, RunResult] = {}
        self._raysurfer: AsyncRaySurfer | None = None

    async def __aenter__(self) -> Agent:
        """Initialize the underlying raysurfer client."""
        snips_desired = SnipsDesired.COMPANY if self._org_id else None
        self._raysurfer = AsyncRaySurfer(
            api_key=self._api_key,
            base_url=self._base_url,
            organization_id=self._org_id,
            snips_desired=snips_desired,
            agent_id=self._agent_id,
        )
        await self._raysurfer.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up resources."""
        if self._raysurfer:
            await self._raysurfer.__aexit__(exc_type, exc_val, exc_tb)

    async def run(
        self,
        user_queries: list[str],
        user_id: str | None = None,
        org_id: str | None = None,
    ) -> list[RunResult]:
        """
        Process a batch of user queries with automatic code caching.

        Each query goes through the raysurfer loop:
        1. raysurfer.search() — find proven cached code matching the query
        2. Execute via Claude with cached code injected as context
        3. raysurfer.upload() — store any new code generated for future reuse

        Returns RunResult objects with run_ids for retroactive feedback.

        Args:
            user_queries: List of tasks/queries to process.
            user_id: User identifier for scoped code retrieval.
            org_id: Override the agent-level org_id for this batch.
        """
        if ClaudeAgentOptions is None:
            raise ImportError(
                "claude_agent_sdk is required for Agent.run(). "
                "Install it: uv pip install claude-agent-sdk"
            )

        effective_org_id = org_id or self._org_id

        if effective_org_id and self._raysurfer:
            self._raysurfer.organization_id = effective_org_id

        results: list[RunResult] = []
        for query in user_queries:
            result = await self._run_single(query, user_id)
            results.append(result)

        return results

    async def _run_single(
        self,
        query: str,
        user_id: str | None,
    ) -> RunResult:
        """Execute a single query and track code lineage."""
        run_id = str(uuid.uuid4())

        options = ClaudeAgentOptions(
            allowed_tools=self._allowed_tools,
            system_prompt=self._system_prompt,
        )
        if self._model:
            options.model = self._model

        client = RaysurferClient(
            options=options,
            agent_id=self._agent_id,
        )

        messages: list[object] = []
        succeeded = False

        async with client:
            await client.query(query)
            async for msg in client.response():
                messages.append(msg)
                if ResultMessage is not None and isinstance(msg, ResultMessage):
                    if msg.subtype == "success":
                        succeeded = True

            # Capture which cached snippets were used for retroactive voting
            code_used = list(client._cached_code_blocks)

        result = RunResult(
            run_id=run_id,
            query=query,
            succeeded=succeeded,
            messages=messages,
            code_used=code_used,
        )
        self._run_log[run_id] = result
        return result

    async def feedback(self, run_id: str, satisfied: bool) -> None:
        """
        Retroactively vote on all code that contributed to a run result.

        When a user expresses satisfaction, every cached snippet that was
        retrieved and used during that run gets a thumbs up. When dissatisfied,
        they get thumbs down. Over time this promotes code that makes users
        happy and demotes code that doesn't.

        New code generated during the run is already AI-voted at upload time.
        This method specifically handles the retroactive user signal on cached
        code that was reused.

        Args:
            run_id: The run_id from a RunResult.
            satisfied: True for thumbs up, False for thumbs down.
        """
        result = self._run_log.get(run_id)
        if result is None:
            raise ValueError(
                f"Unknown run_id: {run_id}. "
                "Run IDs are only valid within the same Agent session."
            )

        if not self._raysurfer:
            raise RuntimeError("Agent is not initialized. Use 'async with Agent() as agent:'")

        tasks = []
        for block in result.code_used:
            tasks.append(
                self._raysurfer.vote_code_snip(
                    task=result.query,
                    code_block_id=block["code_block_id"],
                    code_block_name=block.get("filename", ""),
                    code_block_description=block.get("description", ""),
                    succeeded=satisfied,
                )
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
