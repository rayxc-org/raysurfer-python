"""High-level Agent runner with automatic code caching and AI-driven quality scoring."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from types import TracebackType
from typing import Literal, TypedDict

from raysurfer.client import DEFAULT_BASE_URL, AsyncRaySurfer
from raysurfer.sdk_client import RaysurferClient
from raysurfer.types import SnipsDesired

try:
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage
except ImportError:
    ClaudeAgentOptions = None  # type: ignore[assignment,misc]
    ResultMessage = None  # type: ignore[assignment,misc]


class MessageParam(TypedDict):
    """Anthropic-compatible message format for chat history."""

    role: Literal["user", "assistant"]
    content: str


@dataclass
class RunResult:
    """Result of a single conversation execution."""

    run_id: str
    query: str
    succeeded: bool
    messages: list[object] = field(default_factory=list)
    code_used: list[dict[str, str]] = field(default_factory=list)


class Agent:
    """
    Conversation runner with automatic code caching and AI-driven quality scoring.

    Wraps raysurfer.search() and raysurfer.upload() into a single run() call.
    Accepts Anthropic-typed chat history, searches for proven cached code,
    executes via Claude with that code injected, and stores any new code
    generated. AI automatically scores code quality — no manual feedback needed.

    Usage:
        from raysurfer import Agent

        async with Agent(org_id="acme-corp", user_id="user_123") as agent:
            result = await agent.run(
                messages=[
                    {"role": "user", "content": "Generate a quarterly report from our sales data"},
                ],
            )
    """

    def __init__(
        self,
        *,
        org_id: str | None = None,
        user_id: str | None = None,
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
            user_id: User identifier for scoped code retrieval.
            api_key: RaySurfer API key (or set RAYSURFER_API_KEY env var).
            base_url: API base URL.
            agent_id: Optional agent identifier for agent-scoped isolation.
            allowed_tools: Tools the agent can use (default: Read, Write, Bash).
            system_prompt: System prompt for the underlying Claude agent.
            model: Model to use (default: claude-opus-4-6).
        """
        self._org_id = org_id
        self._user_id = user_id
        self._api_key = api_key
        self._base_url = base_url
        self._agent_id = agent_id
        self._allowed_tools = allowed_tools or ["Read", "Write", "Bash"]
        self._system_prompt = system_prompt or "You are a helpful assistant."
        self._model = model
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
        messages: list[MessageParam],
        user_id: str | None = None,
        org_id: str | None = None,
    ) -> RunResult:
        """
        Process a conversation with automatic code caching.

        Each conversation goes through the raysurfer loop:
        1. raysurfer.search() — find proven cached code matching the query
        2. Execute via Claude with cached code injected as context
        3. raysurfer.upload() — store any new code generated for future reuse

        AI automatically scores code quality on execution — no manual
        feedback needed.

        Args:
            messages: Anthropic-typed chat history (list of role/content dicts).
            user_id: Override the agent-level user_id for this run.
            org_id: Override the agent-level org_id for this run.
        """
        if ClaudeAgentOptions is None:
            raise ImportError(
                "claude_agent_sdk is required for Agent.run(). "
                "Install it: uv pip install claude-agent-sdk"
            )

        effective_org_id = org_id or self._org_id

        if effective_org_id and self._raysurfer:
            self._raysurfer.organization_id = effective_org_id

        # Extract last user message as the search query
        query = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                query = msg["content"]
                break

        if not query:
            raise ValueError("Messages must contain at least one user message.")

        return await self._run_single(query, user_id or self._user_id)

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

        response_messages: list[object] = []
        succeeded = False

        async with client:
            await client.query(query)
            async for msg in client.response():
                response_messages.append(msg)
                if ResultMessage is not None and isinstance(msg, ResultMessage):
                    if msg.subtype == "success":
                        succeeded = True

            # Capture which cached snippets were used
            code_used = list(client._cached_code_blocks)

        return RunResult(
            run_id=run_id,
            query=query,
            succeeded=succeeded,
            messages=response_messages,
            code_used=code_used,
        )
