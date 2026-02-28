"""Tests for RaysurferClient default agent compatibility behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from raysurfer.sdk_types import CodeFile, GetCodeFilesResponse
from raysurfer.types import FileWritten


@dataclass
class FakeClaudeAgentOptions:
    """Minimal ClaudeAgentOptions replacement for unit tests with mocked SDK."""

    tools: dict | None = None
    allowed_tools: list[str] = field(default_factory=list)
    system_prompt: str | dict | None = None
    mcp_servers: dict = field(default_factory=dict)
    permission_mode: str | None = None
    continue_conversation: bool = False
    resume: str | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    disallowed_tools: list[str] = field(default_factory=list)
    model: str | None = None
    fallback_model: str | None = None
    betas: list[str] = field(default_factory=list)
    permission_prompt_tool_name: str | None = None
    cwd: str | Path | None = None
    cli_path: str | Path | None = None
    settings: str | None = None
    add_dirs: list[str | Path] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    extra_args: dict[str, str | None] = field(default_factory=dict)
    max_buffer_size: int | None = None
    debug_stderr: object | None = None
    stderr: object | None = None
    can_use_tool: object | None = None
    hooks: dict | None = None
    user: str | None = None
    include_partial_messages: bool = False
    fork_session: bool = False
    agents: dict | None = None
    setting_sources: list[str] | None = None
    sandbox: dict | None = None
    plugins: list = field(default_factory=list)
    max_thinking_tokens: int | None = None
    output_format: dict | None = None
    enable_file_checkpointing: bool = False


@pytest.fixture
def sdk_module(monkeypatch):
    """Patch sdk_client module to use a deterministic options class in tests."""
    import raysurfer.sdk_client as sdk_client

    monkeypatch.setattr(sdk_client, "ClaudeAgentOptions", FakeClaudeAgentOptions)
    return sdk_client


def _make_code_file() -> CodeFile:
    """Create a minimal code file payload for cache augmentation tests."""
    return CodeFile(
        code_block_id="cb_compat_001",
        filename="solution.py",
        source="def solve(x: int) -> int:\n    return x + 1\n",
        entrypoint="solve",
        description="Simple increment helper",
        language="python",
        score=0.9,
        thumbs_up=5,
        thumbs_down=0,
    )


def test_defaults_apply_when_options_not_provided(sdk_module) -> None:
    """Client should inject compatibility tools + sandbox defaults with no options."""
    client = sdk_module.RaysurferClient()

    assert client._options.tools == {"type": "preset", "preset": "claude_code"}
    assert isinstance(client._options.sandbox, dict)
    assert client._options.sandbox.get("enabled") is True
    assert client._options.sandbox.get("autoAllowBashIfSandboxed") is True


def test_custom_tools_and_sandbox_are_preserved(sdk_module) -> None:
    """Explicit user tool/sandbox config should not be overridden."""
    options = FakeClaudeAgentOptions(
        allowed_tools=["Read", "Write"],
        sandbox={"enabled": False, "allowUnsandboxedCommands": True},
    )

    client = sdk_module.RaysurferClient(options=options)

    assert client._options.allowed_tools == ["Read", "Write"]
    assert client._options.tools is None
    assert isinstance(client._options.sandbox, dict)
    assert client._options.sandbox.get("enabled") is False
    assert client._options.sandbox.get("allowUnsandboxedCommands") is True
    # Default key is still merged for partial configs.
    assert client._options.sandbox.get("autoAllowBashIfSandboxed") is True


@pytest.mark.asyncio
async def test_augment_preserves_tools_and_new_sdk_fields(tmp_path) -> None:
    """Cache prompt augmentation should preserve all Claude options, including tools."""
    import raysurfer.sdk_client as sdk_client

    options = FakeClaudeAgentOptions(
        tools={"type": "preset", "preset": "claude_code"},
        cwd=tmp_path,
        system_prompt="You are a coding agent.",
        max_thinking_tokens=2048,
    )
    # Patch inside this test because it runs in asyncio context.
    original_cls = sdk_client.ClaudeAgentOptions
    sdk_client.ClaudeAgentOptions = FakeClaudeAgentOptions
    client = sdk_client.RaysurferClient(options=options)
    client._cache_enabled = True

    mock_rs = AsyncMock()
    mock_rs.get_code_files = AsyncMock(
        return_value=GetCodeFilesResponse(
            files=[_make_code_file()],
            task="Implement solver",
            total_found=1,
            add_to_llm_prompt="\nUse cached file: solution.py",
        )
    )
    client._raysurfer = mock_rs

    try:
        augmented = await client._augment_options_with_cache("Implement solver")
    finally:
        sdk_client.ClaudeAgentOptions = original_cls

    assert augmented.tools == {"type": "preset", "preset": "claude_code"}
    assert augmented.max_thinking_tokens == 2048
    assert isinstance(augmented.sandbox, dict)
    assert augmented.sandbox.get("enabled") is True
    assert "Use cached file: solution.py" in str(augmented.system_prompt)


@pytest.mark.asyncio
async def test_cache_upload_enables_per_function_reputation(sdk_module, tmp_path) -> None:
    """Auto-upload path should always include per-function reputation extraction."""
    options = FakeClaudeAgentOptions(cwd=tmp_path)
    client = sdk_module.RaysurferClient(options=options)
    client._current_query = "Generate a parser"
    client._task_succeeded = True
    client._parse_this_run_for_ai_voting = True
    client._generated_files = [FileWritten(path="parser.py", content="def parse(v):\n    return v\n")]
    client._execution_logs = ["parser ran successfully"]

    mock_rs = AsyncMock()
    mock_rs.upload_new_code_snip = AsyncMock(return_value=SimpleNamespace(code_blocks_stored=1))
    client._raysurfer = mock_rs

    await client._upload_to_cache()

    mock_rs.upload_new_code_snip.assert_called_once()
    call_kwargs = mock_rs.upload_new_code_snip.call_args.kwargs
    assert call_kwargs["per_function_reputation"] is True
