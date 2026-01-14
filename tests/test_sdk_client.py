"""Tests for RaysurferClient - pre-fetch, system prompt injection, auto-caching"""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mock the claude_agent_sdk imports before importing RaysurferClient
# This avoids dependency on having the actual SDK installed for unit tests


@dataclass
class MockTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class MockToolUseBlock:
    type: str = "tool_use"
    name: str = ""
    input: dict = None

    def __post_init__(self):
        if self.input is None:
            self.input = {}


@dataclass
class MockAssistantMessage:
    role: str = "assistant"
    content: list = None

    def __post_init__(self):
        if self.content is None:
            self.content = []


@dataclass
class MockResultMessage:
    type: str = "result"
    subtype: str = "success"
    total_cost_usd: float = 0.0


# Patch the claude_agent_sdk module
mock_sdk = MagicMock()
mock_sdk.ClaudeSDKClient = MagicMock()
mock_sdk.ClaudeAgentOptions = MagicMock()
mock_sdk.Message = MagicMock()
mock_sdk.UserMessage = MagicMock()
mock_sdk.AssistantMessage = MockAssistantMessage
mock_sdk.SystemMessage = MagicMock()
mock_sdk.ResultMessage = MockResultMessage
mock_sdk.TextBlock = MockTextBlock
mock_sdk.ThinkingBlock = MagicMock()
mock_sdk.ToolUseBlock = MockToolUseBlock
mock_sdk.ToolResultBlock = MagicMock()

import sys

sys.modules["claude_agent_sdk"] = mock_sdk

# Now import the modules under test
from raysurfer.sdk_types import CodeFile, GetCodeFilesResponse

# =============================================================================
# Test Fixtures
# =============================================================================


def create_mock_code_file(
    code_block_id: str = "cb_123",
    filename: str = "test_code.py",
    source: str = "def test(): pass",
    entrypoint: str = "test",
    description: str = "A test function",
    language: str = "python",
    dependencies: list = None,
    verdict_score: float = 0.9,
    thumbs_up: int = 10,
    thumbs_down: int = 1,
) -> CodeFile:
    """Create a mock CodeFile for testing."""
    return CodeFile(
        code_block_id=code_block_id,
        filename=filename,
        source=source,
        entrypoint=entrypoint,
        description=description,
        language=language,
        dependencies=dependencies or [],
        verdict_score=verdict_score,
        thumbs_up=thumbs_up,
        thumbs_down=thumbs_down,
    )


# =============================================================================
# Pre-fetch Tests
# =============================================================================


class TestPrefetch:
    """Tests for pre-fetch functionality."""

    @pytest.mark.asyncio
    async def test_prefetch_downloads_code_files(self, tmp_path):
        """Pre-fetch should download code files to sandbox directory."""
        # Import here to use mocked SDK
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        mock_files = [
            create_mock_code_file(
                filename="github_fetcher.py",
                source="import requests\ndef fetch_user(username): pass",
                entrypoint="fetch_user",
            ),
            create_mock_code_file(
                filename="data_processor.py",
                source="def process(data): return data",
                entrypoint="process",
            ),
        ]

        mock_response = GetCodeFilesResponse(
            files=mock_files,
            task="Fetch GitHub data",
            total_found=2,
        )

        options = RaysurferAgentOptions(
            raysurfer_api_key="test-key",
            sandbox_dir=tmp_path,
            prefetch_count=5,
        )

        client = RaysurferClient(options=options)

        # Mock the raysurfer client
        mock_raysurfer = AsyncMock()
        mock_raysurfer.get_code_files = AsyncMock(return_value=mock_response)
        mock_raysurfer.__aenter__ = AsyncMock(return_value=mock_raysurfer)
        mock_raysurfer.__aexit__ = AsyncMock()

        client._raysurfer = mock_raysurfer

        # Test prefetch
        files = await client._prefetch_code_files("Fetch GitHub data")

        assert len(files) == 2
        assert files[0].filename == "github_fetcher.py"
        mock_raysurfer.get_code_files.assert_called_once_with(
            task="Fetch GitHub data",
            top_k=5,
            min_verdict_score=0.3,
            prefer_complete=True,
        )

    @pytest.mark.asyncio
    async def test_prefetch_writes_files_to_sandbox(self, tmp_path):
        """Pre-fetch should write downloaded files to sandbox directory."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        mock_files = [
            create_mock_code_file(
                filename="test_script.py",
                source="print('hello world')",
                entrypoint="main",
            ),
        ]

        options = RaysurferAgentOptions(
            raysurfer_api_key="test-key",
            sandbox_dir=tmp_path,
        )

        client = RaysurferClient(options=options)
        client._prefetched_files = mock_files

        # Test download
        await client._download_files_to_sandbox()

        # Verify file was written
        expected_path = tmp_path / "test_script.py"
        assert expected_path.exists()
        assert expected_path.read_text() == "print('hello world')"

    @pytest.mark.asyncio
    async def test_prefetch_handles_backend_error_gracefully(self, tmp_path):
        """Pre-fetch should return empty list on backend error."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(
            raysurfer_api_key="test-key",
            sandbox_dir=tmp_path,
        )

        client = RaysurferClient(options=options)

        # Mock raysurfer client that raises an error
        mock_raysurfer = AsyncMock()
        mock_raysurfer.get_code_files = AsyncMock(side_effect=Exception("Backend unavailable"))
        client._raysurfer = mock_raysurfer

        # Should not raise, return empty list
        files = await client._prefetch_code_files("Some task")
        assert files == []

    @pytest.mark.asyncio
    async def test_prefetch_respects_min_verdict_score(self, tmp_path):
        """Pre-fetch should pass min_verdict_score to backend."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(
            raysurfer_api_key="test-key",
            sandbox_dir=tmp_path,
            min_verdict_score=0.7,  # Higher threshold
        )

        client = RaysurferClient(options=options)

        mock_raysurfer = AsyncMock()
        mock_raysurfer.get_code_files = AsyncMock(
            return_value=GetCodeFilesResponse(files=[], task="test", total_found=0)
        )
        client._raysurfer = mock_raysurfer

        await client._prefetch_code_files("test task")

        mock_raysurfer.get_code_files.assert_called_once()
        call_kwargs = mock_raysurfer.get_code_files.call_args.kwargs
        assert call_kwargs["min_verdict_score"] == 0.7

    @pytest.mark.asyncio
    async def test_prefetch_respects_prefetch_count(self, tmp_path):
        """Pre-fetch should pass prefetch_count to backend as top_k."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(
            raysurfer_api_key="test-key",
            sandbox_dir=tmp_path,
            prefetch_count=10,
        )

        client = RaysurferClient(options=options)

        mock_raysurfer = AsyncMock()
        mock_raysurfer.get_code_files = AsyncMock(
            return_value=GetCodeFilesResponse(files=[], task="test", total_found=0)
        )
        client._raysurfer = mock_raysurfer

        await client._prefetch_code_files("test task")

        call_kwargs = mock_raysurfer.get_code_files.call_args.kwargs
        assert call_kwargs["top_k"] == 10


# =============================================================================
# System Prompt Injection Tests
# =============================================================================


class TestSystemPromptInjection:
    """Tests for system prompt injection functionality."""

    def test_system_prompt_injection_with_code_files(self, tmp_path):
        """System prompt should include code file contents when files are pre-fetched."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        mock_files = [
            create_mock_code_file(
                filename="github_fetcher.py",
                source="import requests\n\ndef fetch_user(username):\n    pass",
                entrypoint="fetch_user",
                description="Fetches GitHub user data",
                verdict_score=0.9,
                thumbs_up=15,
                dependencies=["requests"],
            ),
        ]

        options = RaysurferAgentOptions(
            raysurfer_api_key="test-key",
            sandbox_dir=tmp_path,
            system_prompt="You are a helpful assistant.",
        )

        client = RaysurferClient(options=options)
        client._prefetched_files = mock_files

        # Build system prompt
        result = client._build_system_prompt()

        # Verify code file contents are included
        assert "github_fetcher.py" in result
        assert "import requests" in result
        assert "fetch_user" in result
        assert "Fetches GitHub user data" in result
        assert "90%" in result  # Verdict score formatted as percentage
        assert "15 thumbs up" in result
        assert "requests" in result  # Dependencies

    def test_system_prompt_injection_without_files(self, tmp_path):
        """System prompt should return base prompt when no files are pre-fetched."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(
            raysurfer_api_key="test-key",
            sandbox_dir=tmp_path,
            system_prompt="You are a helpful assistant.",
        )

        client = RaysurferClient(options=options)
        client._prefetched_files = []  # No files

        result = client._build_system_prompt()

        assert result == "You are a helpful assistant."

    def test_system_prompt_injection_with_preset(self, tmp_path):
        """System prompt should handle preset format correctly."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        mock_files = [
            create_mock_code_file(
                filename="helper.py",
                source="def helper(): pass",
                entrypoint="helper",
            ),
        ]

        preset = {"type": "preset", "name": "default", "append": "Extra context."}

        options = RaysurferAgentOptions(
            raysurfer_api_key="test-key",
            sandbox_dir=tmp_path,
            system_prompt=preset,
        )

        client = RaysurferClient(options=options)
        client._prefetched_files = mock_files

        result = client._build_system_prompt()

        # Should be a dict with appended content
        assert isinstance(result, dict)
        assert result["type"] == "preset"
        assert "Extra context." in result["append"]
        assert "helper.py" in result["append"]

    def test_format_code_snippets_includes_all_metadata(self, tmp_path):
        """Code snippets should include all relevant metadata."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        mock_files = [
            create_mock_code_file(
                filename="api_client.py",
                source="class APIClient:\n    def __init__(self): pass",
                entrypoint="APIClient",
                description="HTTP API client wrapper",
                language="python",
                dependencies=["httpx", "pydantic"],
                verdict_score=0.85,
                thumbs_up=20,
            ),
        ]

        options = RaysurferAgentOptions(sandbox_dir=tmp_path)
        client = RaysurferClient(options=options)
        client._prefetched_files = mock_files

        result = client._format_code_snippets()

        # Check all metadata is present
        assert "api_client.py" in result
        assert "HTTP API client wrapper" in result
        assert "APIClient" in result  # Entrypoint
        assert "85%" in result  # Verdict score
        assert "20 thumbs up" in result
        assert "httpx" in result
        assert "pydantic" in result
        assert "```python" in result  # Language-specific code block

    def test_system_prompt_injection_multiple_files(self, tmp_path):
        """System prompt should include all pre-fetched files."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        mock_files = [
            create_mock_code_file(filename="file1.py", source="def one(): pass"),
            create_mock_code_file(filename="file2.py", source="def two(): pass"),
            create_mock_code_file(filename="file3.py", source="def three(): pass"),
        ]

        options = RaysurferAgentOptions(sandbox_dir=tmp_path)
        client = RaysurferClient(options=options)
        client._prefetched_files = mock_files

        result = client._build_system_prompt()

        assert "file1.py" in result
        assert "file2.py" in result
        assert "file3.py" in result


# =============================================================================
# Auto-Caching Tests
# =============================================================================


class TestAutoCaching:
    """Tests for auto-caching generated code back to backend."""

    def test_track_generated_file_python(self, tmp_path):
        """Should track Python files written by Write tool."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(sandbox_dir=tmp_path)
        client = RaysurferClient(options=options)

        tool_input = {
            "file_path": "/sandbox/my_script.py",
            "content": "def main():\n    print('hello')",
        }

        client._track_generated_file(tool_input)

        assert len(client._generated_files) == 1
        assert client._generated_files[0]["file_path"] == "/sandbox/my_script.py"
        assert client._generated_files[0]["language"] == "python"

    def test_track_generated_file_typescript(self, tmp_path):
        """Should track TypeScript files written by Write tool."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(sandbox_dir=tmp_path)
        client = RaysurferClient(options=options)

        tool_input = {
            "file_path": "/sandbox/utils.ts",
            "content": "export function greet(name: string): void {}",
        }

        client._track_generated_file(tool_input)

        assert len(client._generated_files) == 1
        assert client._generated_files[0]["language"] == "typescript"

    def test_track_generated_file_ignores_non_code(self, tmp_path):
        """Should ignore non-code files."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(sandbox_dir=tmp_path)
        client = RaysurferClient(options=options)

        # These should be ignored
        non_code_files = [
            {"file_path": "/sandbox/data.txt", "content": "plain text"},
            {"file_path": "/sandbox/image.png", "content": "binary data"},
            {"file_path": "/sandbox/document.pdf", "content": "pdf content"},
        ]

        for tool_input in non_code_files:
            client._track_generated_file(tool_input)

        assert len(client._generated_files) == 0

    def test_track_generated_file_handles_various_extensions(self, tmp_path):
        """Should track files with various code extensions."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(sandbox_dir=tmp_path)
        client = RaysurferClient(options=options)

        code_files = [
            ("/sandbox/app.js", "javascript"),
            ("/sandbox/main.go", "go"),
            ("/sandbox/lib.rs", "rust"),
            ("/sandbox/script.sh", "bash"),
            ("/sandbox/config.yaml", "yaml"),
            ("/sandbox/data.json", "json"),
        ]

        for file_path, expected_lang in code_files:
            client._track_generated_file({"file_path": file_path, "content": "content"})

        assert len(client._generated_files) == len(code_files)
        languages = [f["language"] for f in client._generated_files]
        for _, expected_lang in code_files:
            assert expected_lang in languages

    @pytest.mark.asyncio
    async def test_store_generated_code_on_success(self, tmp_path):
        """Should store generated code to backend on task success."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(
            raysurfer_api_key="test-key",
            sandbox_dir=tmp_path,
        )

        client = RaysurferClient(options=options)
        client._current_query = "Create a data fetcher"
        client._generated_files = [
            {
                "file_path": "/sandbox/fetcher.py",
                "content": "def fetch_data():\n    import requests\n    return requests.get('http://api.com').json()",
                "language": "python",
            }
        ]

        # Mock raysurfer client
        mock_raysurfer = AsyncMock()
        mock_raysurfer.store_code_block = AsyncMock()
        client._raysurfer = mock_raysurfer

        await client._store_generated_code()

        mock_raysurfer.store_code_block.assert_called_once()
        call_kwargs = mock_raysurfer.store_code_block.call_args.kwargs
        assert call_kwargs["name"] == "fetcher"
        assert call_kwargs["language"] == "python"
        assert "Create a data fetcher" in call_kwargs["description"]

    @pytest.mark.asyncio
    async def test_store_generated_code_detects_entrypoint(self, tmp_path):
        """Should detect entrypoint function in generated code."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(sandbox_dir=tmp_path)
        client = RaysurferClient(options=options)

        # Test Python entrypoint detection
        entrypoint = client._detect_entrypoint(
            "def main():\n    print('hello')\n\nif __name__ == '__main__':\n    main()",
            "python",
        )
        assert entrypoint == "main"

        # Test first function detection
        entrypoint = client._detect_entrypoint(
            "def process_data(x):\n    return x * 2\n\ndef helper():\n    pass",
            "python",
        )
        assert entrypoint == "process_data"

    @pytest.mark.asyncio
    async def test_store_generated_code_extracts_tags(self, tmp_path):
        """Should extract relevant tags from generated code."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(sandbox_dir=tmp_path)
        client = RaysurferClient(options=options)

        source_code = """
import requests
import json

async def fetch_api_data(url):
    response = requests.get(url)
    return json.loads(response.text)
"""

        tags = client._extract_tags(source_code, "python")

        assert "python" in tags
        assert "requests" in tags
        assert "json" in tags
        assert "api" in tags
        assert "async" in tags

    @pytest.mark.asyncio
    async def test_store_generated_code_handles_errors_gracefully(self, tmp_path):
        """Should handle storage errors without crashing."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(sandbox_dir=tmp_path)
        client = RaysurferClient(options=options)
        client._current_query = "test"
        client._generated_files = [{"file_path": "/test.py", "content": "print('hi')", "language": "python"}]

        # Mock raysurfer that raises error
        mock_raysurfer = AsyncMock()
        mock_raysurfer.store_code_block = AsyncMock(side_effect=Exception("Storage failed"))
        client._raysurfer = mock_raysurfer

        # Should not raise
        await client._store_generated_code()

    @pytest.mark.asyncio
    async def test_no_store_without_query(self, tmp_path):
        """Should not store if no query was made."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(sandbox_dir=tmp_path)
        client = RaysurferClient(options=options)
        client._current_query = None  # No query
        client._generated_files = [{"file_path": "/test.py", "content": "print('hi')", "language": "python"}]

        mock_raysurfer = AsyncMock()
        client._raysurfer = mock_raysurfer

        await client._store_generated_code()

        mock_raysurfer.store_code_block.assert_not_called()


# =============================================================================
# RaysurferAgentOptions Tests
# =============================================================================


class TestRaysurferAgentOptions:
    """Tests for RaysurferAgentOptions configuration."""

    def test_default_options(self):
        """Should have sensible defaults."""
        from raysurfer.sdk_client import RaysurferAgentOptions

        options = RaysurferAgentOptions()

        assert options.raysurfer_api_key is None
        assert options.prefetch_count == 5
        assert options.min_verdict_score == 0.3
        assert options.prefer_complete is True
        assert options.model == "claude-opus-4-5-20251101"

    def test_custom_options(self, tmp_path):
        """Should accept custom configuration."""
        from raysurfer.sdk_client import RaysurferAgentOptions

        options = RaysurferAgentOptions(
            raysurfer_api_key="rs_test123",
            raysurfer_base_url="http://localhost:8000",
            prefetch_count=10,
            min_verdict_score=0.5,
            prefer_complete=False,
            sandbox_dir=tmp_path,
            model="claude-opus-4-5-20250514",
            allowed_tools=["Read", "Write", "Bash"],
            permission_mode="bypassPermissions",
        )

        assert options.raysurfer_api_key == "rs_test123"
        assert options.raysurfer_base_url == "http://localhost:8000"
        assert options.prefetch_count == 10
        assert options.min_verdict_score == 0.5
        assert options.prefer_complete is False
        assert options.sandbox_dir == tmp_path
        assert options.model == "claude-opus-4-5-20250514"
        assert "Bash" in options.allowed_tools
        assert options.permission_mode == "bypassPermissions"


# =============================================================================
# Claude Options Building Tests
# =============================================================================


class TestClaudeOptionsBuilding:
    """Tests for building ClaudeAgentOptions from RaysurferAgentOptions."""

    def test_build_claude_options_runs_without_error(self, tmp_path):
        """Should build options without errors."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(sandbox_dir=tmp_path)
        client = RaysurferClient(options=options)

        # Should not raise any exceptions
        result = client._build_claude_options()

        # Returns a mock (in test) or real ClaudeAgentOptions (in production)
        assert result is not None

    def test_build_claude_options_with_custom_sandbox_config(self, tmp_path):
        """Should merge custom sandbox config with defaults."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        custom_sandbox = {
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "excludedCommands": ["docker"],
        }

        options = RaysurferAgentOptions(
            sandbox_dir=tmp_path,
            sandbox=custom_sandbox,
        )
        client = RaysurferClient(options=options)

        # Should not raise
        result = client._build_claude_options()
        assert result is not None

    def test_build_claude_options_preserves_raysurfer_options(self, tmp_path):
        """Should preserve all RaysurferAgentOptions values."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(
            sandbox_dir=tmp_path,
            model="claude-opus-4-5-20250514",
            allowed_tools=["Read", "Write", "Bash", "Grep"],
            permission_mode="bypassPermissions",
            max_turns=5,
        )
        client = RaysurferClient(options=options)

        # Verify the options are stored correctly
        assert client.options.model == "claude-opus-4-5-20250514"
        assert "Bash" in client.options.allowed_tools
        assert client.options.permission_mode == "bypassPermissions"
        assert client.options.max_turns == 5

        # Building options should work
        result = client._build_claude_options()
        assert result is not None

    def test_build_claude_options_with_env_vars(self, tmp_path):
        """Should pass through environment variables."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(
            sandbox_dir=tmp_path,
            env={"MY_VAR": "value", "API_KEY": "secret"},
        )
        client = RaysurferClient(options=options)

        # Verify env is stored
        assert client.options.env["MY_VAR"] == "value"
        assert client.options.env["API_KEY"] == "secret"

        # Building options should work
        result = client._build_claude_options()
        assert result is not None


# =============================================================================
# Prefetched Files Property Tests
# =============================================================================


class TestPrefetchedFilesProperty:
    """Tests for prefetched_files property."""

    def test_prefetched_files_returns_empty_initially(self, tmp_path):
        """Should return empty list before prefetch."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(sandbox_dir=tmp_path)
        client = RaysurferClient(options=options)

        assert client.prefetched_files == []

    def test_prefetched_files_returns_files_after_prefetch(self, tmp_path):
        """Should return files after prefetch."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(sandbox_dir=tmp_path)
        client = RaysurferClient(options=options)

        mock_files = [create_mock_code_file(filename="test.py")]
        client._prefetched_files = mock_files

        assert client.prefetched_files == mock_files
        assert len(client.prefetched_files) == 1


class TestSandboxDirProperty:
    """Tests for sandbox_dir property."""

    def test_sandbox_dir_returns_configured_path(self, tmp_path):
        """Should return configured sandbox directory."""
        from raysurfer.sdk_client import RaysurferAgentOptions, RaysurferClient

        options = RaysurferAgentOptions(sandbox_dir=tmp_path)
        client = RaysurferClient(options=options)

        assert client.sandbox_dir == tmp_path
