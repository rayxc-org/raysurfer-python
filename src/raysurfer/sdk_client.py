"""
Drop-in replacement for Claude Agent SDK with automatic code caching.

Simply swap your import and rename your client:
    # Before
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
    client = ClaudeSDKClient(options)
    await client.query("task")

    # After
    from raysurfer import RaysurferClient
    from claude_agent_sdk import ClaudeAgentOptions
    client = RaysurferClient(options)
    await client.query("task")

Options come directly from claude_agent_sdk - no Raysurfer-specific options needed.
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import shutil
import tempfile
import time
import warnings
from collections.abc import AsyncIterator
from types import TracebackType

# Isolate temp directory to avoid file watcher conflicts when running
# nested inside another Claude Code session.
_ISOLATED_TMPDIR: str | None = None


def _setup_isolated_env() -> str:
    """Set up isolated temp directory for nested Claude Code execution."""
    global _ISOLATED_TMPDIR

    if _ISOLATED_TMPDIR is None:
        _ISOLATED_TMPDIR = tempfile.mkdtemp(prefix="raysurfer_sdk_")
        os.environ["TMPDIR"] = _ISOLATED_TMPDIR
        os.environ["TEMP"] = _ISOLATED_TMPDIR
        os.environ["TMP"] = _ISOLATED_TMPDIR
        os.environ["CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"] = "1"
        atexit.register(_cleanup_isolated_env)

    return _ISOLATED_TMPDIR


def _cleanup_isolated_env() -> None:
    """Clean up isolated temp directory on exit."""
    global _ISOLATED_TMPDIR
    if _ISOLATED_TMPDIR and os.path.exists(_ISOLATED_TMPDIR):
        try:
            shutil.rmtree(_ISOLATED_TMPDIR)
        except Exception:
            pass
        _ISOLATED_TMPDIR = None


_setup_isolated_env()

# Import from Claude Agent SDK - these are passed through directly
from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    HookMatcher,
    Message,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from claude_agent_sdk import ClaudeSDKClient as _BaseClaudeSDKClient

from raysurfer.client import AsyncRaySurfer
from raysurfer.sdk_types import CodeFile
from raysurfer.types import FileWritten, JsonDict, SnipsDesired

logger = logging.getLogger(__name__)


class _DebugLogger:
    """Debug logger that outputs timing and diagnostic info when RAYSURFER_DEBUG=true."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._timers: dict[str, float] = {}

    def log(self, *args: str | int | float | bool) -> None:
        if self.enabled:
            print("[raysurfer]", *args)

    def time(self, label: str) -> None:
        if self.enabled:
            self._timers[label] = time.time()

    def time_end(self, label: str) -> None:
        if self.enabled and label in self._timers:
            elapsed = (time.time() - self._timers[label]) * 1000
            print(f"[raysurfer] {label}: {elapsed:.2f}ms")
            del self._timers[label]

    def table(self, data: list[dict[str, str]]) -> None:
        if self.enabled and data:
            # Simple table output
            for row in data:
                print("[raysurfer]  ", row)


# Re-export all SDK types for convenience
__all__ = [
    "RaysurferClient",
    # Re-exported from Claude Agent SDK (use these directly)
    "ClaudeAgentOptions",
    "AgentDefinition",
    "HookMatcher",
    "Message",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "ResultMessage",
    "TextBlock",
    "ThinkingBlock",
    "ToolUseBlock",
    "ToolResultBlock",
]

# Default backend URL
DEFAULT_RAYSURFER_URL = "https://api.raysurfer.com"

# Maximum file size (in bytes) that will be cached from Bash-generated output
MAX_CACHEABLE_FILE_SIZE = 100_000


class RaysurferClient:
    """
    Drop-in replacement for ClaudeSDKClient with automatic Raysurfer caching.

    Usage:
        from raysurfer import RaysurferClient
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(
            allowed_tools=["Read", "Write", "Bash"],
            system_prompt="You are a helpful assistant.",
        )

        async with RaysurferClient(options) as client:
            await client.query("Fetch data from GitHub API")
            async for msg in client.response():
                print(msg)

    Features:
    - Automatic cache retrieval and system prompt augmentation
    - Multi-agent support: subagent prompts are also augmented with cache
    - Bash file tracking: detects files created by Bash commands
    - Hook propagation: user hooks are preserved and work alongside cache hooks

    Set RAYSURFER_API_KEY environment variable to enable caching.
    Options come directly from claude_agent_sdk.ClaudeAgentOptions.
    """

    # File extensions we track from Bash output
    TRACKABLE_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".rb",
        ".go",
        ".rs",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".pdf",
        ".docx",
        ".xlsx",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
        ".xml",
        ".html",
        ".css",
        ".md",
        ".txt",
        ".sh",
        ".sql",
    }

    def __init__(
        self,
        options: ClaudeAgentOptions | None = None,
        workspace_id: str | None = None,
        debug: bool = False,
        public_snips: bool = False,
    ):
        """
        Initialize RaysurferClient.

        Args:
            options: ClaudeAgentOptions from claude_agent_sdk (passed through directly)
            workspace_id: Workspace ID for per-customer isolation (enterprise only)
            debug: Enable debug logging - also enabled via RAYSURFER_DEBUG=true env var
            public_snips: Include community-contributed public snippets in search results
        """
        self._options = options or ClaudeAgentOptions()
        self._base_client: _BaseClaudeSDKClient | None = None
        self._raysurfer: AsyncRaySurfer | None = None
        self._current_query: str | None = None
        self._generated_files: list[FileWritten] = []
        self._bash_generated_files: list[str] = []
        self._task_succeeded: bool = False
        self._cache_enabled: bool = False
        self._cached_code_blocks: list[dict[str, str]] = []
        self._subagent_cache: dict[str, str] = {}
        self._execution_logs: list[str] = []
        self._workspace_id = workspace_id
        self._public_snips = public_snips
        # Initialize debug logger
        debug_enabled = debug or os.environ.get("RAYSURFER_DEBUG", "").lower() == "true"
        self._debug = _DebugLogger(debug_enabled)

    async def __aenter__(self) -> "RaysurferClient":
        """Initialize the client."""
        api_key = os.environ.get("RAYSURFER_API_KEY")
        base_url = os.environ.get("RAYSURFER_BASE_URL", DEFAULT_RAYSURFER_URL)

        self._debug.log("Initializing RaysurferClient")
        self._debug.log("Cache enabled:", bool(api_key))
        self._debug.log("Base URL:", base_url)

        if not api_key:
            warnings.warn("RAYSURFER_API_KEY not set - caching disabled", stacklevel=2)

        if api_key:
            self._cache_enabled = True
            # Auto-set snips_desired="client" when workspace_id is provided
            snips_desired = SnipsDesired.CLIENT if self._workspace_id else None
            self._raysurfer = AsyncRaySurfer(
                api_key=api_key,
                base_url=base_url,
                workspace_id=self._workspace_id,
                snips_desired=snips_desired,
                public_snips=self._public_snips,
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
        if self._base_client:
            await self._base_client.__aexit__(exc_type, exc_val, exc_tb)
        if self._raysurfer:
            await self._raysurfer.__aexit__(exc_type, exc_val, exc_tb)

    async def query(self, prompt: str) -> None:
        """
        Send a query to Claude with Raysurfer caching.

        If RAYSURFER_API_KEY is set, automatically retrieves relevant cached
        code and injects it into the system prompt.

        Args:
            prompt: The task/query to send to Claude
        """
        self._current_query = prompt
        self._generated_files = []
        self._bash_generated_files = []
        self._task_succeeded = False
        self._cached_code_blocks = []
        self._subagent_cache = {}
        self._execution_logs = []

        # Pre-fetch cache for subagents if this is a multi-agent system
        if self._options.agents:
            await self._fetch_subagent_cache(self._options.agents)

        # Retrieve cached code if caching is enabled
        augmented_options = await self._augment_options_with_cache(prompt)

        # Initialize and query the base client
        self._base_client = _BaseClaudeSDKClient(options=augmented_options)
        await self._base_client.__aenter__()
        await self._base_client.query(prompt)

    async def response(self) -> AsyncIterator[Message]:
        """
        Receive and yield response messages from Claude.

        After successful task completion, automatically uploads any
        generated code to the Raysurfer cache.

        Yields:
            Message objects from Claude
        """
        if not self._base_client:
            raise RuntimeError("Must call query() before response()")

        last_bash_command: str | None = None

        # File modification tools to track
        file_modify_tools = {"Write", "Edit", "MultiEdit", "NotebookEdit"}

        async for message in self._base_client.receive_response():
            # Track file modification and Bash tool calls
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        if block.name in file_modify_tools:
                            self._track_file_modify_tool(block.name, block.input)
                        elif block.name == "Bash":
                            last_bash_command = block.input.get("command", "")
                            self._track_bash_file_outputs(last_bash_command)

            # Track files from Bash tool results and capture execution logs
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        # Capture tool result content as execution log
                        content_str = str(block.content) if hasattr(block, "content") else ""
                        if content_str:
                            self._execution_logs.append(content_str[:5000])
                        if last_bash_command:
                            self._extract_files_from_bash_output(
                                last_bash_command,
                                content_str,
                            )
                            last_bash_command = None

            # Check for successful completion
            if isinstance(message, ResultMessage):
                if message.subtype == "success":
                    self._task_succeeded = True

            yield message

        # Upload generated code if task succeeded
        if self._cache_enabled and self._task_succeeded:
            if self._generated_files:
                await self._upload_to_cache()
            await self._cache_bash_generated_files()

        # Submit votes for cached code blocks that were used
        if self._cache_enabled and self._task_succeeded and self._cached_code_blocks:
            await self._submit_votes()

    def _track_file_modify_tool(self, tool_name: str, tool_input: JsonDict) -> None:
        """Track a file modified by Write, Edit, MultiEdit, or NotebookEdit tools."""
        if tool_name == "Write":
            file_path = tool_input.get("file_path", "")
            content = tool_input.get("content", "")
            if file_path and content:
                self._generated_files.append(FileWritten(path=file_path, content=content))
                self._debug.log(f"  → Write tool: {file_path}")
        elif tool_name == "Edit":
            file_path = tool_input.get("file_path", "")
            if file_path:
                # Edit doesn't have full content, mark for later reading
                if file_path not in [f.path for f in self._generated_files]:
                    self._bash_generated_files.append(file_path)
                    self._debug.log(f"  → Edit tool: {file_path}")
        elif tool_name == "MultiEdit":
            file_path = tool_input.get("file_path", "")
            if file_path:
                if file_path not in [f.path for f in self._generated_files]:
                    self._bash_generated_files.append(file_path)
                    self._debug.log(f"  → MultiEdit tool: {file_path}")
        elif tool_name == "NotebookEdit":
            notebook_path = tool_input.get("notebook_path", "")
            if notebook_path:
                if notebook_path not in [f.path for f in self._generated_files]:
                    self._bash_generated_files.append(notebook_path)
                    self._debug.log(f"  → NotebookEdit tool: {notebook_path}")

    def _track_bash_file_outputs(self, command: str) -> None:
        """Extract potential output files from Bash commands."""
        patterns = [
            r">\s*([^\s;&|]+)",
            r">>\s*([^\s;&|]+)",
            r"-o\s+([^\s;&|]+)",
            r"--output[=\s]+([^\s;&|]+)",
            r'savefig\([\'"]([^\'"]+)[\'"]\)',
            r'to_csv\([\'"]([^\'"]+)[\'"]\)',
            r'to_excel\([\'"]([^\'"]+)[\'"]\)',
            r'write\([\'"]([^\'"]+)[\'"]\)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, command)
            for match in matches:
                ext = os.path.splitext(match)[1].lower()
                if ext in self.TRACKABLE_EXTENSIONS:
                    self._bash_generated_files.append(match)

    def _extract_files_from_bash_output(self, command: str, output: str) -> None:
        """Extract files mentioned in Bash command output."""
        file_pattern = r"[/\w.-]+\.\w{2,5}"
        matches = re.findall(file_pattern, output)
        for match in matches:
            ext = os.path.splitext(match)[1].lower()
            if ext in self.TRACKABLE_EXTENSIONS:
                if match not in self._bash_generated_files:
                    self._bash_generated_files.append(match)

    async def _cache_bash_generated_files(self) -> None:
        """Attempt to read and cache files generated by Bash commands."""
        if not self._raysurfer or not self._bash_generated_files:
            return

        for file_path in self._bash_generated_files:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext in {
                    ".py",
                    ".js",
                    ".ts",
                    ".rb",
                    ".go",
                    ".rs",
                    ".java",
                    ".json",
                    ".yaml",
                    ".yml",
                    ".xml",
                    ".html",
                    ".css",
                    ".md",
                    ".txt",
                    ".sh",
                    ".sql",
                    ".csv",
                }:
                    if os.path.exists(file_path) and os.path.getsize(file_path) < MAX_CACHEABLE_FILE_SIZE:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        if content.strip():
                            self._generated_files.append(FileWritten(path=file_path, content=content))
                            logger.debug(f"Tracked Bash-generated file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not read Bash-generated file {file_path}: {e}")

    def _augment_subagent_prompts(self, agents: dict[str, AgentDefinition] | None) -> dict[str, AgentDefinition] | None:
        """Augment subagent prompts with cached code snippets."""
        if not agents or not self._cache_enabled:
            return agents

        augmented_agents = {}
        for name, agent_def in agents.items():
            subagent_snippet = self._subagent_cache.get(name, "")
            if subagent_snippet:
                augmented_prompt = (agent_def.prompt or "") + subagent_snippet
                augmented_agents[name] = AgentDefinition(
                    description=agent_def.description,
                    prompt=augmented_prompt,
                    tools=agent_def.tools,
                    model=agent_def.model,
                )
            else:
                augmented_agents[name] = agent_def

        return augmented_agents

    async def _fetch_subagent_cache(self, agents: dict[str, AgentDefinition] | None) -> None:
        """Pre-fetch cache snippets for all subagents."""
        if not agents or not self._cache_enabled or not self._raysurfer:
            return

        for name, agent_def in agents.items():
            try:
                task = agent_def.description or name
                response = await self._raysurfer.get_code_files(
                    task=task,
                    top_k=3,
                    min_verdict_score=0.3,
                    prefer_complete=True,
                )
                if response.files:
                    self._subagent_cache[name] = self._format_code_snippets(response.files)
                    logger.debug(f"Cached {len(response.files)} code blocks for subagent: {name}")
            except Exception as e:
                logger.debug(f"Failed to fetch cache for subagent {name}: {e}")

    async def _augment_options_with_cache(self, task: str) -> ClaudeAgentOptions:
        """Retrieve cached code, write to filesystem, and tell LLM where files are."""
        if not self._cache_enabled or not self._raysurfer:
            return self._options

        try:
            cwd = self._options.cwd or os.getcwd()
            cache_dir = os.path.join(cwd, ".raysurfer_code")

            self._debug.time("Cache lookup")
            response = await self._raysurfer.get_code_files(
                task=task,
                top_k=5,
                min_verdict_score=0.3,
                prefer_complete=True,
                cache_dir=cache_dir,  # Pass cache_dir to get full paths in add_to_llm_prompt
            )
            self._debug.time_end("Cache lookup")

            self._debug.log(f"Found {len(response.files)} cached files")
            if response.files:
                self._debug.table(
                    [
                        {
                            "filename": f.filename,
                            "score": f"{f.score * 100:.0f}%",
                            "thumbs": f"{f.thumbs_up}/{f.thumbs_down}",
                        }
                        for f in response.files
                    ]
                )

            if response.files:
                logger.info(f"Cache hit: {len(response.files)} snippets retrieved for task")

            if not response.files:
                return self._options

            self._cached_code_blocks = [
                {
                    "code_block_id": f.code_block_id,
                    "filename": f.filename,
                    "description": f.description,
                }
                for f in response.files
            ]

            written_files = self._write_cached_files_to_disk(response.files)

            if not written_files:
                return self._options

            # Use the add_to_llm_prompt from the response instead of generating our own
            cache_notice = response.add_to_llm_prompt
            base_prompt = self._options.system_prompt

            if isinstance(base_prompt, dict) and base_prompt.get("type") == "preset":
                augmented_prompt = {
                    **base_prompt,
                    "append": base_prompt.get("append", "") + cache_notice,
                }
            else:
                augmented_prompt = (base_prompt or "") + cache_notice

            augmented_agents = self._augment_subagent_prompts(self._options.agents)

            # Create new options with augmented prompt - pass through all other options
            return ClaudeAgentOptions(
                allowed_tools=self._options.allowed_tools,
                disallowed_tools=self._options.disallowed_tools,
                permission_mode=self._options.permission_mode,
                system_prompt=augmented_prompt,
                cwd=self._options.cwd,
                add_dirs=self._options.add_dirs,
                max_turns=self._options.max_turns,
                model=self._options.model,
                env=self._options.env,
                mcp_servers=self._options.mcp_servers,
                hooks=self._options.hooks,
                can_use_tool=self._options.can_use_tool,
                setting_sources=self._options.setting_sources,
                include_partial_messages=self._options.include_partial_messages,
                fork_session=self._options.fork_session,
                continue_conversation=self._options.continue_conversation,
                resume=self._options.resume,
                agents=augmented_agents,
                plugins=self._options.plugins,
                enable_file_checkpointing=self._options.enable_file_checkpointing,
                output_format=self._options.output_format,
                sandbox=self._options.sandbox,
                extra_args=self._options.extra_args,
                max_buffer_size=self._options.max_buffer_size,
                stderr=self._options.stderr,
                user=self._options.user,
                settings=self._options.settings,
                permission_prompt_tool_name=self._options.permission_prompt_tool_name,
            )
        except Exception as e:
            logger.warning(f"Cache unavailable: {e}")
            return self._options

    def _write_cached_files_to_disk(self, files: list[CodeFile]) -> list[dict[str, str | float]]:
        """Write cached code files to the working directory."""
        written_files = []
        cwd = self._options.cwd or os.getcwd()

        cache_dir = os.path.join(cwd, ".raysurfer_code")

        # Clear existing cache to avoid stale files from previous runs
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)

        for f in files:
            try:
                file_path = os.path.join(cache_dir, f.filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True) if os.path.dirname(
                    file_path
                ) != cache_dir else None

                with open(file_path, "w", encoding="utf-8") as out_file:
                    out_file.write(f.source)

                written_files.append(
                    {
                        "path": file_path,
                        "filename": f.filename,
                        "description": f.description,
                        "entrypoint": f.entrypoint,
                        "language": f.language,
                        "confidence": f.score,
                    }
                )
                logger.debug(f"Wrote cached file: {file_path}")
            except Exception as e:
                logger.debug(f"Failed to write cached file {f.filename}: {e}")

        return written_files

    def _format_code_snippets(self, files: list[CodeFile]) -> str:
        """Format cached code files as markdown for system prompt."""
        snippets = "\n\n## Cached Code (from Raysurfer)\n\n"
        snippets += "The following pre-validated code is available for this task. "
        snippets += "Use it directly or adapt it as needed.\n\n"

        for f in files:
            snippets += f"### {f.filename}\n"
            snippets += f"**Description**: {f.description}\n"
            snippets += f"**Entrypoint**: `{f.entrypoint}`\n"
            snippets += f"**Confidence**: {f.score:.0%}\n\n"
            snippets += f"```{f.language}\n{f.source}\n```\n\n"

        return snippets

    async def _upload_to_cache(self) -> None:
        """Upload generated code files to the Raysurfer cache (one at a time)."""
        if not self._raysurfer or not self._current_query:
            return

        try:
            self._debug.time("Cache upload")
            self._debug.log(f"Uploading {len(self._generated_files)} files to cache")

            # Join captured execution logs for vote context
            execution_logs = "\n---\n".join(self._execution_logs) if self._execution_logs else None
            if execution_logs:
                self._debug.log(f"Including {len(self._execution_logs)} execution log entries")

            total_stored = 0
            for file in self._generated_files:
                result = await self._raysurfer.upload_new_code_snip(
                    task=self._current_query,
                    file_written=file,
                    succeeded=self._task_succeeded,
                    use_raysurfer_ai_voting=True,
                    execution_logs=execution_logs,
                )
                total_stored += result.code_blocks_stored

            self._debug.time_end("Cache upload")
            if total_stored > 0:
                self._debug.log(f"Cached {total_stored} code blocks")
                logger.info(f"Cached {total_stored} code blocks")
        except Exception as e:
            self._debug.log(f"Cache upload failed: {e}")
            logger.warning(f"Cache upload failed: {e}")

    async def _submit_votes(self) -> None:
        """Submit thumbs up votes for cached code blocks that helped complete the task."""
        if not self._raysurfer or not self._current_query:
            return

        for block in self._cached_code_blocks:
            try:
                await self._raysurfer.vote_code_snip(
                    task=self._current_query,
                    code_block_id=block["code_block_id"],
                    code_block_name=block["filename"],
                    code_block_description=block["description"],
                    succeeded=self._task_succeeded,
                )
                logger.debug(f"Submitted vote for {block['filename']}")
            except Exception as e:
                logger.warning(f"Failed to submit vote for {block['filename']}: {e}")
