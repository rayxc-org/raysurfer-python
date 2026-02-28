"""Helpers for Anthropic programmatic tool calling with materialized Raysurfer snippets."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from raysurfer.sdk_types import CodeFile
from raysurfer.types import FileWritten, SearchMatch, SearchResponse, SubmitExecutionResultResponse

DEFAULT_TOP_K = 5
DEFAULT_MIN_VERDICT_SCORE = 0.3


@dataclass(slots=True)
class ProgrammaticMaterializeContext:
    """Materialized search result returned to callers for prompt injection."""

    tempdir: str
    context_prompt: str
    files: list[CodeFile] = field(default_factory=list)
    top_k: int = DEFAULT_TOP_K
    workspace_id: str | None = None


@dataclass(slots=True)
class ProgrammaticFrameworkResult:
    """Outcome returned by framework-agnostic programmatic runner callbacks."""

    succeeded: bool
    execution_logs: str | None = None


class SupportsAsyncSearchUpload(Protocol):
    """Protocol for async clients that support search and upload operations."""

    async def search(
        self,
        *,
        task: str,
        top_k: int = 5,
        min_verdict_score: float = 0.3,
        prefer_complete: bool = True,
        workspace_id: str | None = None,
    ) -> SearchResponse:
        """Search for cached snippets matching the task."""

    async def upload(
        self,
        *,
        task: str,
        files_written: list[FileWritten] | None = None,
        succeeded: bool = True,
        use_raysurfer_ai_voting: bool = True,
        execution_logs: str | None = None,
        workspace_id: str | None = None,
    ) -> SubmitExecutionResultResponse:
        """Upload one or more files as new snippets."""


class SupportsSyncSearchUpload(Protocol):
    """Protocol for sync clients that support search and upload operations."""

    def search(
        self,
        *,
        task: str,
        top_k: int = 5,
        min_verdict_score: float = 0.3,
        prefer_complete: bool = True,
        workspace_id: str | None = None,
    ) -> SearchResponse:
        """Search for cached snippets matching the task."""

    def upload(
        self,
        *,
        task: str,
        files_written: list[FileWritten] | None = None,
        succeeded: bool = True,
        use_raysurfer_ai_voting: bool = True,
        execution_logs: str | None = None,
        workspace_id: str | None = None,
    ) -> SubmitExecutionResultResponse:
        """Upload one or more files as new snippets."""


class SupportsAsyncAnthropicCreateCallable(Protocol):
    """Protocol for async Anthropic create callables."""

    async def __call__(self, **kwargs: object) -> object:
        """Create an Anthropic message response."""


class SupportsSyncAnthropicCreateCallable(Protocol):
    """Protocol for sync Anthropic create callables."""

    def __call__(self, **kwargs: object) -> object:
        """Create an Anthropic message response."""


class SupportsAsyncProgrammaticSession(Protocol):
    """Protocol for async programmatic materialization session wrappers."""

    async def prepare_turn(self, task: str, *, first_message: bool = True) -> ProgrammaticMaterializeContext:
        """Prepare a turn and optionally run first-message materialization."""

    def append_log(self, log_line: str) -> None:
        """Append an execution log line."""

    async def upload_changed_code(
        self,
        task: str,
        *,
        succeeded: bool = True,
        execution_logs: str | None = None,
        use_raysurfer_ai_voting: bool = True,
    ) -> SubmitExecutionResultResponse | None:
        """Upload changed files as snippets."""

    def cleanup(self, *, remove_tempdir: bool = False) -> None:
        """Clean up temp resources for this session."""


class SupportsSyncProgrammaticSession(Protocol):
    """Protocol for sync programmatic materialization session wrappers."""

    def prepare_turn(self, task: str, *, first_message: bool = True) -> ProgrammaticMaterializeContext:
        """Prepare a turn and optionally run first-message materialization."""

    def append_log(self, log_line: str) -> None:
        """Append an execution log line."""

    def upload_changed_code(
        self,
        task: str,
        *,
        succeeded: bool = True,
        execution_logs: str | None = None,
        use_raysurfer_ai_voting: bool = True,
    ) -> SubmitExecutionResultResponse | None:
        """Upload changed files as snippets."""

    def cleanup(self, *, remove_tempdir: bool = False) -> None:
        """Clean up temp resources for this session."""


class SupportsAsyncFrameworkRunner(Protocol):
    """Protocol for async framework runners (LangSmith/LangChain/custom loops)."""

    async def __call__(self, context: ProgrammaticMaterializeContext) -> ProgrammaticFrameworkResult:
        """Execute a framework-specific programmatic tool-calling run."""


class SupportsSyncFrameworkRunner(Protocol):
    """Protocol for sync framework runners (LangSmith/LangChain/custom loops)."""

    def __call__(self, context: ProgrammaticMaterializeContext) -> ProgrammaticFrameworkResult:
        """Execute a framework-specific programmatic tool-calling run."""


def _validate_top_k(top_k: int) -> int:
    """Validate and normalize top-k snippet selection input."""
    if top_k < 1:
        raise ValueError(
            f"Invalid top_k value: {top_k}. Expected a positive integer (>= 1). "
            "Current tier/state: tier=unknown, top_k_invalid=true. "
            "Fix: pass top_k=1 or higher."
        )
    return top_k


def _build_code_files(matches: list[SearchMatch]) -> list[CodeFile]:
    """Convert unified search matches to materializable code file records."""
    return [
        CodeFile(
            code_block_id=match.code_block.id,
            filename=match.filename,
            source=match.code_block.source,
            entrypoint=match.entrypoint,
            description=match.code_block.description,
            input_schema=match.code_block.input_schema,
            output_schema=match.code_block.output_schema,
            language=match.language,
            dependencies=match.dependencies,
            score=match.score,
            thumbs_up=match.thumbs_up,
            thumbs_down=match.thumbs_down,
        )
        for match in matches
    ]


def _format_context_prompt(files: list[CodeFile], cache_dir: Path) -> str:
    """Render cache guidance prompt with absolute materialized file paths."""
    if not files:
        return ""

    lines = [
        "\n\n## IMPORTANT: Pre-validated Code Files Available\n",
        "The following validated code has been retrieved from the cache. "
        "Use these files directly instead of regenerating code.\n",
    ]

    for code_file in files:
        full_path = (cache_dir / code_file.filename).as_posix()
        lines.append(f"\n### `{code_file.filename}` -> `{full_path}`")
        lines.append(f"- **Description**: {code_file.description}")
        lines.append(f"- **Language**: {code_file.language}")
        lines.append(f"- **Entrypoint**: `{code_file.entrypoint}`")
        lines.append(f"- **Confidence**: {code_file.score:.0%}")
        if code_file.dependencies:
            deps = [f"{name}@{version}" for name, version in code_file.dependencies.items()]
            lines.append(f"- **Dependencies**: {', '.join(deps)}")

    lines.append("\n\n**Instructions**:")
    lines.append("1. Read the cached file(s) before writing new code")
    lines.append("2. Use the cached code as your starting point")
    lines.append("3. Only modify if the task requires specific changes")
    lines.append("4. Do not regenerate code that already exists\n")
    return "\n".join(lines)


def _safe_target(base_dir: Path, relative_path: str) -> Path:
    """Resolve a relative snippet path while preventing directory traversal."""
    resolved_base = base_dir.resolve()
    target = (resolved_base / relative_path).resolve()
    if target == resolved_base:
        return target
    if resolved_base not in target.parents:
        raise ValueError(
            f"Invalid snippet filename: {relative_path!r}. Expected a path inside {resolved_base.as_posix()}. "
            "Current tier/state: tier=unknown, path_outside_tempdir=true. "
            "Fix: ensure snippet filenames are relative paths."
        )
    return target


def _read_text_file(path: Path) -> str | None:
    """Read UTF-8 text content and skip likely-binary or undecodable files."""
    raw = path.read_bytes()
    if b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _hash_content(content: str) -> str:
    """Hash content deterministically for change detection across turns."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _snapshot_hashes(tempdir: Path) -> dict[str, str]:
    """Capture text-file hashes in a materialized temp directory."""
    hashes: dict[str, str] = {}
    if not tempdir.exists():
        return hashes
    for path in sorted(tempdir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        content = _read_text_file(path)
        if content is None:
            continue
        relative_path = path.relative_to(tempdir).as_posix()
        hashes[relative_path] = _hash_content(content)
    return hashes


def _collect_changed_files(tempdir: Path, baseline_hashes: dict[str, str]) -> tuple[list[FileWritten], dict[str, str]]:
    """Return changed/new files with updated hashes for upload."""
    changed_files: list[FileWritten] = []
    current_hashes: dict[str, str] = {}

    for path in sorted(tempdir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        content = _read_text_file(path)
        if content is None:
            continue
        relative_path = path.relative_to(tempdir).as_posix()
        content_hash = _hash_content(content)
        current_hashes[relative_path] = content_hash
        if baseline_hashes.get(relative_path) != content_hash:
            changed_files.append(FileWritten(path=relative_path, content=content))

    return changed_files, current_hashes


def _extract_attr_or_key(value: object, key: str) -> object | None:
    """Read a key-like field from either dict or object values."""
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _is_programmatic_code_execution_tools(tools: object) -> bool:
    """Return True when request tools include Anthropic code execution for programmatic tool calls."""
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_name = tool.get("name")
        tool_type = tool.get("type")
        if isinstance(tool_name, str) and tool_name == "code_execution":
            return True
        if isinstance(tool_type, str) and tool_type.startswith("code_execution_"):
            return True
    return False


def _merge_system_prompt(system_prompt: object, addition: str) -> object:
    """Append context prompt to Anthropic system payload for either string or block-list formats."""
    if not addition:
        return system_prompt
    if system_prompt is None:
        return addition
    if isinstance(system_prompt, str):
        return system_prompt + addition
    if isinstance(system_prompt, list):
        merged = list(system_prompt)
        merged.append({"type": "text", "text": addition})
        return merged
    return system_prompt


def _extract_container_id(response: object) -> str | None:
    """Extract Anthropic container id from response payload when present."""
    container = _extract_attr_or_key(response, "container")
    if isinstance(container, str) and container:
        return container
    if container is not None:
        nested_id = _extract_attr_or_key(container, "id")
        if isinstance(nested_id, str) and nested_id:
            return nested_id
    return None


def _extract_response_logs(response: object) -> list[str]:
    """Extract textual logs from Anthropic response blocks for upload context."""
    logs: list[str] = []
    content = _extract_attr_or_key(response, "content")
    if not isinstance(content, list):
        return logs
    for block in content:
        block_type = _extract_attr_or_key(block, "type")
        if block_type == "text":
            text = _extract_attr_or_key(block, "text")
            if isinstance(text, str) and text.strip():
                logs.append(text[:5000])
        elif block_type == "tool_result":
            result_content = _extract_attr_or_key(block, "content")
            if isinstance(result_content, str) and result_content.strip():
                logs.append(result_content[:5000])
    return logs


def _response_is_end_turn(response: object) -> bool:
    """Return True when Anthropic response reached an end_turn stop reason."""
    stop_reason = _extract_attr_or_key(response, "stop_reason")
    return stop_reason == "end_turn"


class AsyncProgrammaticToolCallingSession:
    """Materialize cache snippets for programmatic tool calling and upload changed files at finish."""

    def __init__(
        self,
        client: SupportsAsyncSearchUpload,
        *,
        top_k: int = DEFAULT_TOP_K,
        workspace_id: str | None = None,
        tempdir: str | None = None,
        min_verdict_score: float = DEFAULT_MIN_VERDICT_SCORE,
        prefer_complete: bool = True,
    ) -> None:
        """Create a session with materialize-first behavior and top-k context defaults."""
        self._client = client
        self._top_k = _validate_top_k(top_k)
        self._workspace_id = workspace_id
        self._min_verdict_score = min_verdict_score
        self._prefer_complete = prefer_complete
        self._owns_tempdir = tempdir is None
        self._tempdir = Path(tempdir) if tempdir is not None else Path(tempfile.mkdtemp(prefix="raysurfer_ptc_"))
        self._tempdir.mkdir(parents=True, exist_ok=True)
        self._baseline_hashes: dict[str, str] = {}
        self._context_prompt = ""
        self._files: list[CodeFile] = []
        self._execution_logs: list[str] = []
        self._prepared = False

    @property
    def tempdir(self) -> str:
        """Return the directory where snippets are materialized for this session."""
        return self._tempdir.as_posix()

    def append_log(self, log_line: str) -> None:
        """Record a log line that should be included when uploading changed files."""
        if log_line.strip():
            self._execution_logs.append(log_line)

    async def prepare_turn(self, task: str, *, first_message: bool = True) -> ProgrammaticMaterializeContext:
        """On first message, search+materialize top-k snippets; on later turns reuse existing tempdir."""
        if first_message:
            search_response = await self._client.search(
                task=task,
                top_k=self._top_k,
                min_verdict_score=self._min_verdict_score,
                prefer_complete=self._prefer_complete,
                workspace_id=self._workspace_id,
            )
            self._files = _build_code_files(search_response.matches)
            for code_file in self._files:
                target = _safe_target(self._tempdir, code_file.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(code_file.source, encoding="utf-8")
            self._context_prompt = _format_context_prompt(self._files, self._tempdir)
            self._baseline_hashes = _snapshot_hashes(self._tempdir)
            self._prepared = True
        elif not self._prepared:
            self._baseline_hashes = _snapshot_hashes(self._tempdir)
            self._prepared = True

        return ProgrammaticMaterializeContext(
            tempdir=self._tempdir.as_posix(),
            context_prompt=self._context_prompt,
            files=list(self._files),
            top_k=self._top_k,
            workspace_id=self._workspace_id,
        )

    async def upload_changed_code(
        self,
        task: str,
        *,
        succeeded: bool = True,
        execution_logs: str | None = None,
        use_raysurfer_ai_voting: bool = True,
    ) -> SubmitExecutionResultResponse | None:
        """Upload all changed files from tempdir as snippets, including execution logs."""
        changed_files, current_hashes = _collect_changed_files(self._tempdir, self._baseline_hashes)
        if not changed_files:
            self._baseline_hashes = current_hashes
            return None

        resolved_logs = execution_logs
        if resolved_logs is None and self._execution_logs:
            resolved_logs = "\n---\n".join(self._execution_logs)

        response = await self._client.upload(
            task=task,
            files_written=changed_files,
            succeeded=succeeded,
            use_raysurfer_ai_voting=use_raysurfer_ai_voting,
            execution_logs=resolved_logs,
            workspace_id=self._workspace_id,
        )

        self._baseline_hashes = current_hashes
        self._execution_logs.clear()
        return response

    def cleanup(self, *, remove_tempdir: bool = False) -> None:
        """Optionally remove the tempdir created by this session."""
        if remove_tempdir and self._owns_tempdir and self._tempdir.exists():
            shutil.rmtree(self._tempdir, ignore_errors=True)


class ProgrammaticToolCallingSession:
    """Synchronous variant of programmatic materialization and changed-code upload."""

    def __init__(
        self,
        client: SupportsSyncSearchUpload,
        *,
        top_k: int = DEFAULT_TOP_K,
        workspace_id: str | None = None,
        tempdir: str | None = None,
        min_verdict_score: float = DEFAULT_MIN_VERDICT_SCORE,
        prefer_complete: bool = True,
    ) -> None:
        """Create a sync session with materialize-first behavior and top-k context defaults."""
        self._client = client
        self._top_k = _validate_top_k(top_k)
        self._workspace_id = workspace_id
        self._min_verdict_score = min_verdict_score
        self._prefer_complete = prefer_complete
        self._owns_tempdir = tempdir is None
        self._tempdir = Path(tempdir) if tempdir is not None else Path(tempfile.mkdtemp(prefix="raysurfer_ptc_"))
        self._tempdir.mkdir(parents=True, exist_ok=True)
        self._baseline_hashes: dict[str, str] = {}
        self._context_prompt = ""
        self._files: list[CodeFile] = []
        self._execution_logs: list[str] = []
        self._prepared = False

    @property
    def tempdir(self) -> str:
        """Return the directory where snippets are materialized for this session."""
        return self._tempdir.as_posix()

    def append_log(self, log_line: str) -> None:
        """Record a log line that should be included when uploading changed files."""
        if log_line.strip():
            self._execution_logs.append(log_line)

    def prepare_turn(self, task: str, *, first_message: bool = True) -> ProgrammaticMaterializeContext:
        """On first message, search+materialize top-k snippets; on later turns reuse existing tempdir."""
        if first_message:
            search_response = self._client.search(
                task=task,
                top_k=self._top_k,
                min_verdict_score=self._min_verdict_score,
                prefer_complete=self._prefer_complete,
                workspace_id=self._workspace_id,
            )
            self._files = _build_code_files(search_response.matches)
            for code_file in self._files:
                target = _safe_target(self._tempdir, code_file.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(code_file.source, encoding="utf-8")
            self._context_prompt = _format_context_prompt(self._files, self._tempdir)
            self._baseline_hashes = _snapshot_hashes(self._tempdir)
            self._prepared = True
        elif not self._prepared:
            self._baseline_hashes = _snapshot_hashes(self._tempdir)
            self._prepared = True

        return ProgrammaticMaterializeContext(
            tempdir=self._tempdir.as_posix(),
            context_prompt=self._context_prompt,
            files=list(self._files),
            top_k=self._top_k,
            workspace_id=self._workspace_id,
        )

    def upload_changed_code(
        self,
        task: str,
        *,
        succeeded: bool = True,
        execution_logs: str | None = None,
        use_raysurfer_ai_voting: bool = True,
    ) -> SubmitExecutionResultResponse | None:
        """Upload all changed files from tempdir as snippets, including execution logs."""
        changed_files, current_hashes = _collect_changed_files(self._tempdir, self._baseline_hashes)
        if not changed_files:
            self._baseline_hashes = current_hashes
            return None

        resolved_logs = execution_logs
        if resolved_logs is None and self._execution_logs:
            resolved_logs = "\n---\n".join(self._execution_logs)

        response = self._client.upload(
            task=task,
            files_written=changed_files,
            succeeded=succeeded,
            use_raysurfer_ai_voting=use_raysurfer_ai_voting,
            execution_logs=resolved_logs,
            workspace_id=self._workspace_id,
        )

        self._baseline_hashes = current_hashes
        self._execution_logs.clear()
        return response

    def cleanup(self, *, remove_tempdir: bool = False) -> None:
        """Optionally remove the tempdir created by this session."""
        if remove_tempdir and self._owns_tempdir and self._tempdir.exists():
            shutil.rmtree(self._tempdir, ignore_errors=True)


async def run_async_framework_programmatic_tool_calling(
    raysurfer: SupportsAsyncSearchUpload,
    task: str,
    runner: SupportsAsyncFrameworkRunner,
    *,
    top_k: int = DEFAULT_TOP_K,
    workspace_id: str | None = None,
    tempdir: str | None = None,
    min_verdict_score: float = DEFAULT_MIN_VERDICT_SCORE,
    prefer_complete: bool = True,
    first_message: bool = True,
    use_raysurfer_ai_voting: bool = True,
) -> tuple[ProgrammaticFrameworkResult, SubmitExecutionResultResponse | None]:
    """Framework-agnostic async wrapper for LangSmith/LangChain/custom Anthropic programmatic loops."""
    session = AsyncProgrammaticToolCallingSession(
        raysurfer,
        top_k=top_k,
        workspace_id=workspace_id,
        tempdir=tempdir,
        min_verdict_score=min_verdict_score,
        prefer_complete=prefer_complete,
    )
    context = await session.prepare_turn(task, first_message=first_message)
    result = await runner(context)
    if result.execution_logs:
        session.append_log(result.execution_logs)
    upload_response = await session.upload_changed_code(
        task,
        succeeded=result.succeeded,
        execution_logs=result.execution_logs,
        use_raysurfer_ai_voting=use_raysurfer_ai_voting,
    )
    return result, upload_response


def run_framework_programmatic_tool_calling(
    raysurfer: SupportsSyncSearchUpload,
    task: str,
    runner: SupportsSyncFrameworkRunner,
    *,
    top_k: int = DEFAULT_TOP_K,
    workspace_id: str | None = None,
    tempdir: str | None = None,
    min_verdict_score: float = DEFAULT_MIN_VERDICT_SCORE,
    prefer_complete: bool = True,
    first_message: bool = True,
    use_raysurfer_ai_voting: bool = True,
) -> tuple[ProgrammaticFrameworkResult, SubmitExecutionResultResponse | None]:
    """Framework-agnostic sync wrapper for LangSmith/LangChain/custom Anthropic programmatic loops."""
    session = ProgrammaticToolCallingSession(
        raysurfer,
        top_k=top_k,
        workspace_id=workspace_id,
        tempdir=tempdir,
        min_verdict_score=min_verdict_score,
        prefer_complete=prefer_complete,
    )
    context = session.prepare_turn(task, first_message=first_message)
    result = runner(context)
    if result.execution_logs:
        session.append_log(result.execution_logs)
    upload_response = session.upload_changed_code(
        task,
        succeeded=result.succeeded,
        execution_logs=result.execution_logs,
        use_raysurfer_ai_voting=use_raysurfer_ai_voting,
    )
    return result, upload_response


class AsyncAnthropicProgrammaticToolCallWrapper:
    """Wrap any async Anthropic create callable with Raysurfer materialize+upload behavior."""

    def __init__(
        self,
        create_message: SupportsAsyncAnthropicCreateCallable,
        session: SupportsAsyncProgrammaticSession,
        task: str,
        *,
        upload_on_end_turn: bool = True,
        use_raysurfer_ai_voting: bool = True,
    ) -> None:
        """Create an async wrapper that handles first-turn cache search/materialization and end-turn uploads."""
        self._create_message = create_message
        self._session = session
        self._task = task
        self._upload_on_end_turn = upload_on_end_turn
        self._use_raysurfer_ai_voting = use_raysurfer_ai_voting
        self._prepared = False
        self._container_id: str | None = None

    async def create(self, **kwargs: object) -> object:
        """Call Anthropic create with automatic first-turn context and optional end-turn changed-code upload."""
        request_payload = dict(kwargs)
        uses_programmatic = _is_programmatic_code_execution_tools(request_payload.get("tools"))

        if uses_programmatic and not self._prepared:
            prepared = await self._session.prepare_turn(self._task, first_message=True)
            request_payload["system"] = _merge_system_prompt(request_payload.get("system"), prepared.context_prompt)
            self._prepared = True
        elif uses_programmatic and self._container_id and request_payload.get("container") is None:
            request_payload["container"] = self._container_id

        response = await self._create_message(**request_payload)

        container_id = _extract_container_id(response)
        if container_id:
            self._container_id = container_id

        for log_line in _extract_response_logs(response):
            self._session.append_log(log_line)

        if uses_programmatic and self._upload_on_end_turn and _response_is_end_turn(response):
            await self._session.upload_changed_code(
                self._task,
                succeeded=True,
                use_raysurfer_ai_voting=self._use_raysurfer_ai_voting,
            )

        return response

    async def finalize(
        self,
        *,
        succeeded: bool = True,
        execution_logs: str | None = None,
        use_raysurfer_ai_voting: bool | None = None,
    ) -> SubmitExecutionResultResponse | None:
        """Manually upload changed files and logs when caller controls finalization timing."""
        effective_ai_voting = (
            self._use_raysurfer_ai_voting if use_raysurfer_ai_voting is None else use_raysurfer_ai_voting
        )
        return await self._session.upload_changed_code(
            self._task,
            succeeded=succeeded,
            execution_logs=execution_logs,
            use_raysurfer_ai_voting=effective_ai_voting,
        )

    def cleanup(self, *, remove_tempdir: bool = False) -> None:
        """Forward cleanup to underlying materialization session."""
        self._session.cleanup(remove_tempdir=remove_tempdir)


class AnthropicProgrammaticToolCallWrapper:
    """Wrap any sync Anthropic create callable with Raysurfer materialize+upload behavior."""

    def __init__(
        self,
        create_message: SupportsSyncAnthropicCreateCallable,
        session: SupportsSyncProgrammaticSession,
        task: str,
        *,
        upload_on_end_turn: bool = True,
        use_raysurfer_ai_voting: bool = True,
    ) -> None:
        """Create a sync wrapper that handles first-turn cache search/materialization and end-turn uploads."""
        self._create_message = create_message
        self._session = session
        self._task = task
        self._upload_on_end_turn = upload_on_end_turn
        self._use_raysurfer_ai_voting = use_raysurfer_ai_voting
        self._prepared = False
        self._container_id: str | None = None

    def create(self, **kwargs: object) -> object:
        """Call Anthropic create with automatic first-turn context and optional end-turn changed-code upload."""
        request_payload = dict(kwargs)
        uses_programmatic = _is_programmatic_code_execution_tools(request_payload.get("tools"))

        if uses_programmatic and not self._prepared:
            prepared = self._session.prepare_turn(self._task, first_message=True)
            request_payload["system"] = _merge_system_prompt(request_payload.get("system"), prepared.context_prompt)
            self._prepared = True
        elif uses_programmatic and self._container_id and request_payload.get("container") is None:
            request_payload["container"] = self._container_id

        response = self._create_message(**request_payload)

        container_id = _extract_container_id(response)
        if container_id:
            self._container_id = container_id

        for log_line in _extract_response_logs(response):
            self._session.append_log(log_line)

        if uses_programmatic and self._upload_on_end_turn and _response_is_end_turn(response):
            self._session.upload_changed_code(
                self._task,
                succeeded=True,
                use_raysurfer_ai_voting=self._use_raysurfer_ai_voting,
            )

        return response

    def finalize(
        self,
        *,
        succeeded: bool = True,
        execution_logs: str | None = None,
        use_raysurfer_ai_voting: bool | None = None,
    ) -> SubmitExecutionResultResponse | None:
        """Manually upload changed files and logs when caller controls finalization timing."""
        effective_ai_voting = (
            self._use_raysurfer_ai_voting if use_raysurfer_ai_voting is None else use_raysurfer_ai_voting
        )
        return self._session.upload_changed_code(
            self._task,
            succeeded=succeeded,
            execution_logs=execution_logs,
            use_raysurfer_ai_voting=effective_ai_voting,
        )

    def cleanup(self, *, remove_tempdir: bool = False) -> None:
        """Forward cleanup to underlying materialization session."""
        self._session.cleanup(remove_tempdir=remove_tempdir)


def wrap_async_anthropic_programmatic_tool_calling(
    create_message: SupportsAsyncAnthropicCreateCallable,
    raysurfer: SupportsAsyncSearchUpload,
    task: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    workspace_id: str | None = None,
    tempdir: str | None = None,
    min_verdict_score: float = DEFAULT_MIN_VERDICT_SCORE,
    prefer_complete: bool = True,
    upload_on_end_turn: bool = True,
    use_raysurfer_ai_voting: bool = True,
) -> AsyncAnthropicProgrammaticToolCallWrapper:
    """Create an async wrapper for Anthropic create call paths with materialized Raysurfer caching."""
    session = AsyncProgrammaticToolCallingSession(
        raysurfer,
        top_k=top_k,
        workspace_id=workspace_id,
        tempdir=tempdir,
        min_verdict_score=min_verdict_score,
        prefer_complete=prefer_complete,
    )
    return AsyncAnthropicProgrammaticToolCallWrapper(
        create_message,
        session,
        task,
        upload_on_end_turn=upload_on_end_turn,
        use_raysurfer_ai_voting=use_raysurfer_ai_voting,
    )


def wrap_anthropic_programmatic_tool_calling(
    create_message: SupportsSyncAnthropicCreateCallable,
    raysurfer: SupportsSyncSearchUpload,
    task: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    workspace_id: str | None = None,
    tempdir: str | None = None,
    min_verdict_score: float = DEFAULT_MIN_VERDICT_SCORE,
    prefer_complete: bool = True,
    upload_on_end_turn: bool = True,
    use_raysurfer_ai_voting: bool = True,
) -> AnthropicProgrammaticToolCallWrapper:
    """Create a sync wrapper for Anthropic create call paths with materialized Raysurfer caching."""
    session = ProgrammaticToolCallingSession(
        raysurfer,
        top_k=top_k,
        workspace_id=workspace_id,
        tempdir=tempdir,
        min_verdict_score=min_verdict_score,
        prefer_complete=prefer_complete,
    )
    return AnthropicProgrammaticToolCallWrapper(
        create_message,
        session,
        task,
        upload_on_end_turn=upload_on_end_turn,
        use_raysurfer_ai_voting=use_raysurfer_ai_voting,
    )
