"""Tests for programmatic tool-calling materialize/upload helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from raysurfer.programmatic import (
    AnthropicProgrammaticToolCallWrapper,
    AsyncAnthropicProgrammaticToolCallWrapper,
    AsyncProgrammaticToolCallingSession,
    ProgrammaticFrameworkResult,
    ProgrammaticToolCallingSession,
    run_async_framework_programmatic_tool_calling,
    run_framework_programmatic_tool_calling,
)
from raysurfer.types import (
    CodeBlock,
    FileWritten,
    SearchMatch,
    SearchResponse,
    SubmitExecutionResultResponse,
)


def _build_search_response() -> SearchResponse:
    return SearchResponse(
        matches=[
            SearchMatch(
                code_block=CodeBlock(
                    id="cb_1",
                    name="example.py",
                    description="Example cached file",
                    source="print('cached')\n",
                    entrypoint="main",
                    language="python",
                ),
                score=0.91,
                thumbs_up=5,
                thumbs_down=0,
                filename="example.py",
                language="python",
                entrypoint="main",
                dependencies={"requests": "2.32.0"},
            )
        ],
        total_found=1,
        cache_hit=True,
    )


class _FakeAsyncClient:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, object]] = []
        self.upload_calls: list[dict[str, object]] = []

    async def search(
        self,
        *,
        task: str,
        top_k: int = 5,
        min_verdict_score: float = 0.3,
        prefer_complete: bool = True,
        workspace_id: str | None = None,
    ) -> SearchResponse:
        self.search_calls.append(
            {
                "task": task,
                "top_k": top_k,
                "min_verdict_score": min_verdict_score,
                "prefer_complete": prefer_complete,
                "workspace_id": workspace_id,
            }
        )
        return _build_search_response()

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
        self.upload_calls.append(
            {
                "task": task,
                "files_written": files_written or [],
                "succeeded": succeeded,
                "use_raysurfer_ai_voting": use_raysurfer_ai_voting,
                "execution_logs": execution_logs,
                "workspace_id": workspace_id,
            }
        )
        return SubmitExecutionResultResponse(
            success=True,
            code_blocks_stored=len(files_written or []),
            message="ok",
        )


class _FakeSyncClient:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, object]] = []
        self.upload_calls: list[dict[str, object]] = []

    def search(
        self,
        *,
        task: str,
        top_k: int = 5,
        min_verdict_score: float = 0.3,
        prefer_complete: bool = True,
        workspace_id: str | None = None,
    ) -> SearchResponse:
        self.search_calls.append(
            {
                "task": task,
                "top_k": top_k,
                "min_verdict_score": min_verdict_score,
                "prefer_complete": prefer_complete,
                "workspace_id": workspace_id,
            }
        )
        return _build_search_response()

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
        self.upload_calls.append(
            {
                "task": task,
                "files_written": files_written or [],
                "succeeded": succeeded,
                "use_raysurfer_ai_voting": use_raysurfer_ai_voting,
                "execution_logs": execution_logs,
                "workspace_id": workspace_id,
            }
        )
        return SubmitExecutionResultResponse(
            success=True,
            code_blocks_stored=len(files_written or []),
            message="ok",
        )


class _FakeContainer:
    def __init__(self, value: str):
        self.id = value


class _FakeBlock:
    def __init__(self, block_type: str, text: str = ""):
        self.type = block_type
        self.text = text


class _FakeResponse:
    def __init__(self, *, stop_reason: str, content: list[_FakeBlock], container_id: str | None = None):
        self.stop_reason = stop_reason
        self.content = content
        self.container = _FakeContainer(container_id) if container_id is not None else None


class _FakeAsyncSession:
    def __init__(self) -> None:
        self.prepare_calls: list[dict[str, object]] = []
        self.logs: list[str] = []
        self.upload_calls: list[dict[str, object]] = []

    async def prepare_turn(self, task: str, *, first_message: bool = True):
        self.prepare_calls.append({"task": task, "first_message": first_message})
        return type(
            "Prepared",
            (),
            {
                "context_prompt": "\nCTX",
            },
        )()

    def append_log(self, log_line: str) -> None:
        self.logs.append(log_line)

    async def upload_changed_code(
        self,
        task: str,
        *,
        succeeded: bool = True,
        execution_logs: str | None = None,
        use_raysurfer_ai_voting: bool = True,
    ) -> SubmitExecutionResultResponse:
        self.upload_calls.append(
            {
                "task": task,
                "succeeded": succeeded,
                "execution_logs": execution_logs,
                "use_raysurfer_ai_voting": use_raysurfer_ai_voting,
            }
        )
        return SubmitExecutionResultResponse(success=True, code_blocks_stored=1, message="ok")

    def cleanup(self, *, remove_tempdir: bool = False) -> None:
        return None


class _FakeSyncSession:
    def __init__(self) -> None:
        self.prepare_calls: list[dict[str, object]] = []
        self.logs: list[str] = []
        self.upload_calls: list[dict[str, object]] = []

    def prepare_turn(self, task: str, *, first_message: bool = True):
        self.prepare_calls.append({"task": task, "first_message": first_message})
        return type(
            "Prepared",
            (),
            {
                "context_prompt": "\nCTX",
            },
        )()

    def append_log(self, log_line: str) -> None:
        self.logs.append(log_line)

    def upload_changed_code(
        self,
        task: str,
        *,
        succeeded: bool = True,
        execution_logs: str | None = None,
        use_raysurfer_ai_voting: bool = True,
    ) -> SubmitExecutionResultResponse:
        self.upload_calls.append(
            {
                "task": task,
                "succeeded": succeeded,
                "execution_logs": execution_logs,
                "use_raysurfer_ai_voting": use_raysurfer_ai_voting,
            }
        )
        return SubmitExecutionResultResponse(success=True, code_blocks_stored=1, message="ok")

    def cleanup(self, *, remove_tempdir: bool = False) -> None:
        return None


@pytest.mark.asyncio
async def test_async_programmatic_session_materializes_and_uploads_changes(tmp_path: Path) -> None:
    fake_client = _FakeAsyncClient()
    session = AsyncProgrammaticToolCallingSession(
        fake_client,
        workspace_id="ws_demo",
        tempdir=(tmp_path / "ptc").as_posix(),
    )

    context = await session.prepare_turn("Generate quarterly report", first_message=True)
    assert context.top_k == 5
    assert "example.py" in context.context_prompt
    assert fake_client.search_calls[0]["workspace_id"] == "ws_demo"

    materialized = tmp_path / "ptc" / "example.py"
    assert materialized.exists()
    materialized.write_text("print('changed')\n", encoding="utf-8")
    (tmp_path / "ptc" / "new.py").write_text("print('new')\n", encoding="utf-8")

    session.append_log("stdout line")
    upload_response = await session.upload_changed_code("Generate quarterly report")

    assert upload_response is not None
    assert upload_response.code_blocks_stored == 2
    upload_call = fake_client.upload_calls[0]
    uploaded_paths = sorted(file.path for file in upload_call["files_written"])
    assert uploaded_paths == ["example.py", "new.py"]
    assert upload_call["workspace_id"] == "ws_demo"
    assert upload_call["execution_logs"] == "stdout line"


def test_sync_programmatic_session_reuses_existing_tempdir_when_not_first_message(tmp_path: Path) -> None:
    fake_client = _FakeSyncClient()
    tempdir = tmp_path / "existing_ptc"
    tempdir.mkdir(parents=True, exist_ok=True)
    file_path = tempdir / "script.py"
    file_path.write_text("print('v1')\n", encoding="utf-8")

    session = ProgrammaticToolCallingSession(
        fake_client,
        tempdir=tempdir.as_posix(),
    )

    context = session.prepare_turn("Follow-up turn", first_message=False)
    assert context.tempdir == tempdir.as_posix()
    assert fake_client.search_calls == []

    file_path.write_text("print('v2')\n", encoding="utf-8")
    session.append_log("follow-up log")
    upload_response = session.upload_changed_code("Follow-up turn")

    assert upload_response is not None
    assert upload_response.code_blocks_stored == 1
    upload_call = fake_client.upload_calls[0]
    assert [file.path for file in upload_call["files_written"]] == ["script.py"]
    assert upload_call["execution_logs"] == "follow-up log"


@pytest.mark.asyncio
async def test_async_wrapper_handles_first_turn_container_and_end_turn_upload() -> None:
    async_session = _FakeAsyncSession()
    captured_requests: list[dict[str, object]] = []
    response_index = 0

    async def fake_create(**kwargs: object) -> object:
        nonlocal response_index
        captured_requests.append(dict(kwargs))
        if response_index == 0:
            response_index += 1
            return _FakeResponse(
                stop_reason="tool_use",
                content=[_FakeBlock("text", "thinking one")],
                container_id="cont_123",
            )
        return _FakeResponse(
            stop_reason="end_turn",
            content=[_FakeBlock("text", "final answer")],
            container_id="cont_123",
        )

    wrapper = AsyncAnthropicProgrammaticToolCallWrapper(
        fake_create,
        async_session,
        "Task A",
    )

    tools = [{"type": "code_execution_20250825", "name": "code_execution"}]
    await wrapper.create(messages=[{"role": "user", "content": "hi"}], tools=tools, system="BASE")
    await wrapper.create(messages=[{"role": "user", "content": "next"}], tools=tools)

    assert async_session.prepare_calls == [{"task": "Task A", "first_message": True}]
    assert isinstance(captured_requests[0]["system"], str)
    assert captured_requests[0]["system"] == "BASE\nCTX"
    assert captured_requests[1]["container"] == "cont_123"
    assert async_session.logs == ["thinking one", "final answer"]
    assert len(async_session.upload_calls) == 1
    assert async_session.upload_calls[0]["task"] == "Task A"


def test_sync_wrapper_handles_first_turn_container_and_end_turn_upload() -> None:
    sync_session = _FakeSyncSession()
    captured_requests: list[dict[str, object]] = []
    response_index = 0

    def fake_create(**kwargs: object) -> object:
        nonlocal response_index
        captured_requests.append(dict(kwargs))
        if response_index == 0:
            response_index += 1
            return _FakeResponse(
                stop_reason="tool_use",
                content=[_FakeBlock("text", "thinking one")],
                container_id="cont_sync",
            )
        return _FakeResponse(
            stop_reason="end_turn",
            content=[_FakeBlock("text", "final sync answer")],
            container_id="cont_sync",
        )

    wrapper = AnthropicProgrammaticToolCallWrapper(
        fake_create,
        sync_session,
        "Task Sync",
    )

    tools = [{"type": "code_execution_20250825", "name": "code_execution"}]
    wrapper.create(messages=[{"role": "user", "content": "hi"}], tools=tools, system="BASE")
    wrapper.create(messages=[{"role": "user", "content": "next"}], tools=tools)

    assert sync_session.prepare_calls == [{"task": "Task Sync", "first_message": True}]
    assert isinstance(captured_requests[0]["system"], str)
    assert captured_requests[0]["system"] == "BASE\nCTX"
    assert captured_requests[1]["container"] == "cont_sync"
    assert sync_session.logs == ["thinking one", "final sync answer"]
    assert len(sync_session.upload_calls) == 1
    assert sync_session.upload_calls[0]["task"] == "Task Sync"


@pytest.mark.asyncio
async def test_async_framework_runner_wrapper_supports_framework_agnostic_loops(
    tmp_path: Path,
) -> None:
    fake_client = _FakeAsyncClient()

    async def fake_runner(context) -> ProgrammaticFrameworkResult:
        out_file = Path(context.tempdir) / "framework_async.py"
        out_file.write_text("print('framework async')\n", encoding="utf-8")
        return ProgrammaticFrameworkResult(
            succeeded=True,
            execution_logs="langsmith async run complete",
        )

    result, upload_response = await run_async_framework_programmatic_tool_calling(
        fake_client,
        "Framework task async",
        fake_runner,
        tempdir=(tmp_path / "framework_async").as_posix(),
        workspace_id="ws_framework",
    )

    assert result.succeeded is True
    assert upload_response is not None
    assert upload_response.code_blocks_stored == 1
    assert fake_client.search_calls[0]["workspace_id"] == "ws_framework"
    assert fake_client.upload_calls[0]["execution_logs"] == "langsmith async run complete"


def test_sync_framework_runner_wrapper_supports_framework_agnostic_loops(
    tmp_path: Path,
) -> None:
    fake_client = _FakeSyncClient()

    def fake_runner(context) -> ProgrammaticFrameworkResult:
        out_file = Path(context.tempdir) / "framework_sync.py"
        out_file.write_text("print('framework sync')\n", encoding="utf-8")
        return ProgrammaticFrameworkResult(
            succeeded=True,
            execution_logs="langsmith sync run complete",
        )

    result, upload_response = run_framework_programmatic_tool_calling(
        fake_client,
        "Framework task sync",
        fake_runner,
        tempdir=(tmp_path / "framework_sync").as_posix(),
        workspace_id="ws_framework_sync",
    )

    assert result.succeeded is True
    assert upload_response is not None
    assert upload_response.code_blocks_stored == 1
    assert fake_client.search_calls[0]["workspace_id"] == "ws_framework_sync"
    assert fake_client.upload_calls[0]["execution_logs"] == "langsmith sync run complete"
