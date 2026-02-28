"""Tests for agent-accessible decorators and raysurfer.yaml loader."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

from raysurfer.accessible import agent_accessible, publish_function_registry
from raysurfer.config import load_config


class _FakeUploadResponse:
    def __init__(self, snippet_name: str):
        self.snippet_name = snippet_name


class _FakeClient:
    def __init__(self) -> None:
        self.upload_calls: list[dict[str, object]] = []
        self.execution_calls: list[dict[str, object]] = []

    async def upload_new_code_snip(
        self,
        *,
        task: str,
        file_written: object,
        succeeded: bool,
        use_raysurfer_ai_voting: bool = True,
        tags: list[str] | None = None,
    ) -> _FakeUploadResponse:
        self.upload_calls.append(
            {
                "task": task,
                "file_written": file_written,
                "succeeded": succeeded,
                "use_raysurfer_ai_voting": use_raysurfer_ai_voting,
                "tags": tags or [],
            }
        )
        return _FakeUploadResponse("registry_fn")

    async def store_execution(self, **kwargs: object) -> dict[str, object]:
        self.execution_calls.append(kwargs)
        return {"success": True}


@pytest.mark.asyncio
async def test_publish_function_registry_adds_tags_and_tracks_usage() -> None:
    client = _FakeClient()

    @agent_accessible("Greets a user")
    def greet(name: str) -> str:
        return f"hi {name}"

    snippet_names = await publish_function_registry(client, [greet])
    assert snippet_names == ["registry_fn"]
    assert client.upload_calls[0]["tags"] == ["function_registry", "agent_accessible"]

    assert greet("ray") == "hi ray"
    await asyncio.sleep(0.01)

    assert len(client.execution_calls) == 1
    assert client.execution_calls[0]["triggering_task"] == "agent_accessible:greet"


def test_load_config_marks_matching_functions(tmp_path: Path) -> None:
    module_path = tmp_path / "sample_module.py"
    module_path.write_text(
        "def allowed_task(name: str) -> str:\n"
        "    return f'allowed-{name}'\n\n"
        "def blocked_task(name: str) -> str:\n"
        "    return f'blocked-{name}'\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "raysurfer.yaml"
    config_path.write_text(
        "agent_access:\n"
        '  call: ["sample_module.py:allowed_*", "sample_module.py:blocked_task"]\n'
        '  deny: ["sample_module.py:blocked_task"]\n',
        encoding="utf-8",
    )

    spec = importlib.util.spec_from_file_location("sample_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    functions = load_config(str(config_path), modules=[module])
    function_names = [fn.__name__ for fn in functions]

    assert function_names == ["allowed_task"]
    assert bool(getattr(functions[0], "_raysurfer_accessible", False))
