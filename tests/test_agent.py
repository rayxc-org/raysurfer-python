"""Tests for high-level codegen app wrappers."""

from __future__ import annotations

import pytest

from raysurfer.agent import AsyncCodegenApp, CodegenApp
from raysurfer.client import AsyncRaySurfer, RaySurfer
from raysurfer.types import ExecuteResult


def _make_execute_result(result_text: str) -> ExecuteResult:
    return ExecuteResult(
        execution_id="exec_123",
        result=result_text,
        exit_code=0,
        duration_ms=42,
        cache_hit=False,
        code_block_id=None,
        error=None,
        tool_calls=[],
    )


@pytest.mark.asyncio
async def test_async_codegen_app_run_uses_default_prompt_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AsyncRaySurfer(api_key="test-key", base_url="http://test.local")
    app = AsyncCodegenApp(
        raysurfer=client,
        codegen_api_key="anthropic-key",
        codegen_model="model-default",
        execution_timeout_seconds=111,
    )

    captured: dict[str, object] = {}

    async def fake_execute_with_sandbox_codegen(
        task: str,
        codegen_api_key: str,
        codegen_prompt: str,
        timeout: int = 300,
        codegen_model: str = "claude-opus-4-6",
    ) -> ExecuteResult:
        captured["task"] = task
        captured["codegen_api_key"] = codegen_api_key
        captured["codegen_prompt"] = codegen_prompt
        captured["timeout"] = timeout
        captured["codegen_model"] = codegen_model
        return _make_execute_result("ok")

    monkeypatch.setattr(client, "execute_with_sandbox_codegen", fake_execute_with_sandbox_codegen)

    result = await app.run("Summarize quarterly data")

    assert result.result == "ok"
    assert captured["task"] == "Summarize quarterly data"
    assert captured["codegen_api_key"] == "anthropic-key"
    assert captured["codegen_prompt"] == "Summarize quarterly data"
    assert captured["timeout"] == 111
    assert captured["codegen_model"] == "model-default"


@pytest.mark.asyncio
async def test_async_codegen_app_tool_delegates_to_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AsyncRaySurfer(api_key="test-key", base_url="http://test.local")
    app = AsyncCodegenApp(raysurfer=client, codegen_api_key="anthropic-key")

    captured: dict[str, object] = {}

    def fake_tool(fn):
        captured["name"] = fn.__name__
        return fn

    monkeypatch.setattr(client, "tool", fake_tool)

    @app.tool
    def get_weather(city: str) -> str:
        return city

    assert captured["name"] == "get_weather"
    assert get_weather("sf") == "sf"


def test_codegen_app_run_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RaySurfer(api_key="test-key", base_url="http://test.local")
    app = CodegenApp(
        raysurfer=client,
        codegen_api_key="default-key",
        codegen_model="default-model",
        execution_timeout_seconds=120,
    )

    captured: dict[str, object] = {}

    def fake_execute_with_sandbox_codegen(
        task: str,
        codegen_api_key: str,
        codegen_prompt: str,
        timeout: int = 300,
        codegen_model: str = "claude-opus-4-6",
    ) -> ExecuteResult:
        captured["task"] = task
        captured["codegen_api_key"] = codegen_api_key
        captured["codegen_prompt"] = codegen_prompt
        captured["timeout"] = timeout
        captured["codegen_model"] = codegen_model
        return _make_execute_result("sync-ok")

    monkeypatch.setattr(client, "execute_with_sandbox_codegen", fake_execute_with_sandbox_codegen)

    result = app.run(
        "Find bugs",
        codegen_prompt="Use static analysis",
        codegen_api_key="override-key",
        codegen_model="override-model",
        execution_timeout_seconds=10,
    )

    assert result.result == "sync-ok"
    assert captured["task"] == "Find bugs"
    assert captured["codegen_api_key"] == "override-key"
    assert captured["codegen_prompt"] == "Use static analysis"
    assert captured["timeout"] == 10
    assert captured["codegen_model"] == "override-model"


def test_codegen_app_missing_codegen_api_key_raises() -> None:
    client = RaySurfer(api_key="test-key", base_url="http://test.local")
    app = CodegenApp(raysurfer=client)

    with pytest.raises(ValueError, match="codegen_api_key"):
        app.run("Do something")
