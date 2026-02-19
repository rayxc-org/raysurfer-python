"""High-level decorator-friendly codegen app wrappers."""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType

from raysurfer.client import DEFAULT_BASE_URL, AsyncRaySurfer, RaySurfer
from raysurfer.types import ExecuteResult, JsonValue, SnipsDesired

DEFAULT_CODEGEN_MODEL = "claude-opus-4-6"
DEFAULT_EXECUTION_TIMEOUT_SECONDS = 300
CODEGEN_DOCS_URL = "https://docs.raysurfer.com/sdk/python#programmatic-tool-calling"


def _missing_codegen_key_error(value: object) -> ValueError:
    return ValueError(
        f"Invalid codegen_api_key value: {value!r}. Expected format: non-empty provider API key string. "
        "Current tier/state: tier=unknown, codegen_api_key_missing=true (not configured on app and not "
        f"provided at call-time). Fix: pass codegen_api_key in app config or run(...). Docs: {CODEGEN_DOCS_URL}"
    )


def _invalid_codegen_prompt_error(value: object) -> ValueError:
    return ValueError(
        f"Invalid codegen_prompt value: {value!r}. Expected format: non-empty prompt string. "
        "Current tier/state: tier=unknown, codegen_prompt_invalid=true. "
        f"Fix: pass codegen_prompt or provide a non-empty task. Docs: {CODEGEN_DOCS_URL}"
    )


class AsyncCodegenApp:
    def __init__(
        self,
        *,
        raysurfer: AsyncRaySurfer | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        organization_id: str | None = None,
        workspace_id: str | None = None,
        snips_desired: SnipsDesired | str | None = None,
        public_snips: bool = False,
        agent_id: str | None = None,
        codegen_api_key: str | None = None,
        codegen_model: str = DEFAULT_CODEGEN_MODEL,
        execution_timeout_seconds: int = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    ) -> None:
        """Create an async codegen app with decorator-style tool registration."""
        self._owns_client = raysurfer is None
        self._raysurfer = raysurfer or AsyncRaySurfer(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            organization_id=organization_id,
            workspace_id=workspace_id,
            snips_desired=snips_desired,
            public_snips=public_snips,
            agent_id=agent_id,
        )
        self._default_codegen_api_key = codegen_api_key
        self._default_codegen_model = codegen_model
        self._default_execution_timeout_seconds = execution_timeout_seconds

    @property
    def raysurfer(self) -> AsyncRaySurfer:
        """Return the underlying async Raysurfer client."""
        return self._raysurfer

    async def __aenter__(self) -> AsyncCodegenApp:
        """Return the app instance for async context manager usage."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the owned client when exiting context."""
        if self._owns_client:
            await self._raysurfer.close()

    def tool(self, fn: Callable[..., JsonValue]) -> Callable[..., JsonValue]:
        """Register a decorated function as a callable tool."""
        return self._raysurfer.tool(fn)

    async def run(
        self,
        task: str,
        *,
        codegen_prompt: str | None = None,
        codegen_api_key: str | None = None,
        codegen_model: str | None = None,
        execution_timeout_seconds: int | None = None,
    ) -> ExecuteResult:
        """Run a task via sandbox code generation with registered tools."""
        resolved_key = self._resolve_codegen_api_key(codegen_api_key)
        resolved_prompt = self._resolve_codegen_prompt(task, codegen_prompt)
        resolved_model = codegen_model or self._default_codegen_model
        resolved_timeout = execution_timeout_seconds or self._default_execution_timeout_seconds
        return await self._raysurfer.execute_with_sandbox_codegen(
            task=task,
            codegen_api_key=resolved_key,
            codegen_prompt=resolved_prompt,
            timeout=resolved_timeout,
            codegen_model=resolved_model,
        )

    async def run_generated_code(
        self,
        task: str,
        user_code: str,
        *,
        execution_timeout_seconds: int | None = None,
    ) -> ExecuteResult:
        """Run explicit user-provided code in the sandbox with registered tools."""
        resolved_timeout = execution_timeout_seconds or self._default_execution_timeout_seconds
        return await self._raysurfer.execute_generated_code(
            task=task,
            user_code=user_code,
            timeout=resolved_timeout,
        )

    def _resolve_codegen_api_key(self, codegen_api_key: str | None) -> str:
        """Resolve and validate the effective codegen API key."""
        value = codegen_api_key if codegen_api_key is not None else self._default_codegen_api_key
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
        raise _missing_codegen_key_error(value)

    def _resolve_codegen_prompt(self, task: str, codegen_prompt: str | None) -> str:
        """Resolve and validate the effective code generation prompt."""
        candidate = codegen_prompt if codegen_prompt is not None else task
        if isinstance(candidate, str):
            stripped = candidate.strip()
            if stripped:
                return stripped
        raise _invalid_codegen_prompt_error(candidate)


class CodegenApp:
    def __init__(
        self,
        *,
        raysurfer: RaySurfer | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        organization_id: str | None = None,
        workspace_id: str | None = None,
        snips_desired: SnipsDesired | str | None = None,
        public_snips: bool = False,
        agent_id: str | None = None,
        codegen_api_key: str | None = None,
        codegen_model: str = DEFAULT_CODEGEN_MODEL,
        execution_timeout_seconds: int = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    ) -> None:
        """Create a sync codegen app with decorator-style tool registration."""
        self._owns_client = raysurfer is None
        self._raysurfer = raysurfer or RaySurfer(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            organization_id=organization_id,
            workspace_id=workspace_id,
            snips_desired=snips_desired,
            public_snips=public_snips,
            agent_id=agent_id,
        )
        self._default_codegen_api_key = codegen_api_key
        self._default_codegen_model = codegen_model
        self._default_execution_timeout_seconds = execution_timeout_seconds

    @property
    def raysurfer(self) -> RaySurfer:
        """Return the underlying sync Raysurfer client."""
        return self._raysurfer

    def __enter__(self) -> CodegenApp:
        """Return the app instance for sync context manager usage."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the owned client when exiting context."""
        if self._owns_client:
            self._raysurfer.close()

    def tool(self, fn: Callable[..., JsonValue]) -> Callable[..., JsonValue]:
        """Register a decorated function as a callable tool."""
        return self._raysurfer.tool(fn)

    def run(
        self,
        task: str,
        *,
        codegen_prompt: str | None = None,
        codegen_api_key: str | None = None,
        codegen_model: str | None = None,
        execution_timeout_seconds: int | None = None,
    ) -> ExecuteResult:
        """Run a task via sandbox code generation with registered tools."""
        resolved_key = self._resolve_codegen_api_key(codegen_api_key)
        resolved_prompt = self._resolve_codegen_prompt(task, codegen_prompt)
        resolved_model = codegen_model or self._default_codegen_model
        resolved_timeout = execution_timeout_seconds or self._default_execution_timeout_seconds
        return self._raysurfer.execute_with_sandbox_codegen(
            task=task,
            codegen_api_key=resolved_key,
            codegen_prompt=resolved_prompt,
            timeout=resolved_timeout,
            codegen_model=resolved_model,
        )

    def run_generated_code(
        self,
        task: str,
        user_code: str,
        *,
        execution_timeout_seconds: int | None = None,
    ) -> ExecuteResult:
        """Run explicit user-provided code in the sandbox with registered tools."""
        resolved_timeout = execution_timeout_seconds or self._default_execution_timeout_seconds
        return self._raysurfer.execute_generated_code(
            task=task,
            user_code=user_code,
            timeout=resolved_timeout,
        )

    def _resolve_codegen_api_key(self, codegen_api_key: str | None) -> str:
        """Resolve and validate the effective codegen API key."""
        value = codegen_api_key if codegen_api_key is not None else self._default_codegen_api_key
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
        raise _missing_codegen_key_error(value)

    def _resolve_codegen_prompt(self, task: str, codegen_prompt: str | None) -> str:
        """Resolve and validate the effective code generation prompt."""
        candidate = codegen_prompt if codegen_prompt is not None else task
        if isinstance(candidate, str):
            stripped = candidate.strip()
            if stripped:
                return stripped
        raise _invalid_codegen_prompt_error(candidate)
