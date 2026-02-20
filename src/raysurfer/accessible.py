"""Agent-accessible function decorator and registry helpers for Raysurfer."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Protocol, get_type_hints

from raysurfer.types import ExecutionState, FileWritten

# Python type -> JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class SupportsRegistryClient(Protocol):
    """Minimal client protocol used by registry/usage helpers."""

    def upload_new_code_snip(
        self,
        *,
        task: str,
        file_written: FileWritten,
        succeeded: bool,
        use_raysurfer_ai_voting: bool = True,
        tags: list[str] | None = None,
    ) -> object:
        """Upload a snippet and return a response object with snippet_name."""

    def store_execution(
        self,
        *,
        code_block_id: str,
        triggering_task: str,
        input_data: dict[str, object],
        output_data: object,
        execution_state: ExecutionState,
        duration_ms: int,
        error_message: str | None = None,
    ) -> object:
        """Store an execution record for usage tracking."""


def _json_safe(value: object) -> object:
    """Convert arbitrary Python objects to JSON-safe structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _build_input_schema(func: Callable[..., object]) -> dict[str, object]:
    """Build JSON Schema from function signature and type hints."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        json_type = "string"
        hint = hints.get(name)
        if hint is not None:
            args = getattr(hint, "__args__", ())
            if args and type(None) in args:
                hint = next((a for a in args if a is not type(None)), hint)
            json_type = _TYPE_MAP.get(hint, "string")

        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, object] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def _is_accessible(func: Callable[..., object]) -> bool:
    """Return whether a function is marked as agent-accessible."""
    return bool(getattr(func, "_raysurfer_accessible", False))


def _get_schema(func: Callable[..., object]) -> dict[str, object] | None:
    """Return attached Raysurfer metadata for a function."""
    schema = getattr(func, "_raysurfer_schema", None)
    if isinstance(schema, dict):
        return schema
    return None


def _set_schema(func: Callable[..., object], schema: dict[str, object]) -> None:
    """Attach Raysurfer metadata to a function."""
    setattr(func, "_raysurfer_schema", schema)


def _get_tracking_client(func: Callable[..., object]) -> SupportsRegistryClient | None:
    """Return the tracking client attached to a function, if present."""
    client = getattr(func, "_raysurfer_client", None)
    if client is None:
        return None
    return client


def _schedule(coro: Awaitable[None]) -> None:
    """Schedule a coroutine from sync code, running immediately if no loop exists."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return
    loop.create_task(coro)


async def _record_usage(
    func: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    result: object,
    error: Exception | None,
    duration_ms: int,
) -> None:
    """Store an execution trace for a decorated function call."""
    client = _get_tracking_client(func)
    if client is None:
        return

    schema = _get_schema(func)
    if schema is None:
        return

    name = str(schema.get("name", getattr(func, "__name__", "unknown")))
    code_block_id = str(schema.get("code_block_id", f"function_registry:{name}"))
    input_data = {
        "args": _json_safe(list(args)),
        "kwargs": _json_safe(kwargs),
    }
    output_data = _json_safe(result) if error is None else {"error": str(error)}
    state = ExecutionState.COMPLETED if error is None else ExecutionState.ERRORED

    try:
        maybe_result = client.store_execution(
            code_block_id=code_block_id,
            triggering_task=f"agent_accessible:{name}",
            input_data=input_data,
            output_data=output_data,
            execution_state=state,
            duration_ms=duration_ms,
            error_message=str(error) if error is not None else None,
        )
        if inspect.isawaitable(maybe_result):
            await maybe_result
    except Exception:
        # Usage tracking is best-effort and should never break function execution.
        return


def agent_accessible(description: str | None = None) -> Callable[..., object]:
    """Mark a function as callable by agents and attach metadata for Raysurfer."""

    def decorator(func: Callable[..., object]) -> Callable[..., object]:
        schema = {
            "name": func.__name__,
            "description": description or func.__doc__ or "",
            "input_schema": _build_input_schema(func),
            "source": inspect.getsource(func),
        }

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapped(*args: object, **kwargs: object) -> object:
                started = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:
                    duration_ms = int((time.perf_counter() - started) * 1000)
                    await _record_usage(async_wrapped, args, kwargs, None, exc, duration_ms)
                    raise
                duration_ms = int((time.perf_counter() - started) * 1000)
                await _record_usage(async_wrapped, args, kwargs, result, None, duration_ms)
                return result

            wrapped: Callable[..., object] = async_wrapped
        else:

            @wraps(func)
            def sync_wrapped(*args: object, **kwargs: object) -> object:
                started = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    duration_ms = int((time.perf_counter() - started) * 1000)
                    _schedule(_record_usage(sync_wrapped, args, kwargs, None, exc, duration_ms))
                    raise
                duration_ms = int((time.perf_counter() - started) * 1000)
                _schedule(_record_usage(sync_wrapped, args, kwargs, result, None, duration_ms))
                return result

            wrapped = sync_wrapped

        setattr(wrapped, "_raysurfer_accessible", True)
        _set_schema(wrapped, schema)
        return wrapped

    return decorator


async def publish_function_registry(
    client: SupportsRegistryClient,
    functions: list[Callable[..., object]],
) -> list[str]:
    """Batch-upload @agent_accessible functions as code blocks to Raysurfer."""
    snippet_names: list[str] = []
    for func in functions:
        if not _is_accessible(func):
            continue

        schema = _get_schema(func)
        if schema is None:
            continue

        upload_result = client.upload_new_code_snip(
            task=f"Call {schema['name']}: {schema['description']}",
            file_written=FileWritten(
                path=f"{schema['name']}.py",
                content=str(schema["source"]),
            ),
            succeeded=True,
            use_raysurfer_ai_voting=False,
            tags=["function_registry", "agent_accessible"],
        )
        if inspect.isawaitable(upload_result):
            response = await upload_result
        else:
            response = upload_result

        snippet_name = getattr(response, "snippet_name", None)
        if isinstance(snippet_name, str) and snippet_name:
            snippet_names.append(snippet_name)
            schema["code_block_id"] = snippet_name
            _set_schema(func, schema)

        set_tracking_client(func, client)

    return snippet_names


def set_tracking_client(func: Callable[..., object], client: SupportsRegistryClient) -> None:
    """Attach a Raysurfer client to a decorated function for usage tracking."""
    setattr(func, "_raysurfer_client", client)
