"""Agent-accessible function decorator for Raysurfer."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import get_type_hints

from raysurfer.types import FileWritten

# Python type -> JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


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
            # Handle Optional[X] (Union[X, None])
            origin = getattr(hint, "__origin__", None)
            if origin is type(None):
                continue
            # Unwrap Optional
            args = getattr(hint, "__args__", ())
            if args and type(None) in args:
                hint = next(a for a in args if a is not type(None))
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


def agent_accessible(description: str | None = None) -> Callable[..., object]:
    """Mark a function as callable by agents and attach metadata for Raysurfer.

    Usage:
        @agent_accessible("Fetches user data from the GitHub API")
        def fetch_user(username: str) -> dict:
            ...
    """

    def decorator(func: Callable[..., object]) -> Callable[..., object]:
        func._raysurfer_accessible = True  # type: ignore[attr-defined]
        func._raysurfer_schema = {  # type: ignore[attr-defined]
            "name": func.__name__,
            "description": description or func.__doc__ or "",
            "input_schema": _build_input_schema(func),
            "source": inspect.getsource(func),
        }
        return func

    return decorator


async def publish_function_registry(
    client: object,
    functions: list[Callable[..., object]],
) -> list[str]:
    """Batch-upload @agent_accessible functions as code blocks to Raysurfer.

    Returns list of code_block_ids for the uploaded functions.
    """
    code_block_ids: list[str] = []
    for func in functions:
        if not getattr(func, "_raysurfer_accessible", False):
            continue
        schema = func._raysurfer_schema  # type: ignore[attr-defined]
        # Use the upload_new_code_snip method from the client
        resp = await client.upload_new_code_snip(  # type: ignore[attr-defined]
            task=f"Call {schema['name']}: {schema['description']}",
            file_written=FileWritten(
                path=f"{schema['name']}.py",
                content=schema["source"],
            ),
            succeeded=True,
            use_raysurfer_ai_voting=False,
        )
        if resp.snippet_name:
            code_block_ids.append(resp.snippet_name)
    return code_block_ids


def set_tracking_client(func: Callable[..., object], client: object) -> None:
    """Attach a Raysurfer client to a decorated function for usage tracking."""
    func._raysurfer_client = client  # type: ignore[attr-defined]
