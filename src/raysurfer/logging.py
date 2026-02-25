"""Per-function telemetry via raysurfer_logging() — agents call this inside cached functions."""

import atexit
import inspect
import json
import sys
import threading
from dataclasses import dataclass, field


@dataclass
class _FunctionTelemetry:
    """Accumulated telemetry for a single function."""

    call_count: int = 0
    total_value_size: int = 0
    empty_count: int = 0
    value_types: dict[str, int] = field(default_factory=dict)


_CAP = 1000
_lock = threading.Lock()
_telemetry: dict[str, _FunctionTelemetry] = {}


def raysurfer_logging(value: object) -> None:
    """Log a value from inside a cached function for per-function telemetry.

    Uses stack inspection to identify the caller — no decorator needed.
    Accumulates metrics (type, size, emptiness) per function in memory,
    flushed automatically on process exit via atexit.
    """
    frame = inspect.currentframe()
    caller = frame.f_back if frame else None
    if caller:
        func_name = caller.f_code.co_name
        if func_name == "<module>":
            func_name = "__module__"
    else:
        func_name = "__unknown__"

    # Compute metrics for this value
    value_type = type(value).__name__
    try:
        value_size = len(value) if hasattr(value, "__len__") else len(str(value))
    except (TypeError, OverflowError):
        value_size = 0
    is_empty = _is_empty(value)

    with _lock:
        entry = _telemetry.get(func_name)
        if entry is None:
            entry = _FunctionTelemetry()
            _telemetry[func_name] = entry

        entry.call_count += 1
        if entry.call_count <= _CAP:
            entry.total_value_size += value_size
            if is_empty:
                entry.empty_count += 1
            entry.value_types[value_type] = entry.value_types.get(value_type, 0) + 1


def _is_empty(value: object) -> bool:
    """Check if a value is considered 'empty' for telemetry purposes."""
    if value is None:
        return True
    if isinstance(value, (str, bytes, list, dict, tuple, set, frozenset)):
        return len(value) == 0
    return False


def get_telemetry_json() -> str:
    """Return accumulated telemetry as a JSON string for in-process SDK reads."""
    with _lock:
        return json.dumps(_build_telemetry_payload())


def reset_telemetry() -> None:
    """Clear all accumulated telemetry (for testing)."""
    with _lock:
        _telemetry.clear()


def _build_telemetry_payload() -> dict[str, dict[str, object]]:
    """Build the telemetry payload dict."""
    functions: dict[str, dict[str, object]] = {}
    for func_name, entry in _telemetry.items():
        avg_size = entry.total_value_size / min(entry.call_count, _CAP) if entry.call_count > 0 else 0
        empty_rate = entry.empty_count / min(entry.call_count, _CAP) if entry.call_count > 0 else 0
        functions[func_name] = {
            "call_count": entry.call_count,
            "avg_value_size": round(avg_size, 2),
            "empty_rate": round(empty_rate, 4),
            "value_types": entry.value_types,
        }
    return {
        "raysurfer_telemetry": {
            "version": 1,
            "functions": functions,
        }
    }


def _flush_telemetry() -> None:
    """Print delimited telemetry JSON to stdout (atexit handler)."""
    with _lock:
        if not _telemetry:
            return
        payload = _build_telemetry_payload()
    try:
        sys.stdout.write("\n--- RAYSURFER_TELEMETRY_START ---\n")
        sys.stdout.write(json.dumps(payload))
        sys.stdout.write("\n--- RAYSURFER_TELEMETRY_END ---\n")
        sys.stdout.flush()
    except (OSError, ValueError):
        pass  # stdout may be closed at exit


atexit.register(_flush_telemetry)
