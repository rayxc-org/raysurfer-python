"""raysurfer.yaml loader for agent-accessible function discovery."""

from __future__ import annotations

import ast
import fnmatch
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from raysurfer.accessible import agent_accessible


@dataclass(slots=True)
class AgentAccessRules:
    """Access rules loaded from raysurfer.yaml."""

    read: list[str] = field(default_factory=list)
    call: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RaysurferConfig:
    """Top-level raysurfer configuration model."""

    agent_access: AgentAccessRules = field(default_factory=AgentAccessRules)


def _coerce_string_list(value: object) -> list[str]:
    """Normalize a config value into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _parse_minimal_yaml(text: str) -> dict[str, object]:
    """Parse the subset of YAML used by raysurfer.yaml without third-party deps."""
    result: dict[str, object] = {}
    current_section: str | None = None
    current_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        if indent == 0 and stripped.endswith(":"):
            current_section = stripped[:-1]
            if current_section not in result:
                result[current_section] = {}
            current_key = None
            continue

        if current_section is None:
            continue

        if indent <= 2 and ":" in stripped:
            key, raw_value = stripped.split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            section = result.get(current_section)
            if not isinstance(section, dict):
                section = {}
                result[current_section] = section

            if raw_value:
                try:
                    parsed_value = ast.literal_eval(raw_value)
                except Exception:
                    parsed_value = raw_value
                section[key] = parsed_value
                current_key = None
            else:
                section[key] = []
                current_key = key
            continue

        if current_key and stripped.startswith("- "):
            section = result.get(current_section)
            if not isinstance(section, dict):
                continue
            existing = section.get(current_key)
            if not isinstance(existing, list):
                existing = []
                section[current_key] = existing
            item = stripped[2:].strip().strip('"').strip("'")
            existing.append(item)

    return result


def _load_config_dict(path: Path) -> dict[str, object]:
    """Load raysurfer.yaml into a dictionary."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        parsed = yaml.safe_load(text)
        if isinstance(parsed, dict):
            return {str(key): value for key, value in parsed.items()}
    except Exception:
        pass
    return _parse_minimal_yaml(text)


def _load_rules(path: Path) -> RaysurferConfig:
    """Parse raysurfer.yaml into a strongly-typed config object."""
    raw = _load_config_dict(path)
    agent_access_raw = raw.get("agent_access", {})
    if not isinstance(agent_access_raw, dict):
        agent_access_raw = {}

    rules = AgentAccessRules(
        read=_coerce_string_list(agent_access_raw.get("read")),
        call=_coerce_string_list(agent_access_raw.get("call")),
        deny=_coerce_string_list(agent_access_raw.get("deny")),
    )
    return RaysurferConfig(agent_access=rules)


def _normalize_path(path: Path) -> str:
    """Normalize filesystem paths to forward-slash form for glob matching."""
    return path.as_posix()


def _relative_source_path(source_path: Path, project_root: Path) -> str:
    """Build a project-relative source path string for selector matching."""
    resolved_source = source_path.resolve()
    resolved_root = project_root.resolve()
    try:
        return _normalize_path(resolved_source.relative_to(resolved_root))
    except ValueError:
        return _normalize_path(resolved_source)


def _matches_any(value: str, patterns: list[str]) -> bool:
    """Return True if value matches any glob pattern."""
    return any(fnmatch.fnmatch(value, pattern) for pattern in patterns)


def load_config(path: str, modules: list[ModuleType]) -> list[Callable[..., object]]:
    """Load raysurfer.yaml, discover matching functions, and mark them agent-accessible."""
    config_path = Path(path).expanduser().resolve()
    config = _load_rules(config_path)
    call_patterns = config.agent_access.call
    deny_patterns = config.agent_access.deny
    project_root = config_path.parent

    selected: list[Callable[..., object]] = []
    seen_functions: set[int] = set()

    for module in modules:
        for _, func in inspect.getmembers(module, inspect.isfunction):
            func_id = id(func)
            if func_id in seen_functions:
                continue

            source_file = inspect.getsourcefile(func) or getattr(module, "__file__", None)
            if not source_file:
                continue

            rel_path = _relative_source_path(Path(source_file), project_root)
            selector = f"{rel_path}:{func.__name__}"

            if call_patterns and not _matches_any(selector, call_patterns):
                continue

            if _matches_any(rel_path, deny_patterns) or _matches_any(selector, deny_patterns):
                continue

            if not bool(getattr(func, "_raysurfer_accessible", False)):
                func = agent_accessible(func.__doc__)(func)

            selected.append(func)
            seen_functions.add(func_id)

    return selected
