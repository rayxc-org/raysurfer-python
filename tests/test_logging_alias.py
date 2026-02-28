"""Tests for telemetry logging aliases exported from raysurfer."""

from __future__ import annotations

import json

from raysurfer import log
from raysurfer.logging import get_telemetry_json, reset_telemetry


def test_log_alias_tracks_telemetry_without_decorator() -> None:
    """Calling `log(...)` should record telemetry from plain functions."""
    reset_telemetry()

    def plain_function() -> dict[str, str]:
        payload = {"status": "ok"}
        log(payload)
        return payload

    plain_function()

    telemetry = json.loads(get_telemetry_json())
    functions = telemetry["raysurfer_telemetry"]["functions"]
    assert "plain_function" in functions
    assert functions["plain_function"]["call_count"] == 1

    reset_telemetry()
