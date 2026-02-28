"""Tests for raw log search client APIs."""

from __future__ import annotations

import json

import pytest

from raysurfer import AsyncRaySurfer, RaySurfer


def _mock_log_search_response(httpx_mock) -> None:
    httpx_mock.add_response(
        json={
            "matches": [
                {
                    "snippet_id": "cb_log_1",
                    "name": "invoice_parser",
                    "filename": "invoice.py",
                    "language": "python",
                    "created_at": "2026-02-28T10:00:00Z",
                    "triggering_query": "parse invoice pdf",
                    "score": 12.4,
                    "preview": "timeout while parsing page 3",
                    "raw_markdown_url": "https://api.raysurfer.com/raw/cb_log_1.md",
                    "log_url": "https://example.com/run/1",
                }
            ],
            "total_found": 1,
            "has_more": False,
        },
        status_code=200,
    )


def test_sync_search_logs_posts_expected_payload(httpx_mock):
    """search_logs should send filter payload and parse matches."""
    _mock_log_search_response(httpx_mock)
    client = RaySurfer(api_key="test-key", base_url="http://test.local")

    response = client.search_logs(
        "timeout page 3",
        limit=15,
        offset=5,
        code_block_id="cb_log_1",
        language="python",
        days_back=30,
    )

    assert response.total_found == 1
    assert response.matches[0].snippet_id == "cb_log_1"
    assert response.matches[0].raw_markdown_url.endswith("/cb_log_1.md")

    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    payload = json.loads(requests[0].content.decode())
    assert payload == {
        "query": "timeout page 3",
        "limit": 15,
        "offset": 5,
        "code_block_id": "cb_log_1",
        "language": "python",
        "days_back": 30,
    }


@pytest.mark.asyncio
async def test_async_search_logs_posts_expected_payload(httpx_mock):
    """Async search_logs should send filter payload and parse matches."""
    _mock_log_search_response(httpx_mock)

    async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
        response = await client.search_logs("timeout page 3")

    assert response.total_found == 1
    assert response.has_more is False
    assert response.matches[0].name == "invoice_parser"

    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    payload = json.loads(requests[0].content.decode())
    assert payload["query"] == "timeout page 3"
    assert payload["limit"] == 20
    assert payload["offset"] == 0
