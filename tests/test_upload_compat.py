"""Compatibility tests for upload_new_code_snips legacy kwargs."""

import json

import pytest

from raysurfer import AsyncRaySurfer, RaySurfer
from raysurfer.types import FileWritten


def _mock_upload_response(httpx_mock) -> None:
    httpx_mock.add_response(
        json={"success": True, "code_blocks_stored": 1, "message": "Stored"},
        status_code=200,
    )


def test_sync_upload_new_code_snips_accepts_files_written_auto_vote(httpx_mock):
    """upload_new_code_snips should accept files_written and auto_vote aliases."""
    _mock_upload_response(httpx_mock)
    client = RaySurfer(api_key="test-key", base_url="http://test.local")

    result = client.upload_new_code_snips(
        task="compat upload",
        files_written=[FileWritten(path="single.py", content="print('ok')")],
        succeeded=True,
        auto_vote=False,
    )

    assert result.success is True
    assert result.code_blocks_stored == 1

    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    payload = json.loads(requests[0].content.decode())
    assert payload["file_written"]["path"] == "single.py"
    assert payload["use_raysurfer_ai_voting"] is False


@pytest.mark.asyncio
async def test_async_upload_new_code_snips_accepts_files_written_auto_vote(httpx_mock):
    """Async upload_new_code_snips should accept files_written and auto_vote aliases."""
    _mock_upload_response(httpx_mock)

    async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
        result = await client.upload_new_code_snips(
            task="async compat upload",
            files_written=[FileWritten(path="single_async.py", content="print('ok')")],
            succeeded=True,
            auto_vote=False,
        )

    assert result.success is True
    assert result.code_blocks_stored == 1

    requests = httpx_mock.get_requests()
    assert len(requests) == 1
    payload = json.loads(requests[0].content.decode())
    assert payload["file_written"]["path"] == "single_async.py"
    assert payload["use_raysurfer_ai_voting"] is False


def test_sync_upload_new_code_snips_multiple_files_aggregates(httpx_mock):
    """Multiple files in files_written should upload sequentially and aggregate."""
    _mock_upload_response(httpx_mock)
    _mock_upload_response(httpx_mock)
    client = RaySurfer(api_key="test-key", base_url="http://test.local")

    result = client.upload_new_code_snips(
        task="multi compat upload",
        files_written=[
            FileWritten(path="one.py", content="print('one')"),
            FileWritten(path="two.py", content="print('two')"),
        ],
        succeeded=True,
    )

    assert result.success is True
    assert result.code_blocks_stored == 2
    assert result.message == "Uploaded 2 files via compatibility path."

    requests = httpx_mock.get_requests()
    assert len(requests) == 2
    payload_1 = json.loads(requests[0].content.decode())
    payload_2 = json.loads(requests[1].content.decode())
    assert payload_1["file_written"]["path"] == "one.py"
    assert payload_2["file_written"]["path"] == "two.py"


def test_sync_upload_new_code_snips_rejects_ambiguous_file_inputs():
    """Passing both file_written and files_written should raise ValueError."""
    client = RaySurfer(api_key="test-key", base_url="http://test.local")

    with pytest.raises(ValueError, match="either file_written or files_written"):
        client.upload_new_code_snips(
            task="ambiguous upload",
            file_written=FileWritten(path="one.py", content="print('one')"),
            files_written=[FileWritten(path="two.py", content="print('two')")],
            succeeded=True,
        )


def test_sync_upload_new_code_snips_requires_file_input():
    """Missing file_written/files_written should raise ValueError."""
    client = RaySurfer(api_key="test-key", base_url="http://test.local")

    with pytest.raises(ValueError, match="provide file_written or files_written"):
        client.upload_new_code_snips(task="missing file", succeeded=True)


def test_sync_search_match_exposes_legacy_score_aliases(httpx_mock):
    """SearchMatch should expose combined/vector/verdict score compatibility fields."""
    httpx_mock.add_response(
        json={
            "matches": [
                {
                    "code_block": {
                        "id": "cb_compat",
                        "name": "compat_example.py",
                        "description": "Compatibility example",
                        "source": "print('hello')",
                        "entrypoint": "main",
                        "language": "python",
                    },
                    "score": 0.81,
                    "thumbs_up": 5,
                    "thumbs_down": 0,
                    "filename": "compat_example.py",
                    "language": "python",
                    "entrypoint": "main",
                }
            ],
            "total_found": 1,
            "cache_hit": True,
        },
        status_code=200,
    )
    client = RaySurfer(api_key="test-key", base_url="http://test.local")

    response = client.search(task="compat query")
    match = response.matches[0]
    assert match.combined_score == 0.81
    assert match.vector_score == 0.81
    assert match.verdict_score == 0.81


@pytest.mark.asyncio
async def test_async_search_match_uses_explicit_vector_verdict_scores(httpx_mock):
    """SearchMatch should preserve explicit vector/verdict scores when provided."""
    httpx_mock.add_response(
        json={
            "matches": [
                {
                    "code_block": {
                        "id": "cb_compat_scores",
                        "name": "compat_scores.py",
                        "description": "Compatibility score example",
                        "source": "print('scores')",
                        "entrypoint": "main",
                        "language": "python",
                    },
                    "score": 0.7,
                    "vector_score": 0.92,
                    "verdict_score": 0.63,
                    "thumbs_up": 3,
                    "thumbs_down": 1,
                    "filename": "compat_scores.py",
                    "language": "python",
                    "entrypoint": "main",
                }
            ],
            "total_found": 1,
            "cache_hit": True,
        },
        status_code=200,
    )

    async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
        response = await client.search(task="compat query async")

    match = response.matches[0]
    assert match.combined_score == 0.7
    assert match.vector_score == 0.92
    assert match.verdict_score == 0.63
