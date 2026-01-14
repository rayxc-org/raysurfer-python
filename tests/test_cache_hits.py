"""Tests for cache hit detection - similar tasks should return cached results"""

import pytest

from raysurfer import (
    AsyncRaySurfer,
    RaySurfer,
)

# =============================================================================
# Cache Hit Detection Tests
# =============================================================================


class TestCacheHitDetection:
    """Tests for detecting cache hits on similar tasks."""

    @pytest.mark.asyncio
    async def test_similar_task_returns_cached_result(self, httpx_mock):
        """Similar tasks should return matching cached code blocks."""
        # Mock response with a high-scoring cached result
        httpx_mock.add_response(
            json={
                "code_blocks": [
                    {
                        "code_block": {
                            "id": "cb_github_fetcher",
                            "name": "GitHub User Fetcher",
                            "description": "Fetches user data from GitHub API",
                            "source": "import requests\n\ndef fetch_github_user(username):\n    response = requests.get(f'https://api.github.com/users/{username}')\n    return response.json()",
                            "entrypoint": "fetch_github_user",
                            "language": "python",
                        },
                        "score": 0.95,  # High semantic similarity
                        "verdict_score": 0.9,  # High quality score
                        "thumbs_up": 25,
                        "thumbs_down": 2,
                        "recent_executions": [],
                    }
                ],
                "total_found": 1,
            },
            status_code=200,
        )

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            # Original task
            result = await client.get_code_snips(task="Fetch user data from GitHub")

            assert result.total_found == 1
            assert len(result.code_blocks) == 1

            # Verify it's a cache hit (high score)
            match = result.code_blocks[0]
            assert match.score >= 0.9  # High similarity = cache hit
            assert match.verdict_score >= 0.8  # Proven to work

    @pytest.mark.asyncio
    async def test_semantically_similar_tasks_return_same_cached_code(self, httpx_mock):
        """Semantically similar but differently worded tasks should match same code."""
        similar_tasks = [
            "Fetch user data from GitHub",
            "Get GitHub user information",
            "Retrieve user profile from GitHub API",
            "Download GitHub user details",
        ]

        for task in similar_tasks:
            httpx_mock.reset()
            httpx_mock.add_response(
                json={
                    "code_blocks": [
                        {
                            "code_block": {
                                "id": "cb_github_fetcher",
                                "name": "GitHub User Fetcher",
                                "description": "Fetches user data from GitHub",
                                "source": "def fetch(): pass",
                                "entrypoint": "fetch",
                                "language": "python",
                            },
                            "score": 0.92,
                            "verdict_score": 0.88,
                            "thumbs_up": 20,
                            "thumbs_down": 1,
                        }
                    ],
                    "total_found": 1,
                },
                status_code=200,
            )

            async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
                result = await client.get_code_snips(task=task)

                assert result.total_found >= 1
                # Same code block should match all similar tasks
                assert result.code_blocks[0].code_block.id == "cb_github_fetcher"

    @pytest.mark.asyncio
    async def test_cache_miss_returns_low_score(self, httpx_mock):
        """Unrelated tasks should return no or low-scoring matches (cache miss)."""
        httpx_mock.add_response(
            json={
                "code_blocks": [],  # No matches
                "total_found": 0,
            },
            status_code=200,
        )

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            result = await client.get_code_snips(task="Send email notification")

            assert result.total_found == 0
            assert len(result.code_blocks) == 0

    @pytest.mark.asyncio
    async def test_partial_cache_hit_with_low_verdict_score(self, httpx_mock):
        """Matching code with low verdict score indicates unproven cache."""
        httpx_mock.add_response(
            json={
                "code_blocks": [
                    {
                        "code_block": {
                            "id": "cb_unproven",
                            "name": "Unproven Fetcher",
                            "description": "A new fetcher with no track record",
                            "source": "def fetch(): pass",
                            "entrypoint": "fetch",
                            "language": "python",
                        },
                        "score": 0.85,  # Good semantic match
                        "verdict_score": 0.1,  # Low quality - not proven
                        "thumbs_up": 1,
                        "thumbs_down": 5,
                    }
                ],
                "total_found": 1,
            },
            status_code=200,
        )

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            result = await client.get_code_snips(task="Fetch some data")

            assert result.total_found == 1
            match = result.code_blocks[0]
            # It's a match but with low confidence
            assert match.verdict_score < 0.5
            assert match.thumbs_down > match.thumbs_up


# =============================================================================
# Retrieve Best Cache Hit Tests
# =============================================================================


class TestRetrieveBestCacheHit:
    """Tests for retrieve_best endpoint cache hit detection."""

    @pytest.mark.asyncio
    async def test_retrieve_best_returns_high_confidence_hit(self, httpx_mock):
        """retrieve_best should return high confidence for good cache hits."""
        httpx_mock.add_response(
            json={
                "best_match": {
                    "code_block": {
                        "id": "cb_best",
                        "name": "Proven Data Fetcher",
                        "description": "Well-tested data fetcher",
                        "source": "def fetch_data(): pass",
                        "entrypoint": "fetch_data",
                        "language": "python",
                    },
                    "combined_score": 0.95,
                    "vector_score": 0.92,
                    "verdict_score": 0.98,
                    "error_resilience": 0.95,
                    "thumbs_up": 50,
                    "thumbs_down": 2,
                },
                "alternative_candidates": [],
                "retrieval_confidence": "high",
            },
            status_code=200,
        )

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            result = await client.retrieve_best(task="Fetch data from API")

            assert result.best_match is not None
            assert result.retrieval_confidence == "high"
            assert result.best_match.combined_score >= 0.9
            assert result.best_match.verdict_score >= 0.9

    @pytest.mark.asyncio
    async def test_retrieve_best_returns_low_confidence_for_miss(self, httpx_mock):
        """retrieve_best should return low confidence for cache misses."""
        httpx_mock.add_response(
            json={
                "best_match": None,
                "alternative_candidates": [],
                "retrieval_confidence": "low",
            },
            status_code=200,
        )

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            result = await client.retrieve_best(task="Very unique task never seen before")

            assert result.best_match is None
            assert result.retrieval_confidence == "low"

    @pytest.mark.asyncio
    async def test_retrieve_best_with_alternatives(self, httpx_mock):
        """retrieve_best should return alternatives when best match is uncertain."""
        httpx_mock.add_response(
            json={
                "best_match": {
                    "code_block": {
                        "id": "cb_primary",
                        "name": "Primary Fetcher",
                        "description": "Primary option",
                        "source": "def primary(): pass",
                        "entrypoint": "primary",
                        "language": "python",
                    },
                    "combined_score": 0.75,
                    "vector_score": 0.8,
                    "verdict_score": 0.7,
                    "error_resilience": 0.8,
                    "thumbs_up": 10,
                    "thumbs_down": 3,
                },
                "alternative_candidates": [
                    {
                        "code_block_id": "cb_alt1",
                        "name": "Alternative 1",
                        "combined_score": 0.72,
                        "reason": "Similar pattern, different approach",
                    },
                    {
                        "code_block_id": "cb_alt2",
                        "name": "Alternative 2",
                        "combined_score": 0.68,
                        "reason": "Newer implementation",
                    },
                ],
                "retrieval_confidence": "medium",
            },
            status_code=200,
        )

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            result = await client.retrieve_best(task="Process some data")

            assert result.best_match is not None
            assert result.retrieval_confidence == "medium"
            assert len(result.alternative_candidates) == 2


# =============================================================================
# Min Verdict Score Filter Tests
# =============================================================================


class TestMinVerdictScoreFilter:
    """Tests for filtering by minimum verdict score."""

    @pytest.mark.asyncio
    async def test_filter_by_min_verdict_score(self, httpx_mock):
        """Should filter results by minimum verdict score."""
        httpx_mock.add_response(
            json={
                "code_blocks": [
                    {
                        "code_block": {
                            "id": "cb_high_quality",
                            "name": "High Quality Fetcher",
                            "description": "Well-tested",
                            "source": "def fetch(): pass",
                            "entrypoint": "fetch",
                            "language": "python",
                        },
                        "score": 0.9,
                        "verdict_score": 0.85,
                        "thumbs_up": 30,
                        "thumbs_down": 2,
                    }
                ],
                "total_found": 1,
            },
            status_code=200,
        )

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            # Request with high min_verdict_score
            result = await client.get_code_snips(task="Fetch data", min_verdict_score=0.8)

            assert result.total_found == 1
            # Only high-quality results returned
            assert result.code_blocks[0].verdict_score >= 0.8

    @pytest.mark.asyncio
    async def test_request_includes_min_verdict_score(self, httpx_mock):
        """Request should include min_verdict_score parameter."""
        httpx_mock.add_response(
            json={"code_blocks": [], "total_found": 0},
            status_code=200,
        )

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            await client.get_code_snips(task="test", min_verdict_score=0.6)

        request = httpx_mock.get_request()
        import json

        body = json.loads(request.content)
        assert body["min_verdict_score"] == 0.6


# =============================================================================
# Multiple Results Ranking Tests
# =============================================================================


class TestMultipleResultsRanking:
    """Tests for ranking multiple cached results."""

    @pytest.mark.asyncio
    async def test_results_sorted_by_combined_score(self, httpx_mock):
        """Results should be sorted by combined score (highest first)."""
        httpx_mock.add_response(
            json={
                "code_blocks": [
                    {
                        "code_block": {
                            "id": "cb_best",
                            "name": "Best Match",
                            "description": "Highest scoring",
                            "source": "def best(): pass",
                            "entrypoint": "best",
                            "language": "python",
                        },
                        "score": 0.95,
                        "verdict_score": 0.9,
                        "thumbs_up": 50,
                        "thumbs_down": 1,
                    },
                    {
                        "code_block": {
                            "id": "cb_second",
                            "name": "Second Best",
                            "description": "Second highest",
                            "source": "def second(): pass",
                            "entrypoint": "second",
                            "language": "python",
                        },
                        "score": 0.85,
                        "verdict_score": 0.8,
                        "thumbs_up": 30,
                        "thumbs_down": 5,
                    },
                    {
                        "code_block": {
                            "id": "cb_third",
                            "name": "Third Match",
                            "description": "Third option",
                            "source": "def third(): pass",
                            "entrypoint": "third",
                            "language": "python",
                        },
                        "score": 0.7,
                        "verdict_score": 0.6,
                        "thumbs_up": 10,
                        "thumbs_down": 8,
                    },
                ],
                "total_found": 3,
            },
            status_code=200,
        )

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            result = await client.get_code_snips(task="Fetch data")

            assert result.total_found == 3
            # Verify order - best first
            assert result.code_blocks[0].code_block.id == "cb_best"
            assert result.code_blocks[1].code_block.id == "cb_second"
            assert result.code_blocks[2].code_block.id == "cb_third"

    @pytest.mark.asyncio
    async def test_top_k_limits_results(self, httpx_mock):
        """top_k should limit number of returned results."""
        httpx_mock.add_response(
            json={
                "code_blocks": [
                    {
                        "code_block": {
                            "id": f"cb_{i}",
                            "name": f"Match {i}",
                            "description": f"Match number {i}",
                            "source": f"def match_{i}(): pass",
                            "entrypoint": f"match_{i}",
                            "language": "python",
                        },
                        "score": 0.9 - (i * 0.05),
                        "verdict_score": 0.8,
                        "thumbs_up": 10,
                        "thumbs_down": 1,
                    }
                    for i in range(3)  # Return 3 even though more exist
                ],
                "total_found": 10,  # 10 total exist
            },
            status_code=200,
        )

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            result = await client.get_code_snips(task="test", top_k=3)

            assert result.total_found == 10  # Backend reports total
            assert len(result.code_blocks) == 3  # But only 3 returned


# =============================================================================
# Cache Hit Identification Helpers
# =============================================================================


class TestCacheHitIdentification:
    """Tests for helper logic to identify cache hits."""

    def test_high_score_indicates_cache_hit(self):
        """High combined score indicates a cache hit."""
        # A score >= 0.8 with good verdict is a cache hit
        is_cache_hit = lambda score, verdict: score >= 0.8 and verdict >= 0.5

        assert is_cache_hit(0.95, 0.9) is True  # Perfect hit
        assert is_cache_hit(0.85, 0.7) is True  # Good hit
        assert is_cache_hit(0.5, 0.9) is False  # Low semantic match
        assert is_cache_hit(0.9, 0.1) is False  # Low quality

    def test_thumbs_ratio_indicates_reliability(self):
        """Thumbs up/down ratio indicates reliability."""

        def reliability_score(thumbs_up, thumbs_down):
            total = thumbs_up + thumbs_down
            if total == 0:
                return 0.5  # Neutral
            return thumbs_up / total

        assert reliability_score(50, 2) > 0.9  # Highly reliable
        assert reliability_score(10, 10) == 0.5  # Neutral
        assert reliability_score(2, 20) < 0.2  # Unreliable


# =============================================================================
# Sync Client Cache Hit Tests
# =============================================================================


class TestSyncCacheHits:
    """Tests for cache hit detection with sync client."""

    def test_sync_similar_task_cache_hit(self, httpx_mock):
        """Sync client should detect cache hits on similar tasks."""
        httpx_mock.add_response(
            json={
                "code_blocks": [
                    {
                        "code_block": {
                            "id": "cb_cached",
                            "name": "Cached Code",
                            "description": "Previously cached code",
                            "source": "def cached(): pass",
                            "entrypoint": "cached",
                            "language": "python",
                        },
                        "score": 0.92,
                        "verdict_score": 0.88,
                        "thumbs_up": 30,
                        "thumbs_down": 2,
                    }
                ],
                "total_found": 1,
            },
            status_code=200,
        )

        with RaySurfer(api_key="test-key", base_url="http://test.local") as client:
            result = client.get_code_snips(task="Get user data from API")

            assert result.total_found == 1
            assert result.code_blocks[0].score >= 0.9
            assert result.code_blocks[0].verdict_score >= 0.8

    def test_sync_cache_miss(self, httpx_mock):
        """Sync client should detect cache misses."""
        httpx_mock.add_response(
            json={"code_blocks": [], "total_found": 0},
            status_code=200,
        )

        with RaySurfer(api_key="test-key", base_url="http://test.local") as client:
            result = client.get_code_snips(task="Completely novel task")

            assert result.total_found == 0
            assert len(result.code_blocks) == 0
