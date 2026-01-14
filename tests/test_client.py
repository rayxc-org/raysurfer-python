"""Comprehensive tests for RaySurfer client - auth, errors, timeouts, retries"""

import httpx
import pytest

from raysurfer import (
    APIError,
    AsyncRaySurfer,
    AuthenticationError,
    RaySurfer,
)

# =============================================================================
# Basic Initialization Tests
# =============================================================================


class TestClientInitialization:
    """Tests for client initialization."""

    def test_sync_client_init(self):
        """Sync client should initialize with api_key and base_url."""
        client = RaySurfer(api_key="test-key", base_url="http://localhost:8000")
        assert client.api_key == "test-key"
        assert client.base_url == "http://localhost:8000"

    def test_sync_client_default_base_url(self):
        """Sync client should have default base URL."""
        client = RaySurfer(api_key="test-key")
        assert client.base_url == "https://api.raysurfer.com"

    def test_sync_client_strips_trailing_slash(self):
        """Sync client should strip trailing slashes from base_url."""
        client = RaySurfer(api_key="test-key", base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_async_client_init(self):
        """Async client should initialize with api_key and base_url."""
        client = AsyncRaySurfer(api_key="test-key", base_url="http://localhost:8000")
        assert client.api_key == "test-key"
        assert client.base_url == "http://localhost:8000"

    def test_async_client_default_base_url(self):
        """Async client should have default base URL."""
        client = AsyncRaySurfer(api_key="test-key")
        assert client.base_url == "https://api.raysurfer.com"

    def test_async_client_strips_trailing_slash(self):
        """Async client should strip trailing slashes from base_url."""
        client = AsyncRaySurfer(api_key="test-key", base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_sync_client_custom_timeout(self):
        """Sync client should accept custom timeout."""
        client = RaySurfer(api_key="test-key", timeout=60.0)
        assert client.timeout == 60.0

    def test_async_client_custom_timeout(self):
        """Async client should accept custom timeout."""
        client = AsyncRaySurfer(api_key="test-key", timeout=60.0)
        assert client.timeout == 60.0


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for context manager behavior."""

    def test_sync_context_manager(self):
        """Sync client should work as context manager."""
        with RaySurfer(api_key="test") as client:
            assert client.api_key == "test"

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Async client should work as context manager."""
        async with AsyncRaySurfer(api_key="test") as client:
            assert client.api_key == "test"

    def test_sync_client_close(self):
        """Sync client should close HTTP client on close()."""
        client = RaySurfer(api_key="test")
        # Force client creation
        _ = client._get_client()
        assert client._client is not None
        client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_async_client_close(self):
        """Async client should close HTTP client on close()."""
        client = AsyncRaySurfer(api_key="test")
        # Force client creation
        _ = await client._get_client()
        assert client._client is not None
        await client.close()
        assert client._client is None


# =============================================================================
# Authentication Error Tests
# =============================================================================


class TestAuthenticationErrors:
    """Tests for authentication error handling."""

    @pytest.mark.asyncio
    async def test_async_auth_error_401(self, httpx_mock):
        """Async client should raise AuthenticationError on 401 response."""
        httpx_mock.add_response(status_code=401, text="Unauthorized")

        async with AsyncRaySurfer(api_key="invalid-key", base_url="http://test.local") as client:
            with pytest.raises(AuthenticationError) as exc_info:
                await client.get_code_snips(task="test task")

            assert str(exc_info.value) == "Invalid API key"

    def test_sync_auth_error_401(self, httpx_mock):
        """Sync client should raise AuthenticationError on 401 response."""
        httpx_mock.add_response(status_code=401, text="Unauthorized")

        with RaySurfer(api_key="invalid-key", base_url="http://test.local") as client:
            with pytest.raises(AuthenticationError) as exc_info:
                client.get_code_snips(task="test task")

            assert str(exc_info.value) == "Invalid API key"

    @pytest.mark.asyncio
    async def test_async_missing_api_key(self, httpx_mock):
        """Async client should still work without API key (if endpoint allows)."""
        httpx_mock.add_response(json={"code_blocks": [], "total_found": 0}, status_code=200)

        async with AsyncRaySurfer(base_url="http://test.local") as client:
            result = await client.get_code_snips(task="test task")
            assert result.total_found == 0


# =============================================================================
# API Error Tests
# =============================================================================


class TestAPIErrors:
    """Tests for API error handling."""

    @pytest.mark.asyncio
    async def test_async_api_error_400(self, httpx_mock):
        """Async client should raise APIError on 400 response."""
        httpx_mock.add_response(status_code=400, text="Bad Request: missing field 'task'")

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            with pytest.raises(APIError) as exc_info:
                await client.get_code_snips(task="")

            assert exc_info.value.status_code == 400
            assert "Bad Request" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_api_error_500(self, httpx_mock):
        """Async client should raise APIError on 500 response."""
        httpx_mock.add_response(status_code=500, text="Internal Server Error")

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            with pytest.raises(APIError) as exc_info:
                await client.get_code_snips(task="test task")

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_async_api_error_404(self, httpx_mock):
        """Async client should raise APIError on 404 response."""
        httpx_mock.add_response(status_code=404, text="Not Found")

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            with pytest.raises(APIError) as exc_info:
                await client.get_code_snips(task="test task")

            assert exc_info.value.status_code == 404

    def test_sync_api_error_400(self, httpx_mock):
        """Sync client should raise APIError on 400 response."""
        httpx_mock.add_response(status_code=400, text="Bad Request")

        with RaySurfer(api_key="test-key", base_url="http://test.local") as client:
            with pytest.raises(APIError) as exc_info:
                client.get_code_snips(task="")

            assert exc_info.value.status_code == 400

    def test_sync_api_error_500(self, httpx_mock):
        """Sync client should raise APIError on 500 response."""
        httpx_mock.add_response(status_code=500, text="Internal Server Error")

        with RaySurfer(api_key="test-key", base_url="http://test.local") as client:
            with pytest.raises(APIError) as exc_info:
                client.get_code_snips(task="test task")

            assert exc_info.value.status_code == 500


# =============================================================================
# Timeout Tests
# =============================================================================


class TestTimeouts:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_async_timeout_configuration(self):
        """Async client should configure timeout on HTTP client."""
        client = AsyncRaySurfer(api_key="test-key", timeout=5.0)
        http_client = await client._get_client()
        assert http_client.timeout.read == 5.0
        await client.close()

    def test_sync_timeout_configuration(self):
        """Sync client should configure timeout on HTTP client."""
        client = RaySurfer(api_key="test-key", timeout=5.0)
        http_client = client._get_client()
        assert http_client.timeout.read == 5.0
        client.close()

    @pytest.mark.asyncio
    async def test_async_timeout_error(self, httpx_mock):
        """Async client should propagate timeout errors."""

        def raise_timeout(request):
            raise httpx.TimeoutException("Request timed out", request=request)

        httpx_mock.add_callback(raise_timeout)

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local", timeout=0.001) as client:
            with pytest.raises(httpx.TimeoutException):
                await client.get_code_snips(task="test task")

    def test_sync_timeout_error(self, httpx_mock):
        """Sync client should propagate timeout errors."""

        def raise_timeout(request):
            raise httpx.TimeoutException("Request timed out", request=request)

        httpx_mock.add_callback(raise_timeout)

        with RaySurfer(api_key="test-key", base_url="http://test.local", timeout=0.001) as client:
            with pytest.raises(httpx.TimeoutException):
                client.get_code_snips(task="test task")


# =============================================================================
# Connection Error Tests
# =============================================================================


class TestConnectionErrors:
    """Tests for connection error handling."""

    @pytest.mark.asyncio
    async def test_async_connection_error(self, httpx_mock):
        """Async client should propagate connection errors."""

        def raise_connection_error(request):
            raise httpx.ConnectError("Connection refused", request=request)

        httpx_mock.add_callback(raise_connection_error)

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            with pytest.raises(httpx.ConnectError):
                await client.get_code_snips(task="test task")

    def test_sync_connection_error(self, httpx_mock):
        """Sync client should propagate connection errors."""

        def raise_connection_error(request):
            raise httpx.ConnectError("Connection refused", request=request)

        httpx_mock.add_callback(raise_connection_error)

        with RaySurfer(api_key="test-key", base_url="http://test.local") as client:
            with pytest.raises(httpx.ConnectError):
                client.get_code_snips(task="test task")


# =============================================================================
# Header Tests
# =============================================================================


class TestRequestHeaders:
    """Tests for request header handling."""

    @pytest.mark.asyncio
    async def test_async_auth_header(self, httpx_mock):
        """Async client should include Authorization header with Bearer token."""
        httpx_mock.add_response(json={"code_blocks": [], "total_found": 0}, status_code=200)

        async with AsyncRaySurfer(api_key="my-secret-key", base_url="http://test.local") as client:
            await client.get_code_snips(task="test task")

        request = httpx_mock.get_request()
        assert request.headers["Authorization"] == "Bearer my-secret-key"

    @pytest.mark.asyncio
    async def test_async_content_type_header(self, httpx_mock):
        """Async client should include Content-Type header."""
        httpx_mock.add_response(json={"code_blocks": [], "total_found": 0}, status_code=200)

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            await client.get_code_snips(task="test task")

        request = httpx_mock.get_request()
        assert request.headers["Content-Type"] == "application/json"

    def test_sync_auth_header(self, httpx_mock):
        """Sync client should include Authorization header with Bearer token."""
        httpx_mock.add_response(json={"code_blocks": [], "total_found": 0}, status_code=200)

        with RaySurfer(api_key="my-secret-key", base_url="http://test.local") as client:
            client.get_code_snips(task="test task")

        request = httpx_mock.get_request()
        assert request.headers["Authorization"] == "Bearer my-secret-key"


# =============================================================================
# Store Code Block Tests
# =============================================================================


class TestStoreCodeBlock:
    """Tests for store_code_block functionality."""

    @pytest.mark.asyncio
    async def test_async_store_code_block_success(self, httpx_mock):
        """Async client should successfully store a code block."""
        httpx_mock.add_response(
            json={
                "success": True,
                "code_block_id": "cb_123",
                "embedding_id": "emb_456",
                "message": "Code block stored successfully",
            },
            status_code=200,
        )

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            result = await client.store_code_block(
                name="GitHub User Fetcher",
                source="def fetch_user(username): pass",
                entrypoint="fetch_user",
                language="python",
                description="Fetches user data from GitHub",
            )

            assert result.success is True
            assert result.code_block_id == "cb_123"
            assert result.embedding_id == "emb_456"

    def test_sync_store_code_block_success(self, httpx_mock):
        """Sync client should successfully store a code block."""
        httpx_mock.add_response(
            json={
                "success": True,
                "code_block_id": "cb_123",
                "embedding_id": "emb_456",
                "message": "Code block stored successfully",
            },
            status_code=200,
        )

        with RaySurfer(api_key="test-key", base_url="http://test.local") as client:
            result = client.store_code_block(
                name="GitHub User Fetcher",
                source="def fetch_user(username): pass",
                entrypoint="fetch_user",
                language="python",
            )

            assert result.success is True
            assert result.code_block_id == "cb_123"

    @pytest.mark.asyncio
    async def test_async_store_code_block_with_all_fields(self, httpx_mock):
        """Async client should send all optional fields when provided."""
        httpx_mock.add_response(
            json={
                "success": True,
                "code_block_id": "cb_123",
                "embedding_id": "emb_456",
                "message": "Stored",
            },
            status_code=200,
        )

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            await client.store_code_block(
                name="Test Block",
                source="def test(): pass",
                entrypoint="test",
                language="python",
                description="A test function",
                input_schema={"name": "str"},
                output_schema={"result": "bool"},
                language_version="3.11",
                dependencies=["requests", "pandas"],
                tags=["api", "fetch"],
                capabilities=["http", "json"],
                example_queries=["Fetch data from API"],
            )

        request = httpx_mock.get_request()
        body = request.content.decode()
        assert "Test Block" in body
        assert "requests" in body
        assert "api" in body


# =============================================================================
# Retrieve Tests
# =============================================================================


class TestRetrieve:
    """Tests for retrieve functionality."""

    @pytest.mark.asyncio
    async def test_async_retrieve_code_blocks(self, httpx_mock):
        """Async client should retrieve code blocks with proper parsing."""
        httpx_mock.add_response(
            json={
                "code_blocks": [
                    {
                        "code_block": {
                            "id": "cb_123",
                            "name": "GitHub Fetcher",
                            "description": "Fetches GitHub data",
                            "source": "def fetch(): pass",
                            "entrypoint": "fetch",
                            "language": "python",
                        },
                        "score": 0.95,
                        "verdict_score": 0.85,
                        "thumbs_up": 10,
                        "thumbs_down": 1,
                        "recent_executions": [],
                    }
                ],
                "total_found": 1,
            },
            status_code=200,
        )

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            result = await client.get_code_snips(task="Fetch GitHub data")

            assert result.total_found == 1
            assert len(result.code_blocks) == 1
            assert result.code_blocks[0].code_block.name == "GitHub Fetcher"
            assert result.code_blocks[0].score == 0.95
            assert result.code_blocks[0].verdict_score == 0.85

    def test_sync_retrieve_code_blocks(self, httpx_mock):
        """Sync client should retrieve code blocks with proper parsing."""
        httpx_mock.add_response(
            json={
                "code_blocks": [
                    {
                        "code_block": {
                            "id": "cb_123",
                            "name": "GitHub Fetcher",
                            "description": "Fetches GitHub data",
                            "source": "def fetch(): pass",
                            "entrypoint": "fetch",
                            "language": "python",
                        },
                        "score": 0.95,
                        "verdict_score": 0.85,
                        "thumbs_up": 10,
                        "thumbs_down": 1,
                    }
                ],
                "total_found": 1,
            },
            status_code=200,
        )

        with RaySurfer(api_key="test-key", base_url="http://test.local") as client:
            result = client.get_code_snips(task="Fetch GitHub data")

            assert result.total_found == 1
            assert result.code_blocks[0].code_block.name == "GitHub Fetcher"

    @pytest.mark.asyncio
    async def test_async_retrieve_with_filters(self, httpx_mock):
        """Async client should send filter parameters."""
        httpx_mock.add_response(
            json={"code_blocks": [], "total_found": 0},
            status_code=200,
        )

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            await client.get_code_snips(task="test", top_k=5, min_verdict_score=0.5)

        request = httpx_mock.get_request()
        import json

        body = json.loads(request.content)
        assert body["top_k"] == 5
        assert body["min_verdict_score"] == 0.5


# =============================================================================
# Retrieve Best Tests
# =============================================================================


class TestRetrieveBest:
    """Tests for retrieve_best functionality."""

    @pytest.mark.asyncio
    async def test_async_retrieve_best_with_match(self, httpx_mock):
        """Async client should retrieve best match with scoring."""
        httpx_mock.add_response(
            json={
                "best_match": {
                    "code_block": {
                        "id": "cb_best",
                        "name": "Best Fetcher",
                        "description": "The best fetcher",
                        "source": "def best(): pass",
                        "entrypoint": "best",
                        "language": "python",
                    },
                    "combined_score": 0.92,
                    "vector_score": 0.88,
                    "verdict_score": 0.95,
                    "error_resilience": 0.9,
                    "thumbs_up": 20,
                    "thumbs_down": 1,
                },
                "alternative_candidates": [
                    {
                        "code_block_id": "cb_alt",
                        "name": "Alternative",
                        "combined_score": 0.75,
                        "reason": "Similar but less proven",
                    }
                ],
                "retrieval_confidence": "high",
            },
            status_code=200,
        )

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            result = await client.retrieve_best(task="Find best solution")

            assert result.best_match is not None
            assert result.best_match.code_block.name == "Best Fetcher"
            assert result.best_match.combined_score == 0.92
            assert result.best_match.verdict_score == 0.95
            assert result.retrieval_confidence == "high"
            assert len(result.alternative_candidates) == 1

    @pytest.mark.asyncio
    async def test_async_retrieve_best_no_match(self, httpx_mock):
        """Async client should handle no match scenario."""
        httpx_mock.add_response(
            json={
                "best_match": None,
                "alternative_candidates": [],
                "retrieval_confidence": "low",
            },
            status_code=200,
        )

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            result = await client.retrieve_best(task="Very obscure task")

            assert result.best_match is None
            assert result.retrieval_confidence == "low"


# =============================================================================
# Get Code Files Tests
# =============================================================================


class TestGetCodeFiles:
    """Tests for get_code_files functionality."""

    @pytest.mark.asyncio
    async def test_async_get_code_files(self, httpx_mock):
        """Async client should retrieve code files for sandbox download."""
        httpx_mock.add_response(
            json={
                "files": [
                    {
                        "code_block_id": "cb_123",
                        "filename": "github_fetcher.py",
                        "source": "import requests\ndef fetch_user(username): pass",
                        "entrypoint": "fetch_user",
                        "description": "Fetches GitHub user data",
                        "language": "python",
                        "dependencies": ["requests"],
                        "verdict_score": 0.9,
                        "thumbs_up": 15,
                        "thumbs_down": 0,
                    }
                ],
                "task": "Fetch GitHub user",
                "total_found": 1,
            },
            status_code=200,
        )

        async with AsyncRaySurfer(api_key="test-key", base_url="http://test.local") as client:
            result = await client.get_code_files(
                task="Fetch GitHub user",
                top_k=5,
                min_verdict_score=0.3,
                prefer_complete=True,
            )

            assert result.total_found == 1
            assert len(result.files) == 1
            assert result.files[0].filename == "github_fetcher.py"
            assert result.files[0].verdict_score == 0.9
            assert "requests" in result.files[0].dependencies

    def test_sync_get_code_files(self, httpx_mock):
        """Sync client should retrieve code files for sandbox download."""
        httpx_mock.add_response(
            json={
                "files": [
                    {
                        "code_block_id": "cb_123",
                        "filename": "fetcher.py",
                        "source": "def fetch(): pass",
                        "entrypoint": "fetch",
                        "description": "Fetches data",
                        "language": "python",
                        "verdict_score": 0.85,
                        "thumbs_up": 10,
                        "thumbs_down": 2,
                    }
                ],
                "task": "Fetch data",
                "total_found": 1,
            },
            status_code=200,
        )

        with RaySurfer(api_key="test-key", base_url="http://test.local") as client:
            result = client.get_code_files(task="Fetch data")

            assert result.total_found == 1
            assert result.files[0].verdict_score == 0.85
