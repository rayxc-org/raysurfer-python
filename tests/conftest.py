"""Pytest configuration and shared fixtures for raysurfer tests."""

import sys
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

# =============================================================================
# Mock Claude Agent SDK
# =============================================================================
# The Claude Agent SDK requires the CLI to be installed, which makes unit testing
# difficult. We mock it here to allow testing the RaysurferClient logic without
# the actual SDK dependency.


@dataclass
class MockTextBlock:
    """Mock TextBlock from claude_agent_sdk."""

    type: str = "text"
    text: str = ""


@dataclass
class MockToolUseBlock:
    """Mock ToolUseBlock from claude_agent_sdk."""

    type: str = "tool_use"
    name: str = ""
    input: dict = None

    def __post_init__(self):
        if self.input is None:
            self.input = {}


@dataclass
class MockAssistantMessage:
    """Mock AssistantMessage from claude_agent_sdk."""

    role: str = "assistant"
    content: list = None

    def __post_init__(self):
        if self.content is None:
            self.content = []


@dataclass
class MockResultMessage:
    """Mock ResultMessage from claude_agent_sdk."""

    type: str = "result"
    subtype: str = "success"
    total_cost_usd: float = 0.0


def create_mock_sdk():
    """Create a mock claude_agent_sdk module."""
    mock_sdk = MagicMock()
    mock_sdk.ClaudeSDKClient = MagicMock()
    mock_sdk.ClaudeAgentOptions = MagicMock()
    mock_sdk.Message = MagicMock()
    mock_sdk.UserMessage = MagicMock()
    mock_sdk.AssistantMessage = MockAssistantMessage
    mock_sdk.SystemMessage = MagicMock()
    mock_sdk.ResultMessage = MockResultMessage
    mock_sdk.TextBlock = MockTextBlock
    mock_sdk.ThinkingBlock = MagicMock()
    mock_sdk.ToolUseBlock = MockToolUseBlock
    mock_sdk.ToolResultBlock = MagicMock()
    return mock_sdk


# Install mock SDK if not already present
if "claude_agent_sdk" not in sys.modules:
    sys.modules["claude_agent_sdk"] = create_mock_sdk()


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "asyncio: mark test as async")


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def sample_code_block_data():
    """Sample code block data for testing."""
    return {
        "id": "cb_sample",
        "name": "Sample Code Block",
        "description": "A sample code block for testing",
        "source": "def sample_function(x):\n    return x * 2",
        "entrypoint": "sample_function",
        "language": "python",
        "language_version": "3.11",
        "dependencies": ["requests"],
        "tags": ["sample", "test"],
        "capabilities": ["math"],
    }


@pytest.fixture
def sample_code_file_data():
    """Sample code file data for testing."""
    return {
        "code_block_id": "cb_sample_file",
        "filename": "sample.py",
        "source": "def process(data):\n    return data.upper()",
        "entrypoint": "process",
        "description": "Processes data",
        "language": "python",
        "dependencies": [],
        "verdict_score": 0.85,
        "thumbs_up": 20,
        "thumbs_down": 3,
    }


@pytest.fixture
def mock_retrieve_response():
    """Mock retrieve response with code blocks."""
    return {
        "code_blocks": [
            {
                "code_block": {
                    "id": "cb_123",
                    "name": "Test Fetcher",
                    "description": "Fetches test data",
                    "source": "def fetch(): pass",
                    "entrypoint": "fetch",
                    "language": "python",
                },
                "score": 0.9,
                "verdict_score": 0.85,
                "thumbs_up": 15,
                "thumbs_down": 2,
                "recent_executions": [],
            }
        ],
        "total_found": 1,
    }


@pytest.fixture
def mock_store_response():
    """Mock store code block response."""
    return {
        "success": True,
        "code_block_id": "cb_new",
        "embedding_id": "emb_new",
        "message": "Code block stored successfully",
    }


@pytest.fixture
def high_quality_code_block():
    """A high-quality code block with excellent scores."""
    return {
        "code_block": {
            "id": "cb_high_quality",
            "name": "High Quality Fetcher",
            "description": "A well-tested, reliable data fetcher",
            "source": """
import requests

def fetch_data(url: str, headers: dict = None) -> dict:
    \"\"\"Fetch JSON data from a URL.\"\"\"
    response = requests.get(url, headers=headers or {})
    response.raise_for_status()
    return response.json()
""",
            "entrypoint": "fetch_data",
            "language": "python",
            "dependencies": ["requests"],
            "tags": ["api", "fetch", "json"],
        },
        "score": 0.95,
        "verdict_score": 0.92,
        "thumbs_up": 50,
        "thumbs_down": 2,
        "recent_executions": [],
    }


@pytest.fixture
def low_quality_code_block():
    """A low-quality code block with poor scores."""
    return {
        "code_block": {
            "id": "cb_low_quality",
            "name": "Untested Snippet",
            "description": "A quick snippet, not well tested",
            "source": "def do_thing(): pass  # TODO: implement",
            "entrypoint": "do_thing",
            "language": "python",
        },
        "score": 0.6,
        "verdict_score": 0.2,
        "thumbs_up": 2,
        "thumbs_down": 10,
        "recent_executions": [],
    }
