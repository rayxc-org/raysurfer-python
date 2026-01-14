"""Types for Claude Agent SDK integration"""

from typing import Any

from pydantic import BaseModel, Field


class CodeFile(BaseModel):
    """A code file ready to be written to sandbox"""

    code_block_id: str
    filename: str  # e.g., "github_fetcher.py"
    source: str  # Full source code
    entrypoint: str  # Function to call
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    language: str
    dependencies: dict[str, str] = Field(default_factory=dict)  # Package name -> version
    score: float = 0.0
    thumbs_up: int = 0
    thumbs_down: int = 0


class GetCodeFilesResponse(BaseModel):
    """Response with code files for a task"""

    files: list[CodeFile]
    task: str
    total_found: int
    add_to_llm_prompt: str = ""  # Pre-formatted string to append to LLM system prompt
