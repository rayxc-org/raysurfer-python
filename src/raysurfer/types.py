"""RaySurfer SDK types - mirrors the backend API types"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, JsonValue, model_validator

# Dictionary of JSON values (common for request/response payloads and schemas)
JsonDict = dict[str, JsonValue]


class ExecutionState(str, Enum):
    """Technical execution outcome - NOT a quality judgment"""

    COMPLETED = "completed"
    ERRORED = "errored"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class AgentVerdict(str, Enum):
    """Agent's judgment on whether an execution was useful"""

    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PENDING = "pending"


class SnipsDesired(str, Enum):
    """Scope of private snippets for retrieval"""

    COMPANY = "company"  # Organization-level snippets (Team or Enterprise tier)
    CLIENT = "client"  # Client workspace snippets (Enterprise tier only)


class CodeBlock(BaseModel):
    """A stored code block with metadata for semantic retrieval"""

    id: str
    name: str
    description: str
    source: str
    entrypoint: str
    input_schema: JsonDict = Field(default_factory=dict)
    output_schema: JsonDict = Field(default_factory=dict)
    language: str
    language_version: str | None = None
    dependencies: dict[str, str] = Field(default_factory=dict)  # Package name -> version
    tags: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    example_queries: list[str] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    agent_id: str | None = None


class ExecutionIO(BaseModel):
    """Stores the actual input/output data"""

    input_data: JsonDict
    input_hash: str = ""
    output_data: JsonValue = None
    output_hash: str = ""
    output_type: str = "unknown"


class AgentReview(BaseModel):
    """Agent's assessment of whether an execution was useful"""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    verdict: AgentVerdict
    reasoning: str
    what_worked: list[str] = Field(default_factory=list)
    what_didnt_work: list[str] = Field(default_factory=list)
    output_was_useful: bool
    output_was_correct: bool
    output_was_complete: bool
    error_was_appropriate: bool | None = None
    would_use_again: bool
    suggested_improvements: list[str] = Field(default_factory=list)
    required_workaround: bool = False
    workaround_description: str | None = None


class ExecutionRecord(BaseModel):
    """Full execution trace"""

    id: str
    code_block_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_state: ExecutionState
    duration_ms: int
    error_message: str | None = None
    error_type: str | None = None
    io: ExecutionIO
    triggering_task: str
    retrieval_score: float = 0.0
    verdict: AgentVerdict = AgentVerdict.PENDING
    review: AgentReview | None = None


class SearchMatch(BaseModel):
    """A code block match with scoring"""

    code_block: CodeBlock
    score: float
    vector_score: float | None = None
    verdict_score: float | None = None
    thumbs_up: int
    thumbs_down: int
    filename: str
    language: str
    entrypoint: str
    dependencies: dict[str, str] = Field(default_factory=dict)  # Package name -> version
    comments: list[JsonDict] = Field(default_factory=list)
    agent_id: str | None = None

    @model_validator(mode="after")
    def _set_default_compat_scores(self) -> "SearchMatch":
        """Backfill compatibility score aliases for legacy wrappers."""
        if self.vector_score is None:
            self.vector_score = self.score
        if self.verdict_score is None:
            self.verdict_score = self.score
        return self

    @property
    def combined_score(self) -> float:
        """Compatibility alias used by older wrappers."""
        return self.score


class SearchResponse(BaseModel):
    """Response from unified search endpoint"""

    matches: list[SearchMatch]
    total_found: int
    cache_hit: bool = False


class BestMatch(BaseModel):
    """The best matching code block with scoring"""

    code_block: CodeBlock
    score: float
    thumbs_up: int
    thumbs_down: int


class AlternativeCandidate(BaseModel):
    """An alternative candidate code block"""

    code_block_id: str
    name: str
    score: float
    reason: str


class FewShotExample(BaseModel):
    """A few-shot example for code generation"""

    task: str
    input_sample: JsonDict
    output_sample: JsonValue
    code_snippet: str


class TaskPattern(BaseModel):
    """A proven task→code mapping"""

    task_pattern: str
    code_block_id: str
    code_block_name: str
    thumbs_up: int
    thumbs_down: int
    last_thumbs_up: datetime | None = None
    last_thumbs_down: datetime | None = None


# Response types
class StoreCodeBlockResponse(BaseModel):
    success: bool
    code_block_id: str
    embedding_id: str
    message: str


class StoreExecutionResponse(BaseModel):
    success: bool
    execution_id: str
    pattern_updated: bool
    message: str


class RetrieveCodeBlockResponse(BaseModel):
    code_blocks: list["CodeBlockMatch"]
    total_found: int


class CodeBlockMatch(BaseModel):
    code_block: CodeBlock
    score: float
    thumbs_up: int
    thumbs_down: int
    recent_executions: list[ExecutionRecord] = Field(default_factory=list)


class RetrieveBestResponse(BaseModel):
    best_match: BestMatch | None
    alternative_candidates: list[AlternativeCandidate]
    retrieval_confidence: str


class FileWritten(BaseModel):
    """A file written during agent execution"""

    path: str
    content: str


class LogFile(BaseModel):
    """A log file for bulk grading (supports binary via base64)."""

    path: str
    content: str
    encoding: Literal["utf-8", "base64"] = "utf-8"
    content_type: str | None = None


class SubmitExecutionResultRequest(BaseModel):
    """Raw execution result - stores a single code file"""

    task: str
    file_written: FileWritten
    succeeded: bool
    use_raysurfer_ai_voting: bool = True
    user_vote: int | None = None


class SubmitExecutionResultResponse(BaseModel):
    """Response from submitting execution result"""

    success: bool
    code_blocks_stored: int
    message: str
    snippet_name: str | None = None


class BulkExecutionResultRequest(BaseModel):
    """Bulk execution upload for grading."""

    prompts: list[str]
    files_written: list[FileWritten]
    log_files: list[LogFile] | None = None
    use_raysurfer_ai_voting: bool = True
    user_votes: dict[str, int] | None = None


class BulkExecutionResultResponse(BaseModel):
    """Response from bulk execution upload."""

    success: bool
    code_blocks_stored: int
    votes_queued: int
    message: str
    status_url: str | None = None


# Auto Review API
class AutoReviewResponse(BaseModel):
    """Response with auto-generated review"""

    success: bool
    execution_id: str
    review: AgentReview
    message: str


# Retrieve Executions API
class RetrieveExecutionsResponse(BaseModel):
    """Response with executions"""

    executions: list[ExecutionRecord]
    total_found: int


# Public Snippet Browsing API
class PublicSnippet(BaseModel):
    """A public community snippet from the curated namespace."""

    id: str
    name: str
    description: str = ""
    source: str = ""
    language: str = "python"
    entrypoint: str = "main"
    thumbs_up: int = 0
    thumbs_down: int = 0
    created_at: str | None = None
    namespace: str = ""


class BrowsePublicResponse(BaseModel):
    """Response from browsing public snippets."""

    snippets: list[PublicSnippet]
    total: int
    has_more: bool = False


class SearchPublicResponse(BaseModel):
    """Response from searching public snippets."""

    snippets: list[PublicSnippet]
    total: int
    query: str


# Execute API types
class ToolDefinition(BaseModel):
    """A tool that can be called during execute()."""

    name: str
    description: str
    parameters: JsonDict


class ToolCallRecord(BaseModel):
    """Record of a tool call made during execution."""

    tool_name: str
    arguments: JsonDict
    result: str | None = None
    error: str | None = None
    duration_ms: int = 0


class ExecuteResult(BaseModel):
    """Result from an execute() call."""

    execution_id: str
    result: str | None = None
    exit_code: int = 0
    duration_ms: int = 0
    cache_hit: bool = False
    code_block_id: str | None = None
    error: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
