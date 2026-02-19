"""RaySurfer SDK client"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid
from collections.abc import Callable
from types import TracebackType

import httpx
import websockets

from raysurfer._version import __version__
from raysurfer.exceptions import (
    APIError,
    AuthenticationError,
    CacheUnavailableError,
    RateLimitError,
)
from raysurfer.sdk_types import CodeFile, GetCodeFilesResponse
from raysurfer.types import (
    AgentReview,
    AgentVerdict,
    AlternativeCandidate,
    AutoReviewResponse,
    BestMatch,
    BrowsePublicResponse,
    BulkExecutionResultResponse,
    CodeBlock,
    CodeBlockMatch,
    ExecuteResult,
    ExecutionIO,
    ExecutionRecord,
    ExecutionState,
    FewShotExample,
    FileWritten,
    JsonDict,
    JsonValue,
    LogFile,
    PublicSnippet,
    RetrieveBestResponse,
    RetrieveCodeBlockResponse,
    RetrieveExecutionsResponse,
    SearchMatch,
    SearchPublicResponse,
    SearchResponse,
    SnipsDesired,
    StoreCodeBlockResponse,
    StoreExecutionResponse,
    SubmitExecutionResultResponse,
    TaskPattern,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.raysurfer.com"

# Maximum number of retry attempts for transient failures
MAX_RETRIES = 3
# Base delay in seconds for exponential backoff
RETRY_BASE_DELAY = 0.5
# HTTP status codes that should trigger a retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class AsyncRaySurfer:
    """Async client for RaySurfer API"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        organization_id: str | None = None,
        workspace_id: str | None = None,
        snips_desired: SnipsDesired | str | None = None,
        public_snips: bool = False,
        agent_id: str | None = None,
    ):
        """
        Initialize the RaySurfer async client.

        Args:
            api_key: RaySurfer API key (or set RAYSURFER_API_KEY env var)
            base_url: API base URL
            timeout: Request timeout in seconds
            organization_id: Optional organization ID for dedicated namespace (team/enterprise)
            workspace_id: Optional workspace ID for client-specific namespace (enterprise only)
            snips_desired: Scope of private snippets - "company" (Team/Enterprise) or "client" (Enterprise only)
            public_snips: Include community-contributed public snippets in search results
            agent_id: Optional agent identifier for agent-scoped snippet isolation
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self.public_snips = public_snips
        self.agent_id = agent_id
        # Convert string to SnipsDesired if needed
        if isinstance(snips_desired, str):
            self.snips_desired = SnipsDesired(snips_desired) if snips_desired else None
        else:
            self.snips_desired = snips_desired
        self._client: httpx.AsyncClient | None = None
        self._registered_tools: dict[str, tuple[ToolDefinition, Callable[..., JsonValue]]] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            # Add organization/workspace headers for namespace routing
            if self.organization_id:
                headers["X-Raysurfer-Org-Id"] = self.organization_id
            if self.workspace_id:
                headers["X-Raysurfer-Workspace-Id"] = self.workspace_id
            # Add snippet retrieval scope headers
            if self.snips_desired:
                headers["X-Raysurfer-Snips-Desired"] = self.snips_desired.value
            # Include community-contributed public snippets
            if self.public_snips:
                headers["X-Raysurfer-Public-Snips"] = "true"
            # SDK version for tracking
            headers["X-Raysurfer-SDK-Version"] = f"python/{__version__}"
            if self.agent_id:
                headers["X-Raysurfer-Agent-Id"] = self.agent_id
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "AsyncRaySurfer":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    def _workspace_headers(self, workspace_id: str | None) -> dict[str, str] | None:
        """Build header overrides for per-request workspace_id."""
        if workspace_id is None:
            return None
        return {"X-Raysurfer-Workspace-Id": workspace_id}

    async def _request(
        self, method: str, path: str, headers_override: dict[str, str] | None = None, **kwargs: JsonValue
    ) -> JsonDict:
        client = await self._get_client()
        last_exception: Exception | None = None

        # Apply per-request header overrides
        request_kwargs = kwargs.copy()
        if headers_override:
            request_kwargs["headers"] = headers_override

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.request(method, path, **request_kwargs)

                if response.status_code == 401:
                    raise AuthenticationError("Invalid API key")
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else RETRY_BASE_DELAY * (2**attempt)
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                        await asyncio.sleep(delay)
                        continue
                    raise RateLimitError(retry_after=delay)
                if response.status_code in RETRYABLE_STATUS_CODES:
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_BASE_DELAY * (2**attempt)
                        logger.warning(
                            f"Server error {response.status_code}, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                        )
                        await asyncio.sleep(delay)
                        continue
                if response.status_code >= 400:
                    raise APIError(response.text, status_code=response.status_code)

                return response.json()
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        f"Network error: {e}, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise CacheUnavailableError(f"Failed to connect after {MAX_RETRIES} attempts: {e}") from e

        # Should not reach here, but just in case
        raise CacheUnavailableError(f"Request failed after {MAX_RETRIES} attempts") from last_exception

    # =========================================================================
    # Store API
    # =========================================================================

    async def store_code_block(
        self,
        name: str,
        source: str,
        entrypoint: str,
        language: str,
        description: str = "",
        input_schema: JsonDict | None = None,
        output_schema: JsonDict | None = None,
        language_version: str | None = None,
        dependencies: dict[str, str] | None = None,
        tags: list[str] | None = None,
        capabilities: list[str] | None = None,
        example_queries: list[str] | None = None,
    ) -> StoreCodeBlockResponse:
        """Store a new code block"""
        data = {
            "name": name,
            "description": description,
            "source": source,
            "entrypoint": entrypoint,
            "language": language,
            "input_schema": input_schema or {},
            "output_schema": output_schema or {},
            "language_version": language_version,
            "dependencies": dependencies or {},
            "tags": tags or [],
            "capabilities": capabilities or [],
            "example_queries": example_queries,
        }
        result = await self._request("POST", "/api/store/code-block", json=data)
        return StoreCodeBlockResponse(**result)

    async def store_execution(
        self,
        code_block_id: str,
        triggering_task: str,
        input_data: JsonDict,
        output_data: JsonValue,
        execution_state: ExecutionState = ExecutionState.COMPLETED,
        duration_ms: int = 0,
        error_message: str | None = None,
        error_type: str | None = None,
        verdict: AgentVerdict | None = None,
        review: AgentReview | None = None,
    ) -> StoreExecutionResponse:
        """Store an execution record"""
        io = ExecutionIO(
            input_data=input_data,
            output_data=output_data,
            output_type=type(output_data).__name__,
        )
        data = {
            "code_block_id": code_block_id,
            "triggering_task": triggering_task,
            "io": io.model_dump(),
            "execution_state": execution_state.value,
            "duration_ms": duration_ms,
            "error_message": error_message,
            "error_type": error_type,
            "verdict": verdict.value if verdict else None,
            "review": review.model_dump() if review else None,
        }
        result = await self._request("POST", "/api/store/execution", json=data)
        return StoreExecutionResponse(**result)

    async def upload_new_code_snip(
        self,
        task: str,
        file_written: FileWritten | None = None,
        succeeded: bool = True,
        use_raysurfer_ai_voting: bool = True,
        user_vote: int | None = None,
        execution_logs: str | None = None,
        run_url: str | None = None,
        workspace_id: str | None = None,
        dependencies: dict[str, str] | None = None,
        public: bool = False,
        vote_source: str | None = None,
        vote_count: int = 1,
        files_written: list[FileWritten] | None = None,
        auto_vote: bool | None = None,
    ) -> SubmitExecutionResultResponse:
        """
        Upload a single code file from an execution.

        Args:
            task: The task that was executed.
            file_written: The file created during execution.
            files_written: Compatibility alias for multiple files. If provided,
                uploads each file sequentially and returns an aggregated result.
            succeeded: Whether the task completed successfully.
            use_raysurfer_ai_voting: Let Raysurfer AI vote on stored blocks (default True).
                Ignored when user_vote is provided.
            auto_vote: Compatibility alias for use_raysurfer_ai_voting.
            user_vote: User-provided vote (1 for thumbs up, -1 for thumbs down).
                When provided, AI voting is automatically skipped.
            execution_logs: Captured stdout/stderr for vote context.
            run_url: URL to the finished run (e.g. logs page, CI run, LangSmith trace).
            workspace_id: Override client-level workspace_id for this request.
            dependencies: Package dependencies with versions (e.g., {"pandas": "2.1.0"}).
            public: Upload to the public community namespace (default False).
            vote_source: Origin of the vote (e.g. "cli", "mcp", "sdk").
            vote_count: Number of votes to apply (default 1).
        """
        if file_written is not None and files_written is not None:
            raise ValueError("Provide either file_written or files_written, not both.")

        if auto_vote is not None:
            use_raysurfer_ai_voting = auto_vote

        if files_written is not None:
            if len(files_written) == 0:
                raise ValueError("files_written must contain at least one file.")

            if len(files_written) == 1:
                file_written = files_written[0]
            else:
                responses: list[SubmitExecutionResultResponse] = []
                for file in files_written:
                    response = await self.upload_new_code_snip(
                        task=task,
                        file_written=file,
                        succeeded=succeeded,
                        use_raysurfer_ai_voting=use_raysurfer_ai_voting,
                        user_vote=user_vote,
                        execution_logs=execution_logs,
                        run_url=run_url,
                        workspace_id=workspace_id,
                        dependencies=dependencies,
                        public=public,
                        vote_source=vote_source,
                        vote_count=vote_count,
                    )
                    responses.append(response)

                return SubmitExecutionResultResponse(
                    success=all(response.success for response in responses),
                    code_blocks_stored=sum(response.code_blocks_stored for response in responses),
                    message=f"Uploaded {len(files_written)} files via compatibility path.",
                )

        if file_written is None:
            raise ValueError("Missing required file input: provide file_written or files_written.")

        data: JsonDict = {
            "task": task,
            "file_written": file_written.model_dump(),
            "succeeded": succeeded,
            "use_raysurfer_ai_voting": use_raysurfer_ai_voting,
        }
        if user_vote is not None:
            data["user_vote"] = user_vote
        if execution_logs is not None:
            data["execution_logs"] = execution_logs
        if run_url is not None:
            data["run_url"] = run_url
        if dependencies is not None:
            data["dependencies"] = dependencies
        if public:
            data["public"] = True
        if vote_source is not None:
            data["vote_source"] = vote_source
        if vote_count != 1:
            data["vote_count"] = vote_count
        result = await self._request(
            "POST", "/api/store/execution-result", headers_override=self._workspace_headers(workspace_id), json=data
        )
        return SubmitExecutionResultResponse(**result)

    # Backwards-compatible alias
    upload_new_code_snips = upload_new_code_snip

    async def upload_bulk_code_snips(
        self,
        prompts: list[str],
        files_written: list[FileWritten],
        log_files: list[LogFile] | None = None,
        use_raysurfer_ai_voting: bool = True,
        user_votes: dict[str, int] | None = None,
        workspace_id: str | None = None,
        vote_source: str | None = None,
        vote_count: int = 1,
    ) -> BulkExecutionResultResponse:
        """
        Bulk upload prompts, logs, and code files for sandboxed grading.

        Args:
            prompts: Ordered list of raw user prompts.
            files_written: Code files to store and grade.
            log_files: Log files (any format; use encoding="base64" for binary).
            use_raysurfer_ai_voting: Let Raysurfer AI vote on stored blocks (default True).
                Ignored when user_votes is provided.
            user_votes: Dict of filename to vote (1 for thumbs up, -1 for thumbs down).
                When provided, AI voting is automatically skipped.
            workspace_id: Override client-level workspace_id for this request.
            vote_source: Origin of the vote (e.g. "cli", "mcp", "sdk").
            vote_count: Number of votes to apply (default 1).
        """
        data: JsonDict = {
            "prompts": prompts,
            "files_written": [f.model_dump() for f in files_written],
            "use_raysurfer_ai_voting": use_raysurfer_ai_voting,
        }
        if log_files is not None:
            data["log_files"] = [f.model_dump() for f in log_files]
        if user_votes is not None:
            data["user_votes"] = user_votes
        if vote_source is not None:
            data["vote_source"] = vote_source
        if vote_count != 1:
            data["vote_count"] = vote_count

        result = await self._request(
            "POST",
            "/api/store/bulk-execution-result",
            headers_override=self._workspace_headers(workspace_id),
            json=data,
        )
        return BulkExecutionResultResponse(**result)

    # =========================================================================
    # Retrieve API
    # =========================================================================

    async def search(
        self,
        task: str,
        top_k: int = 5,
        min_verdict_score: float = 0.3,
        min_human_upvotes: int = 0,
        prefer_complete: bool = False,
        input_schema: JsonDict | None = None,
        workspace_id: str | None = None,
    ) -> SearchResponse:
        """Unified search for cached code snippets.

        Args:
            task: The task to search for.
            top_k: Maximum number of results to return.
            min_verdict_score: Minimum verdict score threshold.
            min_human_upvotes: Minimum number of human upvotes required.
            prefer_complete: Prefer complete code blocks.
            input_schema: Optional input schema for filtering.
            workspace_id: Override client-level workspace_id for this request.
        """
        data = {
            "task": task,
            "top_k": top_k,
            "min_verdict_score": min_verdict_score,
            "min_human_upvotes": min_human_upvotes,
            "prefer_complete": prefer_complete,
            "input_schema": input_schema,
        }
        result = await self._request(
            "POST", "/api/retrieve/search", headers_override=self._workspace_headers(workspace_id), json=data
        )
        matches = [
            SearchMatch(
                code_block=CodeBlock(**m["code_block"]),
                score=m["score"],
                vector_score=m.get("vector_score"),
                verdict_score=m.get("verdict_score"),
                thumbs_up=m["thumbs_up"],
                thumbs_down=m["thumbs_down"],
                filename=m["filename"],
                language=m["language"],
                entrypoint=m["entrypoint"],
                dependencies=m.get("dependencies", {}),
            )
            for m in result["matches"]
        ]
        return SearchResponse(
            matches=matches,
            total_found=result["total_found"],
            cache_hit=result.get("cache_hit", False),
        )

    async def get_code_snips(
        self,
        task: str,
        top_k: int = 10,
        min_verdict_score: float = 0.0,
    ) -> RetrieveCodeBlockResponse:
        """Get cached code snippets -- delegates to search()."""
        response = await self.search(task=task, top_k=top_k, min_verdict_score=min_verdict_score)
        code_blocks = [
            CodeBlockMatch(
                code_block=m.code_block,
                score=m.score,
                thumbs_up=m.thumbs_up,
                thumbs_down=m.thumbs_down,
            )
            for m in response.matches
        ]
        return RetrieveCodeBlockResponse(code_blocks=code_blocks, total_found=response.total_found)

    async def retrieve_best(
        self,
        task: str,
        top_k: int = 10,
        min_verdict_score: float = 0.0,
    ) -> RetrieveBestResponse:
        """Get the best code block -- delegates to search()."""
        response = await self.search(task=task, top_k=top_k, min_verdict_score=min_verdict_score)
        best_match = None
        if response.matches:
            m = response.matches[0]
            best_match = BestMatch(
                code_block=m.code_block,
                score=m.score,
                thumbs_up=m.thumbs_up,
                thumbs_down=m.thumbs_down,
            )
        alternatives = [
            AlternativeCandidate(
                code_block_id=m.code_block.id,
                name=m.code_block.name,
                score=m.score,
                reason=f"{m.thumbs_up} thumbs up, {m.thumbs_down} thumbs down"
                if m.thumbs_up > 0
                else "No execution history",
            )
            for m in response.matches[1:4]
        ]
        return RetrieveBestResponse(
            best_match=best_match,
            alternative_candidates=alternatives,
            retrieval_confidence=str(round(response.matches[0].score, 4)) if response.matches else "0",
        )

    async def get_few_shot_examples(
        self,
        task: str,
        k: int = 3,
    ) -> list[FewShotExample]:
        """Retrieve few-shot examples for code generation"""
        data = {"task": task, "k": k}
        result = await self._request("POST", "/api/retrieve/few-shot-examples", json=data)
        return [FewShotExample(**ex) for ex in result["examples"]]

    async def get_task_patterns(
        self,
        task: str | None = None,
        code_block_id: str | None = None,
        min_thumbs_up: int = 0,
        top_k: int = 20,
    ) -> list[TaskPattern]:
        """Retrieve proven task->code mappings"""
        data = {
            "task": task,
            "code_block_id": code_block_id,
            "min_thumbs_up": min_thumbs_up,
            "top_k": top_k,
        }
        result = await self._request("POST", "/api/retrieve/task-patterns", json=data)
        return [TaskPattern(**p) for p in result["patterns"]]

    async def get_code_files(
        self,
        task: str,
        top_k: int = 5,
        min_verdict_score: float = 0.3,
        prefer_complete: bool = True,
        cache_dir: str = ".raysurfer_code",
    ) -> GetCodeFilesResponse:
        """Get code files -- delegates to search()."""
        response = await self.search(
            task=task, top_k=top_k, min_verdict_score=min_verdict_score, prefer_complete=prefer_complete
        )
        files = [
            CodeFile(
                code_block_id=m.code_block.id,
                filename=m.filename,
                source=m.code_block.source,
                entrypoint=m.entrypoint,
                description=m.code_block.description,
                input_schema=m.code_block.input_schema,
                output_schema=m.code_block.output_schema,
                language=m.language,
                dependencies=m.dependencies,
                score=m.score,
                thumbs_up=m.thumbs_up,
                thumbs_down=m.thumbs_down,
            )
            for m in response.matches
        ]
        add_to_llm_prompt = self._format_llm_prompt(files, cache_dir)
        return GetCodeFilesResponse(
            files=files, task=task, total_found=response.total_found, add_to_llm_prompt=add_to_llm_prompt
        )

    def _format_llm_prompt(self, files: list[CodeFile], cache_dir: str | None = None) -> str:
        """Format a prompt string listing all retrieved code files."""
        if not files:
            return ""

        lines = [
            "\n\n## IMPORTANT: Pre-validated Code Files Available\n",
            "The following validated code has been retrieved from the cache. "
            "Use these files directly instead of regenerating code.\n",
        ]

        for f in files:
            if cache_dir:
                import os

                full_path = os.path.join(cache_dir, f.filename)
                lines.append(f"\n### `{f.filename}` -> `{full_path}`")
            else:
                lines.append(f"\n### `{f.filename}`")
            lines.append(f"- **Description**: {f.description}")
            lines.append(f"- **Language**: {f.language}")
            lines.append(f"- **Entrypoint**: `{f.entrypoint}`")
            lines.append(f"- **Confidence**: {f.score:.0%}")
            if f.dependencies:
                deps = [f"{k}@{v}" for k, v in f.dependencies.items()]
                lines.append(f"- **Dependencies**: {', '.join(deps)}")

        lines.append("\n\n**Instructions**:")
        lines.append("1. Read the cached file(s) before writing new code")
        lines.append("2. Use the cached code as your starting point")
        lines.append("3. Only modify if the task requires specific changes")
        lines.append("4. Do not regenerate code that already exists\n")

        return "\n".join(lines)

    async def vote_code_snip(
        self,
        task: str,
        code_block_id: str,
        code_block_name: str,
        code_block_description: str,
        succeeded: bool,
    ) -> JsonDict:
        """
        Vote on whether a cached code snippet was useful.

        This triggers background voting to assess whether the cached code
        actually helped complete the task successfully.
        """
        data = {
            "task": task,
            "code_block_id": code_block_id,
            "code_block_name": code_block_name,
            "code_block_description": code_block_description,
            "succeeded": succeeded,
        }
        return await self._request("POST", "/api/store/cache-usage", json=data)

    async def comment_on_code_snip(self, code_block_id: str, text: str) -> JsonDict:
        """Add a comment to a cached code snippet."""
        return await self._request("POST", "/api/store/comment", json={
            "code_block_id": code_block_id, "text": text,
        })

    # =========================================================================
    # Auto Review API
    # =========================================================================

    async def auto_review(
        self,
        execution_id: str,
        triggering_task: str,
        execution_state: ExecutionState,
        input_data: JsonDict,
        output_data: JsonValue,
        code_block_name: str,
        code_block_description: str,
        error_message: str | None = None,
    ) -> AutoReviewResponse:
        """
        Get an auto-generated review using Claude Opus 4.6.
        Useful for programmatically reviewing execution results.
        """
        data = {
            "execution_id": execution_id,
            "triggering_task": triggering_task,
            "execution_state": execution_state.value,
            "input_data": input_data,
            "output_data": output_data,
            "code_block_name": code_block_name,
            "code_block_description": code_block_description,
            "error_message": error_message,
        }
        result = await self._request("POST", "/api/store/auto-review", json=data)
        return AutoReviewResponse(
            success=result["success"],
            execution_id=result["execution_id"],
            review=AgentReview(**result["review"]),
            message=result["message"],
        )

    async def get_executions(
        self,
        code_block_id: str | None = None,
        task: str | None = None,
        verdict: AgentVerdict | None = None,
        limit: int = 20,
    ) -> RetrieveExecutionsResponse:
        """Retrieve execution records by code block ID, task, or verdict."""
        data = {
            "code_block_id": code_block_id,
            "task": task,
            "verdict": verdict.value if verdict else None,
            "limit": limit,
        }
        result = await self._request("POST", "/api/retrieve/executions", json=data)
        executions = [ExecutionRecord(**ex) for ex in result["executions"]]
        return RetrieveExecutionsResponse(
            executions=executions,
            total_found=result["total_found"],
        )

    # =========================================================================
    # Execute API (tool calling)
    # =========================================================================

    def tool(self, fn: Callable[..., JsonValue]) -> Callable[..., JsonValue]:
        """Register a function as a tool for execute().

        Introspects the function signature to build a JSON schema.
        Both sync and async callbacks are supported.
        """
        sig = inspect.signature(fn)
        type_map: dict[type, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}
        properties: dict[str, dict[str, str]] = {}
        required: list[str] = []
        for param_name, param in sig.parameters.items():
            annotation = param.annotation
            json_type = type_map.get(annotation, "string")
            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        schema: JsonDict = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        tool_def = ToolDefinition(
            name=fn.__name__,
            description=fn.__doc__ or "",
            parameters=schema,
        )
        self._registered_tools[fn.__name__] = (tool_def, fn)
        return fn

    async def execute(
        self,
        task: str,
        user_code: str | None = None,
        timeout: int = 300,
        codegen_api_key: str | None = None,
        codegen_prompt: str | None = None,
        codegen_model: str = "claude-opus-4-6",
    ) -> ExecuteResult:
        """Execute a task with registered tools in a sandbox.

        Args:
            task: The task to execute.
            user_code: Python code generated client-side to run directly.
            timeout: Maximum execution time in seconds.
            codegen_api_key: Provider API key used inside sandbox codegen mode.
            codegen_prompt: User-provided prompt for sandbox code generation.
            codegen_model: Provider model name for sandbox code generation.
        """
        has_user_code = isinstance(user_code, str) and bool(user_code.strip())
        has_codegen = codegen_api_key is not None or codegen_prompt is not None
        if has_user_code == has_codegen:
            raise ValueError(
                "Invalid execute mode: provide exactly one of user_code or "
                "(codegen_api_key + codegen_prompt). "
                f"Received user_code={user_code!r}, "
                f"codegen_api_key={codegen_api_key is not None}, "
                f"codegen_prompt={codegen_prompt!r}. "
                "Docs: https://docs.raysurfer.com/sdk/python#programmatic-tool-calling"
            )

        if has_codegen:
            if not isinstance(codegen_api_key, str) or not codegen_api_key.strip():
                raise ValueError(
                    f"Invalid codegen_api_key value: {codegen_api_key!r}. "
                    "Expected a non-empty API key string. "
                    "Docs: https://docs.raysurfer.com/sdk/python#programmatic-tool-calling"
                )
            if not isinstance(codegen_prompt, str) or not codegen_prompt.strip():
                raise ValueError(
                    f"Invalid codegen_prompt value: {codegen_prompt!r}. "
                    "Expected a non-empty prompt string. "
                    "Docs: https://docs.raysurfer.com/sdk/python#programmatic-tool-calling"
                )
            if not isinstance(codegen_model, str) or not codegen_model.strip():
                raise ValueError(
                    f"Invalid codegen_model value: {codegen_model!r}. "
                    "Expected a non-empty model string. "
                    "Docs: https://docs.raysurfer.com/sdk/python#programmatic-tool-calling"
                )

        session_id = str(uuid.uuid4())
        ws_url = f"{self.base_url.replace('http', 'ws')}/api/execute/ws/{session_id}"

        ws_headers: dict[str, str] = {}
        if self.api_key:
            ws_headers["Authorization"] = f"Bearer {self.api_key}"

        ws_conn = await websockets.connect(ws_url, additional_headers=ws_headers)

        async def _handle_tool_calls() -> None:
            """Listen for tool_call messages on the WebSocket and dispatch to registered callbacks."""
            try:
                async for raw_msg in ws_conn:
                    msg = json.loads(raw_msg)
                    if msg.get("type") == "tool_call":
                        request_id = msg["request_id"]
                        tool_name = msg["tool_name"]
                        arguments = msg.get("arguments", {})
                        tool_entry = self._registered_tools.get(tool_name)
                        if tool_entry is None:
                            await ws_conn.send(
                                json.dumps(
                                    {
                                        "type": "tool_result",
                                        "request_id": request_id,
                                        "result": f"Error: unknown tool '{tool_name}'",
                                    }
                                )
                            )
                            continue
                        _, callback = tool_entry
                        try:
                            if inspect.iscoroutinefunction(callback):
                                result = await callback(**arguments)
                            else:
                                result = callback(**arguments)
                            await ws_conn.send(
                                json.dumps(
                                    {
                                        "type": "tool_result",
                                        "request_id": request_id,
                                        "result": str(result),
                                    }
                                )
                            )
                        except Exception as exc:
                            await ws_conn.send(
                                json.dumps(
                                    {
                                        "type": "tool_result",
                                        "request_id": request_id,
                                        "result": f"Error: {exc}",
                                    }
                                )
                            )
            except websockets.ConnectionClosed:
                pass

        listener_task = asyncio.create_task(_handle_tool_calls())

        try:
            tool_schemas = [defn.model_dump() for defn, _ in self._registered_tools.values()]
            request_payload: JsonDict = {
                "task": task,
                "tools": tool_schemas,
                "session_id": session_id,
                "timeout_seconds": timeout,
            }
            if has_user_code:
                request_payload["user_code"] = user_code or ""
            else:
                request_payload["codegen"] = {
                    "provider": "anthropic",
                    "api_key": codegen_api_key or "",
                    "model": codegen_model,
                    "prompt": codegen_prompt or "",
                }
            result = await self._request(
                "POST",
                "/api/execute/run",
                json=request_payload,
            )
            return ExecuteResult(**result)
        finally:
            listener_task.cancel()
            try:
                await listener_task
            except asyncio.CancelledError:
                pass
            await ws_conn.close()

    async def run_script(
        self,
        s3_key: str,
        params: dict[str, str] | None = None,
        timeout: int = 300,
    ) -> ExecuteResult:
        """Execute an S3-stored script in a remote sandbox with tool callbacks.

        Args:
            s3_key: S3 key of the script to execute.
            params: Optional parameters injected as environment variables.
            timeout: Maximum execution time in seconds.
        """
        session_id = str(uuid.uuid4())
        ws_url = f"{self.base_url.replace('http', 'ws')}/api/execute/ws/{session_id}"

        ws_headers: dict[str, str] = {}
        if self.api_key:
            ws_headers["Authorization"] = f"Bearer {self.api_key}"

        ws_conn = await websockets.connect(ws_url, additional_headers=ws_headers)

        async def _handle_tool_calls() -> None:
            """Listen for tool_call messages on the WebSocket and dispatch to registered callbacks."""
            try:
                async for raw_msg in ws_conn:
                    msg = json.loads(raw_msg)
                    if msg.get("type") == "tool_call":
                        request_id = msg["request_id"]
                        tool_name = msg["tool_name"]
                        arguments = msg.get("arguments", {})
                        tool_entry = self._registered_tools.get(tool_name)
                        if tool_entry is None:
                            await ws_conn.send(
                                json.dumps(
                                    {
                                        "type": "tool_result",
                                        "request_id": request_id,
                                        "result": f"Error: unknown tool '{tool_name}'",
                                    }
                                )
                            )
                            continue
                        _, callback = tool_entry
                        try:
                            if inspect.iscoroutinefunction(callback):
                                result = await callback(**arguments)
                            else:
                                result = callback(**arguments)
                            await ws_conn.send(
                                json.dumps(
                                    {
                                        "type": "tool_result",
                                        "request_id": request_id,
                                        "result": str(result),
                                    }
                                )
                            )
                        except Exception as exc:
                            await ws_conn.send(
                                json.dumps(
                                    {
                                        "type": "tool_result",
                                        "request_id": request_id,
                                        "result": f"Error: {exc}",
                                    }
                                )
                            )
            except websockets.ConnectionClosed:
                pass

        listener_task = asyncio.create_task(_handle_tool_calls())

        try:
            tool_schemas = [defn.model_dump() for defn, _ in self._registered_tools.values()]
            request_payload: JsonDict = {
                "s3_key": s3_key,
                "tools": tool_schemas,
                "session_id": session_id,
                "timeout_seconds": timeout,
            }
            if params:
                request_payload["params"] = params

            result = await self._request(
                "POST",
                "/api/execute/run-script",
                json=request_payload,
            )
            return ExecuteResult(**result)
        finally:
            listener_task.cancel()
            try:
                await listener_task
            except asyncio.CancelledError:
                pass
            await ws_conn.close()

    async def execute_generated_code(
        self,
        task: str,
        user_code: str,
        timeout: int = 300,
    ) -> ExecuteResult:
        """Execute client-generated Python code in the remote sandbox with tool callbacks."""
        return await self.execute(task=task, user_code=user_code, timeout=timeout)

    async def execute_with_sandbox_codegen(
        self,
        task: str,
        codegen_api_key: str,
        codegen_prompt: str,
        timeout: int = 300,
        codegen_model: str = "claude-opus-4-6",
    ) -> ExecuteResult:
        """Generate Python code inside the sandbox, then execute it with tool callbacks."""
        return await self.execute(
            task=task,
            timeout=timeout,
            codegen_api_key=codegen_api_key,
            codegen_prompt=codegen_prompt,
            codegen_model=codegen_model,
        )

    # =========================================================================
    # Public Snippet Browsing (no API key required)
    # =========================================================================

    async def browse_public(
        self,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "upvoted",
        language: str | None = None,
    ) -> BrowsePublicResponse:
        """Browse community public snippets without authentication.

        Args:
            limit: Maximum number of snippets to return.
            offset: Pagination offset.
            sort_by: Sort order - "upvoted" or "recent".
            language: Filter by programming language.
        """
        data: JsonDict = {
            "limit": limit,
            "offset": offset,
            "sort_by": sort_by,
        }
        if language:
            data["language"] = language
        result = await self._request("POST", "/api/snippets/public/list", json=data)
        snippets = [PublicSnippet(**s) for s in result["snippets"]]
        return BrowsePublicResponse(
            snippets=snippets,
            total=result["total"],
            has_more=result.get("has_more", False),
        )

    async def search_public(
        self,
        query: str,
        limit: int = 20,
        language: str | None = None,
    ) -> SearchPublicResponse:
        """Search community public snippets by keyword without authentication.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            language: Filter by programming language.
        """
        data: JsonDict = {
            "query": query,
            "limit": limit,
        }
        if language:
            data["language"] = language
        result = await self._request("POST", "/api/snippets/public/search", json=data)
        snippets = [PublicSnippet(**s) for s in result["snippets"]]
        return SearchPublicResponse(
            snippets=snippets,
            total=result["total"],
            query=result["query"],
        )


class RaySurfer:
    """Sync client for RaySurfer API"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        organization_id: str | None = None,
        workspace_id: str | None = None,
        snips_desired: SnipsDesired | str | None = None,
        public_snips: bool = False,
        agent_id: str | None = None,
    ):
        """
        Initialize the RaySurfer sync client.

        Args:
            api_key: RaySurfer API key (or set RAYSURFER_API_KEY env var)
            base_url: API base URL
            timeout: Request timeout in seconds
            organization_id: Optional organization ID for dedicated namespace (team/enterprise)
            workspace_id: Optional workspace ID for client-specific namespace (enterprise only)
            snips_desired: Scope of private snippets - "company" (Team/Enterprise) or "client" (Enterprise only)
            public_snips: Include community-contributed public snippets in search results
            agent_id: Optional agent identifier for agent-scoped snippet isolation
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self.public_snips = public_snips
        self.agent_id = agent_id
        # Convert string to SnipsDesired if needed
        if isinstance(snips_desired, str):
            self.snips_desired = SnipsDesired(snips_desired) if snips_desired else None
        else:
            self.snips_desired = snips_desired
        self._client: httpx.Client | None = None
        self._async_inner = AsyncRaySurfer(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            organization_id=organization_id,
            workspace_id=workspace_id,
            snips_desired=snips_desired,
            public_snips=public_snips,
            agent_id=agent_id,
        )

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            # Add organization/workspace headers for namespace routing
            if self.organization_id:
                headers["X-Raysurfer-Org-Id"] = self.organization_id
            if self.workspace_id:
                headers["X-Raysurfer-Workspace-Id"] = self.workspace_id
            # Add snippet retrieval scope headers
            if self.snips_desired:
                headers["X-Raysurfer-Snips-Desired"] = self.snips_desired.value
            # Include community-contributed public snippets
            if self.public_snips:
                headers["X-Raysurfer-Public-Snips"] = "true"
            # SDK version for tracking
            headers["X-Raysurfer-SDK-Version"] = f"python/{__version__}"
            if self.agent_id:
                headers["X-Raysurfer-Agent-Id"] = self.agent_id
            self._client = httpx.Client(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "RaySurfer":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def _workspace_headers(self, workspace_id: str | None) -> dict[str, str] | None:
        """Build header overrides for per-request workspace_id."""
        if workspace_id is None:
            return None
        return {"X-Raysurfer-Workspace-Id": workspace_id}

    def _request(
        self, method: str, path: str, headers_override: dict[str, str] | None = None, **kwargs: JsonValue
    ) -> JsonDict:
        import time as _time

        client = self._get_client()
        last_exception: Exception | None = None

        # Apply per-request header overrides
        request_kwargs = kwargs.copy()
        if headers_override:
            request_kwargs["headers"] = headers_override

        for attempt in range(MAX_RETRIES):
            try:
                response = client.request(method, path, **request_kwargs)

                if response.status_code == 401:
                    raise AuthenticationError("Invalid API key")
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else RETRY_BASE_DELAY * (2**attempt)
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                        _time.sleep(delay)
                        continue
                    raise RateLimitError(retry_after=delay)
                if response.status_code in RETRYABLE_STATUS_CODES:
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_BASE_DELAY * (2**attempt)
                        logger.warning(
                            f"Server error {response.status_code}, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                        )
                        _time.sleep(delay)
                        continue
                if response.status_code >= 400:
                    raise APIError(response.text, status_code=response.status_code)

                return response.json()
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        f"Network error: {e}, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    _time.sleep(delay)
                    continue
                raise CacheUnavailableError(f"Failed to connect after {MAX_RETRIES} attempts: {e}") from e

        # Should not reach here, but just in case
        raise CacheUnavailableError(f"Request failed after {MAX_RETRIES} attempts") from last_exception

    # =========================================================================
    # Store API
    # =========================================================================

    def store_code_block(
        self,
        name: str,
        source: str,
        entrypoint: str,
        language: str,
        description: str = "",
        input_schema: JsonDict | None = None,
        output_schema: JsonDict | None = None,
        language_version: str | None = None,
        dependencies: dict[str, str] | None = None,
        tags: list[str] | None = None,
        capabilities: list[str] | None = None,
        example_queries: list[str] | None = None,
    ) -> StoreCodeBlockResponse:
        """Store a new code block"""
        data = {
            "name": name,
            "description": description,
            "source": source,
            "entrypoint": entrypoint,
            "language": language,
            "input_schema": input_schema or {},
            "output_schema": output_schema or {},
            "language_version": language_version,
            "dependencies": dependencies or {},
            "tags": tags or [],
            "capabilities": capabilities or [],
            "example_queries": example_queries,
        }
        result = self._request("POST", "/api/store/code-block", json=data)
        return StoreCodeBlockResponse(**result)

    def store_execution(
        self,
        code_block_id: str,
        triggering_task: str,
        input_data: JsonDict,
        output_data: JsonValue,
        execution_state: ExecutionState = ExecutionState.COMPLETED,
        duration_ms: int = 0,
        error_message: str | None = None,
        error_type: str | None = None,
        verdict: AgentVerdict | None = None,
        review: AgentReview | None = None,
    ) -> StoreExecutionResponse:
        """Store an execution record"""
        io = ExecutionIO(
            input_data=input_data,
            output_data=output_data,
            output_type=type(output_data).__name__,
        )
        data = {
            "code_block_id": code_block_id,
            "triggering_task": triggering_task,
            "io": io.model_dump(),
            "execution_state": execution_state.value,
            "duration_ms": duration_ms,
            "error_message": error_message,
            "error_type": error_type,
            "verdict": verdict.value if verdict else None,
            "review": review.model_dump() if review else None,
        }
        result = self._request("POST", "/api/store/execution", json=data)
        return StoreExecutionResponse(**result)

    def upload_new_code_snip(
        self,
        task: str,
        file_written: FileWritten | None = None,
        succeeded: bool = True,
        use_raysurfer_ai_voting: bool = True,
        user_vote: int | None = None,
        execution_logs: str | None = None,
        run_url: str | None = None,
        workspace_id: str | None = None,
        dependencies: dict[str, str] | None = None,
        public: bool = False,
        vote_source: str | None = None,
        vote_count: int = 1,
        files_written: list[FileWritten] | None = None,
        auto_vote: bool | None = None,
    ) -> SubmitExecutionResultResponse:
        """
        Upload a single code file from an execution.

        Args:
            task: The task that was executed.
            file_written: The file created during execution.
            files_written: Compatibility alias for multiple files. If provided,
                uploads each file sequentially and returns an aggregated result.
            succeeded: Whether the task completed successfully.
            use_raysurfer_ai_voting: Let Raysurfer AI vote on stored blocks (default True).
                Ignored when user_vote is provided.
            auto_vote: Compatibility alias for use_raysurfer_ai_voting.
            user_vote: User-provided vote (1 for thumbs up, -1 for thumbs down).
                When provided, AI voting is automatically skipped.
            execution_logs: Captured stdout/stderr for vote context.
            run_url: URL to the finished run (e.g. logs page, CI run, LangSmith trace).
            workspace_id: Override client-level workspace_id for this request.
            dependencies: Package dependencies with versions (e.g., {"pandas": "2.1.0"}).
            public: Upload to the public community namespace (default False).
            vote_source: Origin of the vote (e.g. "cli", "mcp", "sdk").
            vote_count: Number of votes to apply (default 1).
        """
        if file_written is not None and files_written is not None:
            raise ValueError("Provide either file_written or files_written, not both.")

        if auto_vote is not None:
            use_raysurfer_ai_voting = auto_vote

        if files_written is not None:
            if len(files_written) == 0:
                raise ValueError("files_written must contain at least one file.")

            if len(files_written) == 1:
                file_written = files_written[0]
            else:
                responses: list[SubmitExecutionResultResponse] = []
                for file in files_written:
                    response = self.upload_new_code_snip(
                        task=task,
                        file_written=file,
                        succeeded=succeeded,
                        use_raysurfer_ai_voting=use_raysurfer_ai_voting,
                        user_vote=user_vote,
                        execution_logs=execution_logs,
                        run_url=run_url,
                        workspace_id=workspace_id,
                        dependencies=dependencies,
                        public=public,
                        vote_source=vote_source,
                        vote_count=vote_count,
                    )
                    responses.append(response)

                return SubmitExecutionResultResponse(
                    success=all(response.success for response in responses),
                    code_blocks_stored=sum(response.code_blocks_stored for response in responses),
                    message=f"Uploaded {len(files_written)} files via compatibility path.",
                )

        if file_written is None:
            raise ValueError("Missing required file input: provide file_written or files_written.")

        data: JsonDict = {
            "task": task,
            "file_written": file_written.model_dump(),
            "succeeded": succeeded,
            "use_raysurfer_ai_voting": use_raysurfer_ai_voting,
        }
        if user_vote is not None:
            data["user_vote"] = user_vote
        if execution_logs is not None:
            data["execution_logs"] = execution_logs
        if run_url is not None:
            data["run_url"] = run_url
        if dependencies is not None:
            data["dependencies"] = dependencies
        if public:
            data["public"] = True
        if vote_source is not None:
            data["vote_source"] = vote_source
        if vote_count != 1:
            data["vote_count"] = vote_count
        result = self._request(
            "POST", "/api/store/execution-result", headers_override=self._workspace_headers(workspace_id), json=data
        )
        return SubmitExecutionResultResponse(**result)

    # Backwards-compatible alias
    upload_new_code_snips = upload_new_code_snip

    def upload_bulk_code_snips(
        self,
        prompts: list[str],
        files_written: list[FileWritten],
        log_files: list[LogFile] | None = None,
        use_raysurfer_ai_voting: bool = True,
        user_votes: dict[str, int] | None = None,
        workspace_id: str | None = None,
        vote_source: str | None = None,
        vote_count: int = 1,
    ) -> BulkExecutionResultResponse:
        """
        Bulk upload prompts, logs, and code files for sandboxed grading.

        Args:
            prompts: Ordered list of raw user prompts.
            files_written: Code files to store and grade.
            log_files: Log files (any format; use encoding="base64" for binary).
            use_raysurfer_ai_voting: Let Raysurfer AI vote on stored blocks (default True).
                Ignored when user_votes is provided.
            user_votes: Dict of filename to vote (1 for thumbs up, -1 for thumbs down).
                When provided, AI voting is automatically skipped.
            workspace_id: Override client-level workspace_id for this request.
            vote_source: Origin of the vote (e.g. "cli", "mcp", "sdk").
            vote_count: Number of votes to apply (default 1).
        """
        data: JsonDict = {
            "prompts": prompts,
            "files_written": [f.model_dump() for f in files_written],
            "use_raysurfer_ai_voting": use_raysurfer_ai_voting,
        }
        if log_files is not None:
            data["log_files"] = [f.model_dump() for f in log_files]
        if user_votes is not None:
            data["user_votes"] = user_votes
        if vote_source is not None:
            data["vote_source"] = vote_source
        if vote_count != 1:
            data["vote_count"] = vote_count

        result = self._request(
            "POST",
            "/api/store/bulk-execution-result",
            headers_override=self._workspace_headers(workspace_id),
            json=data,
        )
        return BulkExecutionResultResponse(**result)

    # =========================================================================
    # Retrieve API
    # =========================================================================

    def search(
        self,
        task: str,
        top_k: int = 5,
        min_verdict_score: float = 0.3,
        min_human_upvotes: int = 0,
        prefer_complete: bool = False,
        input_schema: JsonDict | None = None,
        workspace_id: str | None = None,
    ) -> SearchResponse:
        """Unified search for cached code snippets.

        Args:
            task: The task to search for.
            top_k: Maximum number of results to return.
            min_verdict_score: Minimum verdict score threshold.
            min_human_upvotes: Minimum number of human upvotes required.
            prefer_complete: Prefer complete code blocks.
            input_schema: Optional input schema for filtering.
            workspace_id: Override client-level workspace_id for this request.
        """
        data = {
            "task": task,
            "top_k": top_k,
            "min_verdict_score": min_verdict_score,
            "min_human_upvotes": min_human_upvotes,
            "prefer_complete": prefer_complete,
            "input_schema": input_schema,
        }
        result = self._request(
            "POST", "/api/retrieve/search", headers_override=self._workspace_headers(workspace_id), json=data
        )
        matches = [
            SearchMatch(
                code_block=CodeBlock(**m["code_block"]),
                score=m["score"],
                vector_score=m.get("vector_score"),
                verdict_score=m.get("verdict_score"),
                thumbs_up=m["thumbs_up"],
                thumbs_down=m["thumbs_down"],
                filename=m["filename"],
                language=m["language"],
                entrypoint=m["entrypoint"],
                dependencies=m.get("dependencies", {}),
            )
            for m in result["matches"]
        ]
        return SearchResponse(
            matches=matches,
            total_found=result["total_found"],
            cache_hit=result.get("cache_hit", False),
        )

    def get_code_snips(
        self,
        task: str,
        top_k: int = 10,
        min_verdict_score: float = 0.0,
    ) -> RetrieveCodeBlockResponse:
        """Get cached code snippets -- delegates to search()."""
        response = self.search(task=task, top_k=top_k, min_verdict_score=min_verdict_score)
        code_blocks = [
            CodeBlockMatch(
                code_block=m.code_block,
                score=m.score,
                thumbs_up=m.thumbs_up,
                thumbs_down=m.thumbs_down,
            )
            for m in response.matches
        ]
        return RetrieveCodeBlockResponse(code_blocks=code_blocks, total_found=response.total_found)

    def retrieve_best(
        self,
        task: str,
        top_k: int = 10,
        min_verdict_score: float = 0.0,
    ) -> RetrieveBestResponse:
        """Get the best code block -- delegates to search()."""
        response = self.search(task=task, top_k=top_k, min_verdict_score=min_verdict_score)
        best_match = None
        if response.matches:
            m = response.matches[0]
            best_match = BestMatch(
                code_block=m.code_block,
                score=m.score,
                thumbs_up=m.thumbs_up,
                thumbs_down=m.thumbs_down,
            )
        alternatives = [
            AlternativeCandidate(
                code_block_id=m.code_block.id,
                name=m.code_block.name,
                score=m.score,
                reason=f"{m.thumbs_up} thumbs up, {m.thumbs_down} thumbs down"
                if m.thumbs_up > 0
                else "No execution history",
            )
            for m in response.matches[1:4]
        ]
        return RetrieveBestResponse(
            best_match=best_match,
            alternative_candidates=alternatives,
            retrieval_confidence=str(round(response.matches[0].score, 4)) if response.matches else "0",
        )

    def get_few_shot_examples(
        self,
        task: str,
        k: int = 3,
    ) -> list[FewShotExample]:
        """Retrieve few-shot examples for code generation"""
        data = {"task": task, "k": k}
        result = self._request("POST", "/api/retrieve/few-shot-examples", json=data)
        return [FewShotExample(**ex) for ex in result["examples"]]

    def get_task_patterns(
        self,
        task: str | None = None,
        code_block_id: str | None = None,
        min_thumbs_up: int = 0,
        top_k: int = 20,
    ) -> list[TaskPattern]:
        """Retrieve proven task->code mappings"""
        data = {
            "task": task,
            "code_block_id": code_block_id,
            "min_thumbs_up": min_thumbs_up,
            "top_k": top_k,
        }
        result = self._request("POST", "/api/retrieve/task-patterns", json=data)
        return [TaskPattern(**p) for p in result["patterns"]]

    def get_code_files(
        self,
        task: str,
        top_k: int = 5,
        min_verdict_score: float = 0.3,
        prefer_complete: bool = True,
        cache_dir: str = ".raysurfer_code",
    ) -> GetCodeFilesResponse:
        """Get code files -- delegates to search()."""
        response = self.search(
            task=task, top_k=top_k, min_verdict_score=min_verdict_score, prefer_complete=prefer_complete
        )
        files = [
            CodeFile(
                code_block_id=m.code_block.id,
                filename=m.filename,
                source=m.code_block.source,
                entrypoint=m.entrypoint,
                description=m.code_block.description,
                input_schema=m.code_block.input_schema,
                output_schema=m.code_block.output_schema,
                language=m.language,
                dependencies=m.dependencies,
                score=m.score,
                thumbs_up=m.thumbs_up,
                thumbs_down=m.thumbs_down,
            )
            for m in response.matches
        ]
        add_to_llm_prompt = self._format_llm_prompt(files, cache_dir)
        return GetCodeFilesResponse(
            files=files, task=task, total_found=response.total_found, add_to_llm_prompt=add_to_llm_prompt
        )

    def _format_llm_prompt(self, files: list[CodeFile], cache_dir: str | None = None) -> str:
        """Format a prompt string listing all retrieved code files."""
        if not files:
            return ""

        lines = [
            "\n\n## IMPORTANT: Pre-validated Code Files Available\n",
            "The following validated code has been retrieved from the cache. "
            "Use these files directly instead of regenerating code.\n",
        ]

        for f in files:
            if cache_dir:
                import os

                full_path = os.path.join(cache_dir, f.filename)
                lines.append(f"\n### `{f.filename}` -> `{full_path}`")
            else:
                lines.append(f"\n### `{f.filename}`")
            lines.append(f"- **Description**: {f.description}")
            lines.append(f"- **Language**: {f.language}")
            lines.append(f"- **Entrypoint**: `{f.entrypoint}`")
            lines.append(f"- **Confidence**: {f.score:.0%}")
            if f.dependencies:
                deps = [f"{k}@{v}" for k, v in f.dependencies.items()]
                lines.append(f"- **Dependencies**: {', '.join(deps)}")

        lines.append("\n\n**Instructions**:")
        lines.append("1. Read the cached file(s) before writing new code")
        lines.append("2. Use the cached code as your starting point")
        lines.append("3. Only modify if the task requires specific changes")
        lines.append("4. Do not regenerate code that already exists\n")

        return "\n".join(lines)

    def vote_code_snip(
        self,
        task: str,
        code_block_id: str,
        code_block_name: str,
        code_block_description: str,
        succeeded: bool,
    ) -> JsonDict:
        """
        Vote on whether a cached code snippet was useful.

        This triggers background voting to assess whether the cached code
        actually helped complete the task successfully.
        """
        data = {
            "task": task,
            "code_block_id": code_block_id,
            "code_block_name": code_block_name,
            "code_block_description": code_block_description,
            "succeeded": succeeded,
        }
        return self._request("POST", "/api/store/cache-usage", json=data)

    def comment_on_code_snip(self, code_block_id: str, text: str) -> JsonDict:
        """Add a comment to a cached code snippet."""
        return self._request("POST", "/api/store/comment", json={
            "code_block_id": code_block_id, "text": text,
        })

    # =========================================================================
    # Auto Review API
    # =========================================================================

    def auto_review(
        self,
        execution_id: str,
        triggering_task: str,
        execution_state: ExecutionState,
        input_data: JsonDict,
        output_data: JsonValue,
        code_block_name: str,
        code_block_description: str,
        error_message: str | None = None,
    ) -> AutoReviewResponse:
        """
        Get an auto-generated review using Claude Opus 4.6.
        Useful for programmatically reviewing execution results.
        """
        data = {
            "execution_id": execution_id,
            "triggering_task": triggering_task,
            "execution_state": execution_state.value,
            "input_data": input_data,
            "output_data": output_data,
            "code_block_name": code_block_name,
            "code_block_description": code_block_description,
            "error_message": error_message,
        }
        result = self._request("POST", "/api/store/auto-review", json=data)
        return AutoReviewResponse(
            success=result["success"],
            execution_id=result["execution_id"],
            review=AgentReview(**result["review"]),
            message=result["message"],
        )

    def get_executions(
        self,
        code_block_id: str | None = None,
        task: str | None = None,
        verdict: AgentVerdict | None = None,
        limit: int = 20,
    ) -> RetrieveExecutionsResponse:
        """Retrieve execution records by code block ID, task, or verdict."""
        data = {
            "code_block_id": code_block_id,
            "task": task,
            "verdict": verdict.value if verdict else None,
            "limit": limit,
        }
        result = self._request("POST", "/api/retrieve/executions", json=data)
        executions = [ExecutionRecord(**ex) for ex in result["executions"]]
        return RetrieveExecutionsResponse(
            executions=executions,
            total_found=result["total_found"],
        )

    # =========================================================================
    # Execute API (tool calling)
    # =========================================================================

    def tool(self, fn: Callable[..., JsonValue]) -> Callable[..., JsonValue]:
        """Register a function as a tool for execute(). Delegates to async client."""
        return self._async_inner.tool(fn)

    def execute(
        self,
        task: str,
        user_code: str | None = None,
        timeout: int = 300,
        codegen_api_key: str | None = None,
        codegen_prompt: str | None = None,
        codegen_model: str = "claude-opus-4-6",
    ) -> ExecuteResult:
        """Execute a task with registered tools in a sandbox."""
        return asyncio.get_event_loop().run_until_complete(
            self._async_inner.execute(
                task=task,
                user_code=user_code,
                timeout=timeout,
                codegen_api_key=codegen_api_key,
                codegen_prompt=codegen_prompt,
                codegen_model=codegen_model,
            )
        )

    def run_script(
        self,
        s3_key: str,
        params: dict[str, str] | None = None,
        timeout: int = 300,
    ) -> ExecuteResult:
        """Execute an S3-stored script in a remote sandbox with tool callbacks."""
        return asyncio.get_event_loop().run_until_complete(
            self._async_inner.run_script(s3_key=s3_key, params=params, timeout=timeout)
        )

    def execute_generated_code(
        self,
        task: str,
        user_code: str,
        timeout: int = 300,
    ) -> ExecuteResult:
        """Execute client-generated Python code in the remote sandbox with tool callbacks."""
        return asyncio.get_event_loop().run_until_complete(
            self._async_inner.execute_generated_code(task=task, user_code=user_code, timeout=timeout)
        )

    def execute_with_sandbox_codegen(
        self,
        task: str,
        codegen_api_key: str,
        codegen_prompt: str,
        timeout: int = 300,
        codegen_model: str = "claude-opus-4-6",
    ) -> ExecuteResult:
        """Generate Python code inside the sandbox, then execute it with tool callbacks."""
        return asyncio.get_event_loop().run_until_complete(
            self._async_inner.execute_with_sandbox_codegen(
                task=task,
                codegen_api_key=codegen_api_key,
                codegen_prompt=codegen_prompt,
                timeout=timeout,
                codegen_model=codegen_model,
            )
        )

    # =========================================================================
    # Public Snippet Browsing (no API key required)
    # =========================================================================

    def browse_public(
        self,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "upvoted",
        language: str | None = None,
    ) -> BrowsePublicResponse:
        """Browse community public snippets without authentication.

        Args:
            limit: Maximum number of snippets to return.
            offset: Pagination offset.
            sort_by: Sort order - "upvoted" or "recent".
            language: Filter by programming language.
        """
        data: JsonDict = {
            "limit": limit,
            "offset": offset,
            "sort_by": sort_by,
        }
        if language:
            data["language"] = language
        result = self._request("POST", "/api/snippets/public/list", json=data)
        snippets = [PublicSnippet(**s) for s in result["snippets"]]
        return BrowsePublicResponse(
            snippets=snippets,
            total=result["total"],
            has_more=result.get("has_more", False),
        )

    def search_public(
        self,
        query: str,
        limit: int = 20,
        language: str | None = None,
    ) -> SearchPublicResponse:
        """Search community public snippets by keyword without authentication.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            language: Filter by programming language.
        """
        data: JsonDict = {
            "query": query,
            "limit": limit,
        }
        if language:
            data["language"] = language
        result = self._request("POST", "/api/snippets/public/search", json=data)
        snippets = [PublicSnippet(**s) for s in result["snippets"]]
        return SearchPublicResponse(
            snippets=snippets,
            total=result["total"],
            query=result["query"],
        )
