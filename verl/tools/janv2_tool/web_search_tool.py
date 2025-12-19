import json
import logging
import os
import re
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor
import requests

from verl.utils.rollout_trace import rollout_trace_op

# CHANGED: Import from parent package (verl.tools) instead of relative
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class PoolMode(Enum):
    """Execution pool mode enumeration."""
    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count


class WebSearchExecutionWorker:
    """Worker for executing web search operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        return TokenBucketWorker.options(name="websearch-rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing web search: {e}")
                    return f"Error: {str(e)}", {"error": str(e)}
        else:
            return fn(*fn_args, **fn_kwargs)


def init_websearch_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize web search execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(WebSearchExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


def _create_preview(text: str, max_sentences: int = 3, max_chars: int = 300) -> str:
    """Create a preview from text (first few sentences)."""
    if not text:
        return "No preview available"
    
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    preview_sentences = sentences[:max_sentences]
    preview = ' '.join(preview_sentences)
    
    if len(preview) > max_chars:
        preview = preview[:max_chars].rsplit(' ', 1)[0] + '...'
    
    return preview


class WebSearchTool(BaseTool):
    """Web search tool for retrieving information from RAG server."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # RAG server configuration
        self.rag_server_url = config.get("rag_server_url", os.environ.get("RAG_SERVER_URL", "http://localhost:2223"))
        
        # Search configuration
        self.num_results = config.get("num_results", 10)
        self.topk_retrieval = config.get("topk_retrieval", 30)
        
        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 100)
        self.rate_limit = config.get("rate_limit", 100)
        self.timeout = config.get("timeout", 600)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_websearch_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        logger.info(f"Initialized WebSearchTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "queries": [],
            "results": [],
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    def perform_search(
        self, 
        instance_id: str, 
        query: str, 
        server_url: str, 
        num_results: int,
        topk_retrieval: int,
        timeout: int
    ) -> tuple[str, dict]:
        """Perform web search using RAG server."""
        metadata = {
            "query": query,
            "num_results_requested": num_results,
            "num_results_returned": 0,
            "status": "unknown",
            "error": None,
        }

        try:
            payload = {
                "queries": [query],
                "topk_retrieval": max(num_results * 3, topk_retrieval),
                "topk_rerank": num_results,
                "return_scores": False
            }
            
            response = requests.post(
                f"{server_url}/retrieve",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            
            metadata["status_code"] = response.status_code
            
            if response.status_code != 200:
                error_msg = f"Error: RAG server returned status {response.status_code}: {response.text}"
                metadata["error"] = error_msg
                metadata["status"] = "error"
                return error_msg, metadata
            
            result = response.json()
            documents = result.get('result', [[]])[0]
            
            if not documents:
                metadata["status"] = "no_results"
                return "No results found", metadata
            
            # Format results with URLs and previews
            formatted_results = []
            for i, doc in enumerate(documents, 1):
                title = doc.get('title', f'Document {i}')
                text = doc.get('text', '').strip()
                url = f"doc_{doc.get('doc_id')}"
                preview = _create_preview(text)
                
                formatted_results.append(
                    f"Result {i}:\n"
                    f"Title: {title}\n"
                    f"URL: {url}\n"
                    f"Preview: {preview}\n"
                )
            
            result_text = "\n".join(formatted_results)
            metadata["num_results_returned"] = len(documents)
            metadata["status"] = "success"
            
            return result_text, metadata

        except requests.exceptions.ConnectionError:
            error_msg = "Error: Could not connect to RAG server."
            metadata["error"] = error_msg
            metadata["status"] = "connection_error"
            return error_msg, metadata
        except requests.exceptions.Timeout:
            error_msg = "Error: Request to RAG server timed out."
            metadata["error"] = error_msg
            metadata["status"] = "timeout"
            return error_msg, metadata
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            metadata["error"] = error_msg
            metadata["status"] = "error"
            return error_msg, metadata

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query = parameters.get("query")

        if not query or not isinstance(query, str):
            error_msg = "Error: 'query' is missing or not a string in parameters."
            logger.error(f"[WebSearchTool] {error_msg}")
            return ToolResponse(text=error_msg), 0.0, {}

        try:
            result_text, metadata = await self.execution_pool.execute.remote(
                self.perform_search, 
                instance_id, 
                query, 
                self.rag_server_url, 
                self.num_results,
                self.topk_retrieval,
                self.timeout
            )

            self._instance_dict[instance_id]["queries"].append(query)
            self._instance_dict[instance_id]["results"].append(result_text)

            metrics = {
                "query": metadata.get("query"),
                "num_results_returned": metadata.get("num_results_returned", 0),
                "status": metadata.get("status"),
                "error": metadata.get("error"),
            }

            return ToolResponse(text=result_text), 0.0, metrics

        except Exception as e:
            error_result = f"Search execution failed: {e}"
            logger.error(f"[WebSearchTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]