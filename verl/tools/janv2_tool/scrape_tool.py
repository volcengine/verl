import json
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor
import requests

from verl.utils.rollout_trace import rollout_trace_op

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


class VisitExecutionWorker:
    """Worker for executing visit operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        return TokenBucketWorker.options(name="visit-rate-limiter", get_if_exists=True).remote(rate_limit)

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
                    logger.warning(f"Error when executing visit: {e}")
                    return f"Error: {str(e)}", {"error": str(e)}
        else:
            return fn(*fn_args, **fn_kwargs)


def init_visit_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize visit execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(VisitExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class ScrapeTool(BaseTool):
    """Tool for visiting URLs and retrieving document content from RAG server."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # RAG server configuration
        self.rag_server_url = config.get("rag_server_url", os.environ.get("RAG_SERVER_URL", "http://localhost:2223"))
        
        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 50)
        self.rate_limit = config.get("rate_limit", 50)
        self.timeout = config.get("timeout", 600)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_visit_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        logger.info(f"Initialized VisitTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "visited_urls": [],
            "responses": [],
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    def visit_url(self, instance_id: str, url: str, server_url: str, timeout: int) -> tuple[str, dict]:
        """Visit a URL and retrieve its content from the RAG server."""
        metadata = {
            "url": url,
            "status_code": None,
            "error": None,
        }

        try:
            payload = {"url": url}
            
            response = requests.post(
                f"{server_url}/visit",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            
            metadata["status_code"] = response.status_code
            
            if response.status_code != 200:
                error_msg = f"Error: Could not visit {url}. Server returned status {response.status_code}"
                metadata["error"] = error_msg
                return error_msg, metadata
            
            result = response.json()
            documents = result.get('result', [[]])[0]
            
            if not documents:
                error_msg = f"Error: Could not find content for {url}"
                metadata["error"] = error_msg
                return error_msg, metadata
            
            doc = documents[0]
            title = doc.get('title', 'Untitled')
            content = doc.get('text', '').strip()
            
            result_text = f"Title: {title}\nURL: {url}\n\nFull Content:\n{content}"
            metadata["title"] = title
            metadata["content_length"] = len(content)
            
            return result_text, metadata

        except requests.exceptions.ConnectionError:
            error_msg = "Error: Could not connect to RAG server."
            metadata["error"] = error_msg
            return error_msg, metadata
        except requests.exceptions.Timeout:
            error_msg = "Error: Request to RAG server timed out."
            metadata["error"] = error_msg
            return error_msg, metadata
        except Exception as e:
            error_msg = f"Error visiting {url}: {str(e)}"
            metadata["error"] = error_msg
            return error_msg, metadata

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        url = parameters.get("url")

        if not url or not isinstance(url, str):
            error_msg = "Error: 'url' is missing or not a string in parameters."
            logger.error(f"[VisitTool] {error_msg}")
            return ToolResponse(text=error_msg), 0.0, {}

        try:
            result_text, metadata = await self.execution_pool.execute.remote(
                self.visit_url, instance_id, url, self.rag_server_url, self.timeout
            )

            self._instance_dict[instance_id]["visited_urls"].append(url)
            self._instance_dict[instance_id]["responses"].append(result_text)

            metrics = {
                "url": metadata.get("url"),
                "status_code": metadata.get("status_code"),
                "content_length": metadata.get("content_length", 0),
                "error": metadata.get("error"),
            }

            return ToolResponse(text=result_text), 0.0, metrics

        except Exception as e:
            error_result = f"Visit execution failed: {e}"
            logger.error(f"[VisitTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]