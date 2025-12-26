import asyncio
import logging
import random
import shutil
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, Literal

import aiohttp
from pydantic import BaseModel,ConfigDict
from typing_extensions import Self

from swerex.exceptions import SwerexException
from swerex.runtime.abstract import (
    AbstractRuntime,
    Action,
    CloseResponse,
    CloseSessionRequest,
    CloseSessionResponse,
    Command,
    CommandResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    IsAliveResponse,
    Observation,
    ReadFileRequest,
    ReadFileResponse,
    UploadRequest,
    UploadResponse,
    WriteFileRequest,
    WriteFileResponse,
    _ExceptionTransfer,
)
from swerex.utils.log import get_logger
from swerex.utils.wait import _wait_until_alive


class RemoteRuntimeConfig(BaseModel):
    auth_token: str
    """The token to use for authentication."""
    host: str = "http://127.0.0.1"
    """The host to connect to."""
    port: int | None = None
    """The port to connect to."""
    timeout: float = 0.15
    """The timeout for the runtime."""
    base_url: str | None = None
    """The base URL for remote runtime connection (for VEFAAS)."""
    extra_params: dict[str, Any] | None = None
    """Extra parameters for remote runtime connection (for VEFAAS)."""

    type: Literal["remote"] = "remote"
    """Discriminator for (de)serialization/CLI. Do not change."""

    model_config = ConfigDict(extra="forbid")

    def get_runtime(self) -> AbstractRuntime:
        from swerex.runtime.remote import RemoteRuntime

        return RemoteRuntime.from_config(self)



class RemoteRuntime(AbstractRuntime):
    def __init__(
            self,
            *,
            logger: logging.Logger | None = None,
            **kwargs: Any,
    ):
        """A runtime that connects to a remote server.

        Args:
            **kwargs: Keyword arguments to pass to the `RemoteRuntimeConfig` constructor.
        """
        self._config = RemoteRuntimeConfig(**kwargs)
        self.logger = logger or get_logger("rex-runtime")
        if not self._config.host.startswith("http"):
            self.logger.warning("Host %s does not start with http, adding http://", self._config.host)
            self._config.host = f"http://{self._config.host}"

    @classmethod
    def from_config(cls, config: RemoteRuntimeConfig) -> Self:
        return cls(**config.model_dump())

    def _get_timeout(self, timeout: float | None = None) -> float:
        if timeout is None:
            return self._config.timeout
        return timeout

    @property
    def _headers(self) -> dict[str, str]:
        """Request headers to use for authentication."""
        headers = {}
        if self._config.auth_token:
            headers["X-API-Key"] = self._config.auth_token
        if hasattr(self._config, 'extra_params') and self._config.extra_params:
            if "faasInstanceName" in self._config.extra_params:
                headers["X-Faas-Instance-Name"] = str(self._config.extra_params["faasInstanceName"])
        return headers

    @property
    def _api_url(self) -> str:
        # Prioritize base_url if provided (for VEFAAS deployments)
        if hasattr(self._config, 'base_url') and self._config.base_url:
            return self._config.base_url
        # Fall back to host/port combination for direct connections
        if self._config.port is None:
            return self._config.host
        return f"{self._config.host}:{self._config.port}"

    def _handle_transfer_exception(self, exc_transfer: _ExceptionTransfer) -> None:
        """Reraise exceptions that were thrown on the remote."""
        if exc_transfer.traceback:
            self.logger.critical("Traceback: \n%s", exc_transfer.traceback)
        module, _, exc_name = exc_transfer.class_path.rpartition(".")
        print(module, exc_name)
        if module == "builtins":
            module_obj = __builtins__
        else:
            if module not in sys.modules:
                self.logger.debug("Module %s not in sys.modules, trying to import it", module)
                try:
                    __import__(module)
                except ImportError:
                    self.logger.debug("Failed to import module %s", module)
                    exc = SwerexException(exc_transfer.message)
                    raise exc from None
            module_obj = sys.modules[module]
        try:
            if isinstance(module_obj, dict):
                # __builtins__, sometimes
                exception = module_obj[exc_name](exc_transfer.message)
            else:
                exception = getattr(module_obj, exc_name)(exc_transfer.message)
        except (AttributeError, TypeError):
            self.logger.error(
                f"Could not initialize transferred exception: {exc_transfer.class_path!r}. "
                f"Transfer object: {exc_transfer}"
            )
            exception = SwerexException(exc_transfer.message)
        exception.extra_info = exc_transfer.extra_info
        raise exception from None

    async def _handle_response_errors(self, response: aiohttp.ClientResponse) -> None:
        """Raise exceptions found in the request response."""
        if response.status == 511:
            data = await response.json()
            exc_transfer = _ExceptionTransfer(**data["swerexception"])
            self._handle_transfer_exception(exc_transfer)
        if response.status >= 400:
            data = await response.json()
            self.logger.critical("Received error response: %s", data)
            response.raise_for_status()

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        """Checks if the runtime is alive.

        Internal server errors are thrown, everything else just has us return False
        together with the message.
        """
        try:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True), proxy="http://sys-proxy-rd-relay.byted.org:8118") as session:
                timeout_value = self._get_timeout(timeout)
                async with session.get(
                        f"{self._api_url}/is_alive",
                        headers=self._headers,
                        timeout=aiohttp.ClientTimeout(total=timeout_value),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return IsAliveResponse(**data)
                    elif response.status == 511:
                        data = await response.json()
                        exc_transfer = _ExceptionTransfer(**data["swerexception"])
                        self._handle_transfer_exception(exc_transfer)

                    data = await response.json()
                    msg = f"Status code {response.status} from {self._api_url}/is_alive. Message: {data.get('detail')}"
                    return IsAliveResponse(is_alive=False, message=msg)
        except aiohttp.ClientError:
            msg = f"Failed to connect to {self._config.host}\n"
            msg += traceback.format_exc()
            return IsAliveResponse(is_alive=False, message=msg)
        except Exception:
            msg = f"Failed to connect to {self._config.host}\n"
            msg += traceback.format_exc()
            return IsAliveResponse(is_alive=False, message=msg)

    async def wait_until_alive(self, *, timeout: float = 60.0):
        return await _wait_until_alive(self.is_alive, timeout=timeout)

    async def _request(self, endpoint: str, payload: BaseModel | None, output_class: Any, num_retries: int = 0):
        """Small helper to make requests to the server and handle errors and output."""
        request_url = f"{self._api_url}/{endpoint}"
        request_id = str(uuid.uuid4())
        headers = self._headers.copy()
        headers["X-Request-ID"] = request_id  # idempotency key for the request

        retry_count = 0
        last_exception: Exception | None = None
        retry_delay = 0.1
        backoff_max = 5

        while retry_count <= num_retries:
            try:
                # Use fixed proxy
                proxy = "http://sys-proxy-rd-relay.byted.org:8118"
                connector = aiohttp.TCPConnector(force_close=True)
                session_kwargs = {'connector': connector}
                session_kwargs['proxy'] = proxy
                async with aiohttp.ClientSession(**session_kwargs) as session:
                    async with session.post(
                            request_url,
                            json=payload.model_dump() if payload else None,
                            headers=headers,
                    ) as resp:
                        await self._handle_response_errors(resp)
                        return output_class(**await resp.json())
            except Exception as e:
                last_exception = e
                retry_count += 1
                if retry_count <= num_retries:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    retry_delay += random.uniform(0, 0.5)
                    retry_delay = min(retry_delay, backoff_max)
                    continue
                self.logger.error("Error making request %s after %d retries: %s", request_id, num_retries, e)
        raise last_exception  # type: ignore

    async def create_session(self, request: CreateSessionRequest) -> CreateSessionResponse:
        """Creates a new session."""
        return await self._request("create_session", request, CreateSessionResponse)

    async def run_in_session(self, action: Action) -> Observation:
        """Runs a command in a session."""
        return await self._request("run_in_session", action, Observation)

    async def close_session(self, request: CloseSessionRequest) -> CloseSessionResponse:
        """Closes a shell session."""
        return await self._request("close_session", request, CloseSessionResponse)

    async def execute(self, command: Command) -> CommandResponse:
        """Executes a command (independent of any shell session)."""
        return await self._request("execute", command, CommandResponse)

    async def read_file(self, request: ReadFileRequest) -> ReadFileResponse:
        """Reads a file"""
        return await self._request("read_file", request, ReadFileResponse)

    async def write_file(self, request: WriteFileRequest) -> WriteFileResponse:
        """Writes a file"""
        return await self._request("write_file", request, WriteFileResponse)

    async def upload(self, request: UploadRequest) -> UploadResponse:
        """Uploads a file"""
        source = Path(request.source_path).resolve()
        self.logger.debug("Uploading file from %s to %s", request.source_path, request.target_path)

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as session:
            if source.is_dir():
                # Ignore cleanup errors: See https://github.com/SWE-agent/SWE-agent/issues/1005
                with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
                    zip_path = Path(temp_dir) / "zipped_transfer.zip"
                    shutil.make_archive(str(zip_path.with_suffix("")), "zip", source)
                    self.logger.debug("Created zip file at %s", zip_path)

                    with open(zip_path, "rb") as f:
                        data = aiohttp.FormData()
                        data.add_field("file", f, filename=zip_path.name, content_type="application/zip")
                        data.add_field("target_path", request.target_path)
                        data.add_field("unzip", "true")

                        async with session.post(
                                f"{self._api_url}/upload", data=data, headers=self._headers
                        ) as response:
                            await self._handle_response_errors(response)
                            return UploadResponse(**(await response.json()))
            elif source.is_file():
                self.logger.debug("Uploading file from %s to %s", source, request.target_path)

                with open(source, "rb") as f:
                    data = aiohttp.FormData()
                    data.add_field("file", f, filename=source.name)
                    data.add_field("target_path", request.target_path)
                    data.add_field("unzip", "false")

                    async with session.post(f"{self._api_url}/upload", data=data, headers=self._headers) as response:
                        await self._handle_response_errors(response)
                        return UploadResponse(**(await response.json()))
            else:
                msg = f"Source path {source} is not a file or directory"
                raise ValueError(msg)

    async def close(self) -> CloseResponse:
        """Closes the runtime."""
        return await self._request("close", None, CloseResponse)
