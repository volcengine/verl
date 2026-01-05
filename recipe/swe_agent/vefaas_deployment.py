import logging
import time
import uuid
import asyncio,os

from typing import Any,Literal
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, model_validator

from swerex import PACKAGE_NAME, REMOTE_EXECUTABLE_NAME
from swerex.exceptions import DeploymentNotStartedError

from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import DeploymentHook, CombinedDeploymentHook
from swerex.runtime.abstract import IsAliveResponse
# from swerex.runtime.remote import RemoteRuntime
# from swerex.runtime.config import RemoteRuntimeConfig

from recipe.swe_agent.remote_runtime import RemoteRuntime,RemoteRuntimeConfig
from swerex.utils.log import get_logger
from swerex.utils.wait import _wait_until_alive

from dotenv import load_dotenv
import volcenginesdkcore
import volcenginesdkvefaas


from volcenginesdkcore.rest import ApiException

class VefaasDeploymentConfig(BaseModel):
    """Configuration for VEFAAS deployment."""

    image: str | None = None
    """Docker image to use for the sandbox. If not provided, uses random from image_list_file."""
    command: str = "python3 -m swerex.server --auth-token {token}"
    """Command to run in the sandbox with authentication token."""
    timeout: float = 60.0
    """Timeout for runtime operations."""
    startup_timeout: float = 120.0
    """Timeout waiting for runtime to start."""
    function_id: str | None = None
    """VEFAAS function ID."""
    function_route: str | None = None
    """VEFAAS function Route."""
    instance_id: str | None = None
    """SWE-Bench instance ID."""

    type: Literal["vefaas"] = "vefaas"
    """Discriminator for (de)serialization/CLI. Do not change."""
    model_config = ConfigDict(extra="forbid")

    def get_deployment(self) -> AbstractDeployment:
        return VefaasDeployment.from_config(self)


class VefaasDeployment(AbstractDeployment):
    def __init__(self, *, logger: logging.Logger | None = None, **kwargs: Any):
        load_dotenv()
        self._config = VefaasDeploymentConfig(**kwargs)
        self._runtime: RemoteRuntime | None = None
        self.logger = logger or get_logger("rex-deploy-vefaas")
        self._hooks = CombinedDeploymentHook()
        self._sandbox_id: str | None = None
        self._stopped: bool = False

        access_key = os.getenv("VOLCE_ACCESS_KEY") or os.getenv("VOLCENGINE_ACCESS_KEY")
        secret_key = os.getenv("VOLCE_SECRET_KEY") or os.getenv("VOLCENGINE_SECRET_KEY")
        region = os.getenv("VEFAAS_REGION", "cn-beijing")
        if not all([access_key, secret_key, region]):
            raise ValueError(
                "VOLCE_ACCESS_KEY, VOLCE_SECRET_KEY, and VEFAAS_REGION must be set"
            )
        self._vefaas_client = get_vefaas_client(access_key, secret_key, region)

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: VefaasDeploymentConfig) -> Self:
        return cls(**config.model_dump())

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        if self._runtime is None:
            raise DeploymentNotStartedError("Runtime not started")
        return await self._runtime.is_alive(timeout=timeout)

    async def _wait_until_alive(self, timeout: float = 10.0):
        try:
            return await _wait_until_alive(
                self.is_alive, timeout=timeout, function_timeout=0.5
            )
        except TimeoutError as e:
            self.logger.error("Runtime did not start within timeout.")
            await self.stop()
            raise e

    def _get_token(self) -> str:
        return str(uuid.uuid4())

    async def start(self):
        self.logger.info("Starting vefaas deployment")
        print(f"Starting vefaas deployment, function_id: {self._config.function_id}, instance_id: {self._config.instance_id}")
        function_id = self._config.function_id or os.getenv("VEFAAS_FUNCTION_ID")
        if not function_id:
            raise ValueError("VEFAAS_FUNCTION_ID environment variable not set")

        image = self._config.image

        if not image:
            raise ValueError("No image specified and no image list provided")

        token = self._get_token()
        command = self._config.command.format(token=token)

        self.logger.info(f"Creating sandbox with image {image}")
        print(f"Creating sandbox with image {image}, function_id: {function_id}, instance_id: {self._config.instance_id}")
        self._hooks.on_custom_step("Creating vefaas sandbox")
        loop = asyncio.get_running_loop()
        self._sandbox_id = await loop.run_in_executor(
            None,
            create_sandbox,
            self._vefaas_client,
            function_id,
            image,
            command,
            self.logger,
        )

        if not self._sandbox_id:
            raise RuntimeError("Failed to create sandbox")

        self.logger.info(f"Sandbox {self._sandbox_id} created")
        print(f"Sandbox {self._sandbox_id} created, function_id: {function_id}, instance_id: {self._config.instance_id}")
        self._hooks.on_custom_step("Starting runtime")

        function_route = self._config.function_route or os.getenv("VEFAAS_FUNCTION_ROUTE")
        if not function_route:
            raise ValueError("VEFAAS_FUNCTION_ROUTE environment variable not set")

        print(f"function_route: {function_route}, function_id: {function_id}, instance_id: {self._config.instance_id}")
        runtime_config = RemoteRuntimeConfig(
            base_url=function_route,
            extra_params={"faasInstanceName": self._sandbox_id},
            auth_token=token,
            timeout=self._config.timeout,
        )
        self._runtime = RemoteRuntime.from_config(runtime_config)

        #await self._wait_until_alive(timeout=self._config.startup_timeout)
        self.logger.info("Runtime started")
        print(f"Runtime started, function_id: {function_id}, instance_id: {self._config.instance_id}")

    async def stop(self):
        # Prevent duplicate stops
        if getattr(self, '_stopped', False):
            return

        if self._runtime:
            await self._runtime.close()
            self._runtime = None

        if self._sandbox_id:
            self.logger.info(f"Deleting sandbox {self._sandbox_id}")
            function_id = os.getenv("VEFAAS_FUNCTION_ID")
            if not function_id:
                self.logger.error(
                    "VEFAAS_FUNCTION_ID not set, cannot delete sandbox"
                )
                return

            try:
                # loop = asyncio.get_running_loop()
                # await loop.run_in_executor(
                #     None,
                #     delete_sandbox,
                #     self._vefaas_client,
                #     function_id,
                #     self._sandbox_id,
                #     self.logger,
                # )
                self.logger.info(f"Sandbox {self._sandbox_id} deleted")
            except Exception as e:
                self.logger.error(f"Failed to delete sandbox {self._sandbox_id}: {e}")
            finally:
                self._sandbox_id = None

        self._stopped = True

    @property
    def runtime(self) -> RemoteRuntime:
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    def __del__(self):
        if hasattr(self, '_sandbox_id') and self._sandbox_id and not getattr(self, '_stopped', False):
            msg = "Ensuring vefaas deployment is stopped because object is deleted"
            try:
                self.logger.debug(msg)
            except Exception:
                print(msg)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.stop())
                else:
                    loop.run_until_complete(self.stop())
            except Exception:
                pass
        # Mark as stopped to prevent duplicate cleanup
        self._stopped = True



def get_vefaas_client(
        access_key: str, secret_key: str, region: str
) -> volcenginesdkvefaas.VEFAASApi:
    configuration = volcenginesdkcore.Configuration()
    configuration.ak = access_key
    configuration.sk = secret_key
    configuration.read_timeout = 40
    configuration.connect_timeout = 40
    configuration.auto_retry = False
    configuration.region = region
    configuration.client_side_validation = True
    configuration.proxy = "http://sys-proxy-rd-relay.byted.org:8118"
    api_client = volcenginesdkcore.ApiClient(configuration)
    return volcenginesdkvefaas.VEFAASApi(api_client)


def create_sandbox(
        client: volcenginesdkvefaas.VEFAASApi,
        function_id: str,
        image: str,
        command: str,
        logger: logging.Logger,
) -> str | None:
    if image.startswith("swebench/"):
        image_name = image.replace("swebench/", "", 1)
        image = f"enterprise-public-cn-beijing.cr.volces.com/swe-bench/{image_name}"

    instance_image_info = (
        volcenginesdkvefaas.InstanceImageInfoForCreateSandboxInput(
            image=image,
            port=8000,  # swerex server port
            command=command,
        )
    )
    start_time = time.time()
    try:
        resp = client.create_sandbox(
            volcenginesdkvefaas.CreateSandboxRequest(
                function_id=function_id,
                instance_image_info=instance_image_info,
                timeout=120,
            )
        )
        end_time = time.time()
        logger.info(
            f"Sandbox {resp.sandbox_id} created in {end_time - start_time:.2f}s"
        )
        return resp.sandbox_id
    except Exception as e:
        end_time = time.time()
        logger.error(
            f"Sandbox creation for {image} failed in {end_time - start_time:.2f}s: {e}"
        )
        return None


def delete_sandbox(
        client: volcenginesdkvefaas.VEFAASApi,
        function_id: str,
        sandbox_id: str,
        logger: logging.Logger,
):
    if sandbox_id is None:
        return
    try:
        client.kill_sandbox(
            volcenginesdkvefaas.KillSandboxRequest(
                function_id=function_id,
                sandbox_id=sandbox_id,
            )
        )
    except ApiException as e:
        logger.error(f"Exception when deleting sandbox {sandbox_id}: {e}")
