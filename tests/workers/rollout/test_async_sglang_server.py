# Copyright 2023-2024 SGLang Team
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from omegaconf import DictConfig


@patch.dict(
    "sys.modules",
    {
        "verl.workers.rollout.sglang_rollout.sglang_rollout": MagicMock(SGLangRollout=MagicMock()),
    },
)
class TestAsyncSglangServer:
    @pytest.fixture
    def server_config(self):
        return DictConfig({"rollout": {"tensor_model_parallel_size": 2}})

    @pytest.mark.asyncio
    @patch("verl.workers.rollout.sglang_rollout.async_sglang_server.ray.util.list_named_actors")
    @patch("verl.workers.rollout.async_server.AsyncServerBase._start_fastapi_server", new_callable=AsyncMock)
    @pytest.mark.filterwarnings("ignore:Ray state API is no longer experimental:DeprecationWarning")
    async def test_init_engine(self, mock_start_fastapi_server, mock_list_actors, server_config):
        mock_list_actors.return_value = [
            {"name": "test_prefixWorkerDict_1:0", "namespace": "test"},
            {"name": "test_prefixWorkerDict_1:1", "namespace": "test"},
            {"name": "test_prefixWorkerDict_0:0", "namespace": "test"},
            {"name": "test_prefixWorkerDict_0:1", "namespace": "test"},
            {"name": "test_prefixWorkerDict_1:2", "namespace": "test"},
            {"name": "test_prefixWorkerDict_1:3", "namespace": "test"},
            {"name": "test_prefixWorkerDict_0:2", "namespace": "test"},
            {"name": "test_prefixWorkerDict_0:3", "namespace": "test"},
        ]
        from verl.workers.rollout.sglang_rollout.async_sglang_server import AsyncSglangServer

        ActualClassToInstantiate = AsyncSglangServer
        if hasattr(AsyncSglangServer, "__ray_metadata__") and hasattr(AsyncSglangServer.__ray_metadata__, "modified_class"):
            ActualClassToInstantiate = AsyncSglangServer.__ray_metadata__.modified_class

        def mock_get_actor_side_effect(name, namespace=None):
            # Create a new mock actor for each call
            actor_mock = MagicMock()

            # Support .name attribute access
            actor_mock.name = name  # Use 'name' here

            # Support ['name'] item access by mocking __getitem__
            def getitem_mock(key):
                if key == "name":
                    return name  # Use 'name' here
                # For other keys, return a new MagicMock to mimic default behavior or raise KeyError
                # Returning a MagicMock is consistent with the original error's cause for unmocked keys
                return MagicMock(name=f"mock.__getitem__('{key}')")

            actor_mock.__getitem__.side_effect = getitem_mock

            return actor_mock

        # Verify instance.workers is correctly populated
        with patch("verl.workers.rollout.sglang_rollout.async_sglang_server.ray.get_actor", side_effect=mock_get_actor_side_effect):
            # Instance 1
            instance = ActualClassToInstantiate(server_config, 4, 0, "test_prefix")
            await instance.init_engine()

            assert len(instance.workers) == 2
            assert instance.master_worker["name"] == "test_prefixWorkerDict_0:0"
            assert instance.workers[0].name == "test_prefixWorkerDict_0:0"
            assert instance.workers[1].name == "test_prefixWorkerDict_0:1"

            # Instance 2
            instance = ActualClassToInstantiate(server_config, 4, 1, "test_prefix")
            await instance.init_engine()

            assert len(instance.workers) == 2
            assert instance.master_worker["name"] == "test_prefixWorkerDict_0:2"
            assert instance.workers[0].name == "test_prefixWorkerDict_0:2"
            assert instance.workers[1].name == "test_prefixWorkerDict_0:3"

            # Instance 3
            instance = ActualClassToInstantiate(server_config, 4, 3, "test_prefix")
            await instance.init_engine()

            assert len(instance.workers) == 2
            assert instance.master_worker["name"] == "test_prefixWorkerDict_1:2"
            assert instance.workers[0].name == "test_prefixWorkerDict_1:2"
            assert instance.workers[1].name == "test_prefixWorkerDict_1:3"
