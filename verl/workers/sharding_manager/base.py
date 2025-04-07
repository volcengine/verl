# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Sharding manager to implement HybridEngine
"""
import abc

from verl import DataProto


class BaseShardingManager(metaclass=abc.ABCMeta):

    def __enter__(self) -> None:
        return self.enter_sharding_context()

    # Put type hints.
    def __exit__(self,
                 exc_type: type,
                 exc_value: Exception,
                 traceback: object) -> None:

        return self.exit_sharding_context(exc_type, exc_value, traceback)

    @abc.abstractmethod
    def enter_sharding_context(self) -> None:
        """
        For explicitly entering sharding context, e.g., DAPO rollout.
        """
        pass

    @abc.abstractmethod
    def exit_sharding_context(self, 
                              exc_type: type,
                              exc_value: Exception,
                              traceback: object) -> None:
        """
        For explicitly exiting sharding context, e.g., DAPO rollout.
        """
        pass

    def preprocess_data(self, data: DataProto) -> DataProto:
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        return data
