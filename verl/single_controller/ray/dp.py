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

import ray

from verl.single_controller.ray.base import RayWorkerGroup, RayResourcePool, RayClassWithInitArgs


@ray.remote
class RefBasicRayActor:
    ...


class DPEngineRayWorkerGroup(RayWorkerGroup):

    class DummyModule:

        def __init__(self, core, methods_names) -> None:
            self.core = core

            def func_generator(method_name):

                def func(*args, **kwargs):
                    return self.core.execute_all_async("execute_engine", method_name, *args, **kwargs)

                return func

            for method_name in methods_names:
                setattr(self, method_name, func_generator(method_name))

    def __init__(self, name_prefix, process_dispatch_scheme, use_gpu, engine_type, *args, **kwargs) -> None:
        from torch import nn
        # print(f"in DataParallelEngineWrapper, name_prefix = {name_prefix}")
        if isinstance(process_dispatch_scheme, RayResourcePool):
            rpdc = process_dispatch_scheme
        else:
            rpdc = RayResourcePool(process_on_nodes=process_dispatch_scheme,
                                   use_gpu=use_gpu,
                                   name_prefix=name_prefix,
                                   max_colocate_count=1)
        rcia = RayClassWithInitArgs(cls=engine_type, *args, **kwargs)

        self._engine_type = engine_type

        super().__init__(rpdc, rcia)

        nn_module_methods = [
            method_name for method_name in dir(nn.Module)
            if callable(getattr(nn.Module, method_name)) and not method_name.startswith("__")
        ]
        nn_module_methods += ["__call__"]

        def func_generator(method_name):

            def func(*args, **kwargs):
                return self.execute_all_async(method_name, *args, **kwargs)

            return func

        print(f"{engine_type} has methods: {dir(engine_type)}")
        for method_name in dir(engine_type):
            try:
                is_callable = callable(getattr(engine_type, method_name))
            except Exception as _:
                pass
            else:
                if is_callable and method_name not in dir(RefBasicRayActor):
                    print(f"register method: {method_name}")
                    setattr(self, method_name, func_generator(method_name))

        self.module = DPEngineRayWorkerGroup.DummyModule(self, nn_module_methods)

    @property
    def engine(self):
        return self.module

    def get_model_size_on_rank_zero(self):
        results = ray.get([worker.get_model_size_on_rank_zero.remote() for worker in self._workers])

        for result in results:
            if result is not None:
                return result
