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
Usage: python scripts/print_hydra_config.py [--config-path=<path>] [--config-name=<name>] +group/path=config_name

NOTE: The `config_path` is relative to the script directory, while the `searchpath` is relative to the working directory.
c.f. https://hydra.cc/docs/advanced/search_path/

Example:

```bash
python scripts/print_hydra_config.py data.last_user_msg_template="\"{{ content }}\n\nLet\'s think step by step.\""
```

```
WRONG: "{{ content }}\\n\\nLet\\'s think step by step."
```

```bash
python scripts/print_hydra_config.py "hydra.searchpath=[configs/rl_trainer]" +data=cot_sfx_template
```

```
# CORRECT
"{{ content | trim }}\n\nLet's think step by step."
```

```bash
python scripts/print_hydra_config.py --config-path="../recipe/prime/config" --config-name="prime_trainer" \
    "hydra.searchpath=[verl/trainer/config,configs/rl_trainer]" \
        +data=cot_sfx_template
```

```
# CORRECT
"{{ content | trim }}\n\nLet's think step by step."
```
"""
import hydra
from pprint import pprint
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path='../verl/trainer/config', config_name='ppo_trainer', version_base=None)
def main(config: DictConfig):
    pprint(OmegaConf.to_container(config, resolve=True))


if __name__ == "__main__":
    main()
