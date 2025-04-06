Customized Dataset
===================

verl's PPOTrainer uses `RLHFDataset <https://github.com/volcengine/verl/blob/main/verl/utils/dataset/rl_dataset.py>`_ by default, which requires preprocessing datasets into a format that RLHFDataset can handle. For examples on how to process data, you can refer to the GSM8K data preprocessing example: `gsm8k.py <https://github.com/volcengine/verl/blob/main/examples/data_preprocess/gsm8k.py>`_.

PPOTrainer also supports user-defined datasets. To use a custom dataset, you need to specify it in the data configuration:

.. code:: yaml

   data:
     ...
     custom_cls:
        path: path/to/your/dataset/python/file
        name: dataset_name

Here, ``path`` is the path to the file containing your customized dataset class. If not specified, the pre-implemented dataset will be used. ``name`` is the name of the dataset class within the specified file.

Requirements for Custom Datasets
-------------------------------

There are some constraints on your customized dataset:

1. Your dataset needs to inherit from ``torch.utils.data.Dataset`` and implement the required ``__len__`` and ``__getitem__`` functions.
2. Your dataset's ``__getitem__`` function should retrun a dict type.
3. Your dataset's ``__init__`` function needs to have a parameter ``data_files`` to receive ``config.data.train_files`` or ``config.data.val_files``.

For example:

.. code:: python

   import torch
   from torch.utils.data import Dataset
   import pandas as pd

   class MyCustomDataset(Dataset):
       def __init__(self, data_files):
           if isinstance(data_files, str):
               data_files = [data_files]
               
           dfs = []
           for file in data_files:
               df = pd.read_parquet(file)
               dfs.append(df)
           
           self.data = pd.concat(dfs, ignore_index=True)
           
       def __len__(self):
           return len(self.data)
           
       def __getitem__(self, idx):
           sample = self.data.iloc[idx]
           return {
               "prompt": sample["prompt"],
               "response": sample["response"]
           }

Using the Full Data Config
-------------------------

If your dataset needs the entire data config, you can add a ``config`` parameter in your dataset's ``__init__`` function:

.. code:: python

   import torch
   from torch.utils.data import Dataset
   import pandas as pd

   class MyConfigAwareDataset(Dataset):
       def __init__(self, data_files, config):
           self.config = config
           
           if isinstance(data_files, str):
               data_files = [data_files]
               
           dfs = []
           for file in data_files:
               df = pd.read_parquet(file)
               dfs.append(df)
           
           self.data = pd.concat(dfs, ignore_index=True)
           
           # Use configuration values
           self.prompt_key = config.data.prompt_key
           self.max_length = config.data.max_prompt_length
           
       def __len__(self):
           return len(self.data)
           
       def __getitem__(self, idx):
           sample = self.data.iloc[idx]
           return {
               self.prompt_key: sample[self.prompt_key][:self.max_length],
               "response": sample["response"]
           }

Using Specific Config Fields
--------------------------

If you only want specific fields from the data config, for example, ``prompt_key`` and ``max_prompt_length``, you can just add these names to your dataset's ``__init__`` function. verl will automatically extract these specified arguments and look up the data config to pass these values when initializing your dataset, so keep the parameter names the same as the keys in the config.

.. code:: python

   import torch
   from torch.utils.data import Dataset
   import pandas as pd

   class MySpecificConfigDataset(Dataset):
       def __init__(self, data_files, prompt_key, max_prompt_length):
           if isinstance(data_files, str):
               data_files = [data_files]

           dfs = []
           for file in data_files:
               df = pd.read_parquet(file)
               dfs.append(df)
           
           self.data = pd.concat(dfs, ignore_index=True)
           self.prompt_key = prompt_key
           self.max_prompt_length = max_prompt_length
           
       def __len__(self):
           return len(self.data)
           
       def __getitem__(self, idx):
           sample = self.data.iloc[idx]
           return {
               self.prompt_key: sample[self.prompt_key][:self.max_prompt_length],
               "response": sample["response"]
           }

Special Cases: Tokenizer and Processor
------------------------------------

There are some exceptions: when you specify ``tokenizer`` in your dataset's ``__init__`` parameters, you will get the Tokenizer instance instead of the tokenizer path (str) specified in the data config. The same applies to ``processor``.

.. code:: python

   import torch
   from torch.utils.data import Dataset
   import pandas as pd

   class MyTokenizerDataset(Dataset):
       def __init__(self, data_files, tokenizer, max_prompt_length):
           if isinstance(data_files, str):
               data_files = [data_files]
               
           dfs = []
           for file in data_files:
               df = pd.read_parquet(file)
               dfs.append(df)
           
           self.data = pd.concat(dfs, ignore_index=True)
           self.tokenizer = tokenizer
           self.max_prompt_length = max_prompt_length
           
       def __len__(self):
           return len(self.data)
           
       def __getitem__(self, idx):
           sample = self.data.iloc[idx]
           prompt = sample["prompt"]

           inputs = self.tokenizer(
               prompt,
               truncation=True,
               max_length=self.max_prompt_length,
               return_tensors="pt"
           )
           
           return {
               "input_ids": inputs.input_ids.squeeze(0),
               "attention_mask": inputs.attention_mask.squeeze(0),
               "response": sample["response"]
           }

Custom Configuration
-----------------

If you want additional configuration for your customized dataset, you can add the config under the data config section and use the corresponding argument to receive these configs. For example:

.. code:: yaml

   data:
     ...
     custom_cls:
        path: path/to/your/dataset/python/file
        name: dataset_name
     custom_config:
        your_key1: your_value1
        your_key2: your_value2
        ...

To receive this customized config, your dataset should have a parameter ``custom_config`` (the same name as in the data config):

.. code:: python

   import torch
   from torch.utils.data import Dataset
   import pandas as pd

   class MyCustomConfigDataset(Dataset):
       def __init__(self, data_files, custom_config):
           if isinstance(data_files, str):
               data_files = [data_files]
               
           dfs = []
           for file in data_files:
               df = pd.read_parquet(file)
               dfs.append(df)
           
           self.data = pd.concat(dfs, ignore_index=True)

           self.special_key = custom_config.get("your_key1")
           self.special_value = custom_config.get("your_key2")
           
       def __len__(self):
           return len(self.data)
           
       def __getitem__(self, idx):
           sample = self.data.iloc[idx]
           
           if self.special_key == "filter_by_length":
               if len(sample["prompt"]) > self.special_value:
                   return {
                       "prompt": sample["prompt"][:self.special_value],
                       "response": sample["response"],
                       "truncated": True
                   }

           return {
               "prompt": sample["prompt"],
               "response": sample["response"],
               "truncated": False
           }