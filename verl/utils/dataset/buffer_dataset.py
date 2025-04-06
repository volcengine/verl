from copy import deepcopy
import logging
import multiprocessing
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from torch.utils.data import Dataset

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


class BufferDataset(Dataset):

    def __init__(self,
                 tokenizer,
                 buffer=None,
                 processor=None,
                 prompt_key="prompt",
                 max_prompt_length=1024,
                 request_id_key="request_id",
                 truncation='error',
                 length=1000000) -> None:
        super().__init__()
        self.buffer = buffer if buffer else multiprocessing.Queue()
        self.prompt_key = prompt_key
        self.request_id_key = request_id_key
        self.processor = processor
        self.tokenizer = tokenizer
        self.length = length
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation

    def __len__(self):
        return self.length

    # override this method to support various message format
    def process_messages(self, messages: List):
        if messages[0]['role'] == 'system':
            raw_prompt = "\n".join([x['content'] for x in messages[1:]])
        else:
            raw_prompt = "\n".join([x['content'] for x in messages])
        return raw_prompt

    # by default, BufferDataset is designed as a Prompt Dataset
    def __getitem__(self, index):
        request_id, request = self.buffer.get()
        row_dict = dict()
        row_dict[self.request_id_key] = request_id

        if request_id != "":
            messages = deepcopy(request.messages)
            raw_prompt = self.process_messages(messages)
        else:
            row_dict = request
            chat = row_dict.pop(self.prompt_key)
            raw_prompt = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=raw_prompt,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)
        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict


class MultiRoundBufferDataset(BufferDataset):
    _manager = multiprocessing.Manager()

    def __init__(self, *args, max_concurrent_wait=16, score_key="score", **kwargs):
        super().__init__(*args, **kwargs)
        self.finish_dict = self._manager.dict()
        self.multi_round_buffer = self._manager.dict()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_wait)
        self.score_key = score_key

    def put_batch(self, data_proto: DataProto):
        request_ids = data_proto.non_tensor_batch.pop("request_id")

        list_of_tensordict = list(data_proto.batch.unbind(0))
        keys = data_proto.non_tensor_batch.keys()
        meta_info = data_proto.meta_info

        enqueue_count = 0
        buffer_count = 0
        for idx, request_id, td in zip(range(len(request_ids)), request_ids, list_of_tensordict):
            non_tensor_dict = {key: data_proto.non_tensor_batch[key][idx] for key in keys}
            if request_id == "":
                # single-round item
                self.buffer.put((td, non_tensor_dict, meta_info))
                # logging.info(f"enqueue a regular prompt with td = {td}")
                enqueue_count += 1
            else:
                # multi-round item
                if request_id not in self.multi_round_buffer:
                    self.multi_round_buffer[request_id] = [(td, non_tensor_dict, meta_info)]
                    buffer_count += 1
                else:
                    trajectory = self.multi_round_buffer[request_id]
                    trajectory.append((td, non_tensor_dict, meta_info))
                    self.multi_round_buffer[request_id] = trajectory
                    buffer_count += 1
        logging.info(f"enqueue_count = {enqueue_count}, {self.buffer.qsize()}")
        logging.info(f"buffer_count = {buffer_count}, {len(self.multi_round_buffer)}")

    def _wait_sample_ready_and_enqueue(self, request_id, score, total_round):
        max_retry = 300
        for i in range(max_retry):
            if request_id not in self.multi_round_buffer:
                time.sleep(10)
                continue
            trajectory = self.multi_round_buffer[request_id]
            if len(trajectory) > total_round:
                self.multi_round_buffer.pop(request_id)
                return
            if len(trajectory) < total_round:
                time.sleep(10)
                continue
            for td, non_tensor_batch, meta_info in trajectory:
                td[self.score_key] = score
                self.buffer.put((td, non_tensor_batch, meta_info))
            self.multi_round_buffer.pop(request_id)
            return

    def notify_score(self, request_id, score, total_round_count=None):
        self.finish_dict[request_id] = score
        if total_round_count is not None:
            self.executor.submit(self._wait_sample_ready_and_enqueue, request_id, score, total_round_count)
        return {}

    # by default, MultiRoundBufferDataset is designed as a Train Dataset
    def __getitem__(self, idx):
        while True:
            if self.buffer.empty():
                logging.debug("buffer is empty")
                time.sleep(1)
            else:
                try:
                    item = self.buffer.get(timeout=1)
                    if self.score_key not in item[0]:
                        item[0][self.score_key] = -1
                    return item
                except Exception as e:
                    logging.warning(f"failed to get sample from buffer: {e}")
