"""
Utility functions for teacher model knowledge distillation.

Functions:
    get_teacher_knowledge: Retrieve teacher model's top-k predictions and log probabilities.
"""
import time
from types import SimpleNamespace

import torch
from torch.nn import functional as F
import numpy as np
from verl import DataProto


def chunk_list(lst, n):
    assert n > 0 and len(lst) >= n and len(lst) % n == 0
    chunk_size = len(lst) // n
    for i in range(0, len(lst), chunk_size):
        yield lst[i: i + chunk_size]


def get_teacher_knowledge(batch: DataProto, teacher_client, n_server_workers=1, is_async=False):
    """
    Retrieve teacher model's top-k predictions and log probabilities for knowledge distillation.

    Args:
        batch (DataProto): Input batch containing input_ids and attention_mask
        teacher_client: Client for communicating with teacher model
        n_server_workers (int): Number of parallel workers for teacher model inference
        is_async (bool): Whether to use asynchronous processing

    Returns:
        If is_async=True: SimpleNamespace with get() method to process futures
        If is_async=False: Processed DataProto containing teacher knowledge

    Raises:
        RuntimeError: If teacher model request fails
    """
    input_ids = []
    attention_mask = batch.batch["attention_mask"].to(torch.bool)
    # response_length = batch.meta_info["response_length"]

    for ids, mask in zip(batch.batch["input_ids"], attention_mask):
        input_ids.append(ids[mask].tolist())
        
    all_teacher_topk_logps = []
    all_teacher_topk_indices = []

    batch_size = len(input_ids)
    assert batch_size % n_server_workers == 0
    micro_batch_size = batch_size // n_server_workers
    futures = []
    tik = time.time()
    tok = None
    def cb(future):
        nonlocal tok
        tok = time.time()

    for i in range(0, batch_size, micro_batch_size):
        fut = teacher_client.submit(input_ids[i: i + micro_batch_size])
        fut.add_done_callback(cb)
        futures.append(fut)

    def handle_futures():
        for future in futures:
            try:
                _, teacher_topk_logps, teacher_topk_indices = future.result()
            except Exception as e:
                raise RuntimeError(f"Teacher request failed: {e}") from e

            all_teacher_topk_logps.extend(teacher_topk_logps)
            all_teacher_topk_indices.extend(teacher_topk_indices)

        # teacher_topk_logps = [x.to(params_dtype) for x in all_teacher_topk_logps]
        # teacher_topk_indices = [x.to(params_dtype) for x in all_teacher_topk_indices]
        teacher_topk_logps, teacher_topk_indices = all_teacher_topk_logps, all_teacher_topk_indices

        real_seq_lens = torch.tensor([x.size(0) for x in teacher_topk_logps], dtype=torch.int32)

        topk = teacher_topk_logps[0].size(-1)

        teacher_topk_logps_total = torch.cat(teacher_topk_logps)
        teacher_topk_indices_total = torch.cat(teacher_topk_indices)
        teacher_knowledge_shape = list(batch.batch["input_ids"].shape) + [topk]
        teacher_topk_logps_padded = torch.zeros(*teacher_knowledge_shape, 
                                                dtype=teacher_topk_logps_total.dtype)
        teacher_topk_indices_padded = torch.zeros(*teacher_knowledge_shape, 
                                                  dtype=teacher_topk_indices_total.dtype)
        teacher_topk_logps_padded[attention_mask] = teacher_topk_logps_total
        teacher_topk_indices_padded[attention_mask] = teacher_topk_indices_total

        output_batch = DataProto.from_single_dict(
            data={"real_seq_lens": real_seq_lens},
            meta_info={
                "timing": {"get_teacher_knowledge": tok - tik},
            }
        )

        output_batch.non_tensor_batch.update({
            "teacher_topk_logps": teacher_topk_logps_padded.numpy(),
            "teacher_topk_indices": teacher_topk_indices_padded.numpy()
        })

        return output_batch
    
    if is_async:
        return SimpleNamespace(get=handle_futures)
    else:
        return handle_futures()
    

if __name__ == "__main__":
    batch = DataProto.load_from_disk("gen_batch_output")
    from teacher import TeacherClient
    teacher_client = TeacherClient(server_ip="10.215.192.141", server_port=15555)
    output_batch = get_teacher_knowledge(batch, 2, teacher_client)
    output_batch_chunks = output_batch.chunk(2)
    
    for data in output_batch_chunks:
        topk = data.meta_info["topk"]
        seq_lens = data.batch["seq_lens"]
        teacher_topk_logps = data.batch["teacher_topk_logps"].view(-1, topk)
        teacher_topk_indices = data.batch["teacher_topk_indices"].view(-1, topk)

        attention_mask = data.batch["attention_mask"]
        batch_size, sequence_length = attention_mask.size(0), attention_mask.size(1)
        teacher_topk_logps_padded = torch.zeros(batch_size, sequence_length, topk, dtype=teacher_topk_logps.dtype)
        teacher_topk_indices_padded = torch.zeros(batch_size, sequence_length, topk, dtype=teacher_topk_indices.dtype)
        
        teacher_topk_logps_padded[attention_mask] = teacher_topk_logps[:seq_lens.sum()]
        teacher_topk_indices_padded[attention_mask] = teacher_topk_indices[:seq_lens.sum()]

        data.batch["teacher_topk_logps"] = teacher_topk_logps_padded
        data.batch["teacher_topk_indices"] = teacher_topk_indices_padded

        assert (data.batch["teacher_topk_logps"] == data.batch["teacher_topk_logps_padded"]).all()
        assert (data.batch["teacher_topk_indices"] == data.batch["teacher_topk_indices_padded"]).all()