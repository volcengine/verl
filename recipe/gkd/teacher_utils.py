import torch
from verl import DataProto


def get_teacher_knowledge(batch: DataProto, teacher_client, n_server_workers=1):
    input_ids = []
    attention_mask = batch.batch["attention_mask"].to(torch.bool)

    for ids, mask in zip(batch.batch["input_ids"], attention_mask):
        input_ids.append(ids[mask].tolist())
        
    all_teacher_topk_logps = []
    all_teacher_topk_indices = []

    batch_size = len(input_ids)
    assert batch_size % n_server_workers == 0
    micro_batch_size = batch_size // n_server_workers
    futures = []
    for i in range(0, batch_size, micro_batch_size):
        futures.append(teacher_client.submit(input_ids[i:i+micro_batch_size]))

    for future in futures:
        try:
            _, teacher_topk_logps, teacher_topk_indices = future.result()
        except Exception as e:
            raise RuntimeError(f"Teacher request failed: {e}") from e

        all_teacher_topk_logps.extend(teacher_topk_logps)
        all_teacher_topk_indices.extend(teacher_topk_indices)

    teacher_topk_logps, teacher_topk_indices = all_teacher_topk_logps, all_teacher_topk_indices
            
    topk = teacher_topk_logps[0].size(-1)
    logp_dtype = teacher_topk_logps[0].dtype
    indice_dtype = teacher_topk_indices[0].dtype
    teacher_knowledge_shape = list(batch.batch["input_ids"].shape) + [topk]
    output_batch = DataProto.from_single_dict({"teacher_topk_logps": torch.zeros(*teacher_knowledge_shape, dtype=logp_dtype),
                                                "teacher_topk_indices": torch.zeros(*teacher_knowledge_shape, dtype=indice_dtype)})
    output_batch.batch['teacher_topk_logps'][attention_mask] = torch.cat(teacher_topk_logps)
    output_batch.batch['teacher_topk_indices'][attention_mask] = torch.cat(teacher_topk_indices)
    
    return output_batch