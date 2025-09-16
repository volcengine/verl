# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

import re
import torch
from typing import Dict, List, Tuple, Optional

def extract_think_content(text: str) -> Dict[str, str]:
    """提取think内容和answer内容"""
    pattern = r"<think>(.*?)</think>(.*)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return {
            "think_content": match.group(1).strip(),
            "answer_content": match.group(2).strip(),
            "has_think": True
        }
    return {"think_content": "", "answer_content": text, "has_think": False}

def create_think_mask(response_ids: torch.Tensor, tokenizer, device=None) -> torch.Tensor:
    """创建think mask，1表示think部分，0表示answer部分"""
    think_start_tokens = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    
    if len(think_start_tokens) == 0 or len(think_end_tokens) == 0:
        return torch.zeros_like(response_ids, dtype=torch.bool)
    
    think_start_id = think_start_tokens[0]
    think_end_id = think_end_tokens[-1]
    
    think_mask = torch.zeros_like(response_ids, dtype=torch.bool)
    
    for i, seq in enumerate(response_ids):
        think_start_pos = (seq == think_start_id).nonzero(as_tuple=True)[0]
        think_end_pos = (seq == think_end_id).nonzero(as_tuple=True)[0]
        
        if len(think_start_pos) > 0 and len(think_end_pos) > 0:
            start_idx = think_start_pos[0]
            end_idx = think_end_pos[0] + 1
            think_mask[i, start_idx:end_idx] = True
    
    return think_mask

def process_think_responses(responses: List[str]) -> List[Dict[str, str]]:
    """批量处理think响应"""
    return [extract_think_content(resp) for resp in responses]