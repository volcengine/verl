# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

import torch
from verl import DataProto
from verl.utils.think_utils import create_think_mask, process_think_responses
from verl.workers.rollout.vllm_rollout import vLLMRollout

class ThinkAwarevLLMRollout(vLLMRollout):
    """扩展vLLMRollout以支持think处理"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_thinking = kwargs.get('enable_thinking', False)
    
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # 调用父类的生成方法
        result = super().generate_sequences(prompts, **kwargs)
        
        # 如果启用think处理，添加think相关信息
        if self.enable_thinking and prompts.meta_info.get("enable_thinking", False):
            batch_size = result.batch["responses"].size(0)
            
            # 解码响应文本并分离think内容
            think_info = []
            for i in range(batch_size):
                response_text = self.tokenizer.decode(
                    result.batch["responses"][i], skip_special_tokens=True
                )
                think_data = extract_think_content(response_text)
                think_info.append(think_data)
            
            # 创建think mask
            think_mask = create_think_mask(
                result.batch["responses"], self.tokenizer
            )
            
            # 添加到结果中
            result.batch["think_mask"] = think_mask
            result.non_tensor_batch["think_info"] = think_info
        
        return result