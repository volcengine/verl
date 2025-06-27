# verl image v2

## Important packages version

```txt
cuda==12.4
cudnn==9.8.0
torch==2.6.0
flash_attn=2.8.0    ##
sglang==0.4.6.post5
vllm==0.8.5.post1
transformer_engine==2.3
megatron.core==core_v0.12.1
vidia-cudnn-cu12==9.8.0.87
```

## Deprecate Reason

FA 2.8's wheel will not support CXXABI=False in CUDA 12.4 request, refer to [https://github.com/Dao-AILab/flash-attention/issues/1717](https://github.com/Dao-AILab/flash-attention/issues/1717)