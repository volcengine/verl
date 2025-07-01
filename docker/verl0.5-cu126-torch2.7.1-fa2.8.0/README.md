# verl image with verl v0.5

## Important packages version

```txt
cuda==12.4
cudnn==9.8.0
torch==2.6.0
flash_attn=2.8.0    ##
sglang==0.4.8
vllm==0.8.5.post1
transformer_engine==2.3
megatron.core==core_v0.12.1
vidia-cudnn-cu12==9.8.0.87
```

## Target

- Base image:
    - `verlai/verl:base-verl0.5-cu126-cudnn9.8-torch2.7.1-fa2.8.0-fi0.2.6`: We offer a base image with flash infer 0.2.6.post1 built in
- App image:
    - `verlai/verl:app-verl0.5-sglang0.4.8-mcore0.12.1`
    - `verlai/verl:app-verl0.5-sglang0.4.8-mcore0.13.0-preview`
- vllm temporarily not support latest version