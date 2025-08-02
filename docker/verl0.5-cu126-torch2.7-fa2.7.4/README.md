# verl image with verl v0.5

## Important packages version

```txt
cuda==12.6
cudnn==9.8.0
torch==2.7.1
flash_attn=2.7.4.post1
sglang==0.4.9.post6
vllm==0.8.5.post1
vidia-cudnn-cu12==9.8.0.87
transformer_engine==2.3
megatron.core==core_v0.12.2
# Preview
transformer_engine==2.5
megatron.core==core_r0.13.0
```

## Target

- Base image:
    - `verlai/verl:base-verl0.5-cu126-cudnn9.8-torch2.7.0-fa2.7.4`: We offer a base image with deep ep built in, for vllm
    - `verlai/verl:base-verl0.5-cu126-cudnn9.8-torch2.7.1-fa2.7.4`: We offer a base image with deep ep built in, for sglang
- App image:
    - `verlai/verl:app-verl0.5-vllm0.9.1-mcore0.12.2-te2.2`
    - `verlai/verl:app-verl0.5-sglang0.4.8-mcore0.12.2-te2.2`
    - `verlai/verl:app-verl0.5-sglang0.4.9.post6-mcore0.12.2-te2.2`