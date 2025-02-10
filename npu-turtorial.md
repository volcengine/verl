环境信息：

1. NPU 基础镜像，安装最新的 CANN-Toolkit & Kernel & NNAL
2. 安装 veRL 和相关依赖
	1. veRL
		1. `git clone https://github.com/chendong-1998/verl.git`
		2. `cd verl && git checkout support-ascend-npu` 
		3. `pip install -r requirements.txt`
	2. vLLM on NPU
		1. 参考该 PR https://github.com/vllm-project/vllm/pull/8054
		2. `git clone -b 1130/npu_support https://github.com/Chendong98/vllm.git`
		3. `cd vllm && VLLM_TARGET_DEVICE=npu pip install -e .`
	3. Megatron
		1. `git clone https://github.com/NVIDIA/Megatron-LM.git`
		2. `git checkout core_r0.6.0`
		3. `git apply verl/patches/megatron_v0.6_npu.patch`
		4. `pip install -e .`
	4. Mindspeed
		1. `git clone https://gitee.com/ascend/MindSpeed.git`
		2. `git checkout core_r0.6.0` (or commit id `e7ea32a1e054`)
		3. `pip install -e .`
	5. Ascend-Apex
		1. `git clone https://gitee.com/ascend/apex.git ascend-apex`
		2. `cd ascend-apex && bash scripts/build.sh --python=3.10`
		3. 安装构建的 whl 包

另外可能需要校对模型、数据集路径、修改 ` verl/third_party/vllm/__init__.py` 指定安装的 vllm develop 版本（可能形如 `0.1.dev3628+g06f1b1d.d20250116.npu`）使用 `0.6.4` 目录的 spmd vllm。

最后进入 veRL 执行 `bash example/ppo_trainer/run_llama_3.2_1b_megatron.sh` 即可在 NPU 上运行。
