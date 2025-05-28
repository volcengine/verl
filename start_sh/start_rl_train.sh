export WANDB_API_KEY=7a5831d26740f1e82f08b0878d05950f8cbf727e


ps aux | grep verl.trainer.main_ppo | grep -v grep | awk '{print $2}' | xargs kill -9
# 定义时间戳函数
function now() {
    date '+%Y-%m-%d-%H-%M'
}

# 确保 now() 函数已经定义
# 创建日志目录
mkdir -p logs

# 设置 GPU 并运行，使用合适的日志路径
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
sleep 0.1

nohup bash examples/sglang_multiturn/search_r1_like/run_qwen2.5-3b_instruct_search_multiturn.sh trainer.experiment_name=qwen2.5-3b-it_rm-searchR1-like-sgl-multiturn-ftest$(now) > logs/searchR1-like$(now).log 2>&1 &