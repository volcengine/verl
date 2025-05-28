save_path=/local_wiki
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2
function now() {
    date '+%Y-%m-%d-%H-%M'
}

# 确保 now() 函数已经定义
# 创建日志目录
mkdir -p logs

# 设置 GPU 并运行，使用合适的日志路径
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# ps aux | grep retrieval_server | grep -v grep | awk '{print $2}' | xargs kill -9
lsof -ti:8000 | xargs kill -9
sleep 0.1

nohup python examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path --faiss_gpu > logs/retriever_$(now).log 2>&1 &
