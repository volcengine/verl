corpus_file=data/corpus/corpus.jsonl # jsonl
save_dir=data/corpus
retriever_name=e5 # this is for indexing naming
retriever_model=intfloat/e5-base-v2

echo "Starting FlashRAG server..."
# python flashrag_server.py \
#     --index_path $save_dir/${retriever_name}_Flat.index \
#     --corpus_path $corpus_file \
#     --retrieval_topk 25 \
#     --retriever_name $retriever_name \
#     --retriever_model $retriever_model \
#     --reranking_topk 10 \
#     --reranker_model "cross-encoder/ms-marco-MiniLM-L12-v2" \
#     --reranker_batch_size 64 \
#     --host "0.0.0.0" \
#     --port 3030 \
#     --faiss_gpu \
#     --workers 64

export INDEX_PATH=$save_dir/${retriever_name}_Flat.index
export CORPUS_PATH=$corpus_file
export RETRIEVER_NAME=$retriever_name
export RETRIEVER_MODEL=$retriever_model

uvicorn flashrag_server:app \
    --host 0.0.0.0 \
    --port 3030 \
    --workers 64