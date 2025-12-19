SCRIPT_NAME="retrieval_server.py"
CORPUS_PATH="/mnt/nas/bachvd/Code-Agent/verl/data/searchR1_processed_direct/database/small_wiki_1.jsonl"
INDEX_PATH="/mnt/nas/bachvd/Code-Agent/verl/data/searchR1_processed_direct/database/wiki-18_e5.index"
RETRIEVER_MODEL="intfloat/e5-base-v2"
RETRIEVER_NAME="e5"
TOP_K=5
export CUDA_VISIBLE_DEVICES=0
python retrieval_server.py \
    --corpus_path "$CORPUS_PATH" \
    --index_path "$INDEX_PATH" \
    --retriever_model "$RETRIEVER_MODEL" \
    --retriever_name "$RETRIEVER_NAME" \
    --topk "$TOP_K"