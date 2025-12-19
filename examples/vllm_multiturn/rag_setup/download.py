from huggingface_hub import snapshot_download, hf_hub_download

snapshot_download(repo_id="jan-hq/Musique-subset", 
                repo_type="dataset",
                local_dir="data/",
                max_workers=64,
                )
# hf_hub_download(repo_id="openslr/librispeech_asr", 
#                 repo_type="dataset",
#                 revision="refs/convert/parquet", 
#                 filename="clean/train.360/0000.parquet", 
#                 local_dir=".",
#                 )