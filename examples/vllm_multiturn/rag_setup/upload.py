from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="/home/jovyan/visual-thinker-workspace/deep-research/src/index_musique_db",
    path_in_repo="corpus/", # Upload to a specific folder
    repo_id="jan-hq/Musique-subset",
    repo_type="dataset",
)