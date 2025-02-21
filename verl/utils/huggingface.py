"""Utilities for interacting with the Hugging Face Hub."""

import os
from typing import Optional, Union, List
from huggingface_hub import HfApi
from .model_card import generate_model_card
import wandb


def get_base_model_name(model_path: str) -> str:
    """Extract the base model name from a model path.
    
    Args:
        model_path (`str`):
            Path to the model. Can be a local path or a Hugging Face Hub model ID.
            
    Returns:
        `str`: The base model name.
    """
    if model_path.startswith("~/"):
        model_path = os.path.expanduser(model_path)
    if os.path.isdir(model_path):
        # If it's a local path, get the basename
        return os.path.basename(model_path)
    return model_path


def get_dataset_name(train_files: Union[str, List[str]]) -> Optional[str]:
    """Extract dataset name from training files path.
    
    Args:
        train_files (`str` or `List[str]`):
            Path or list of paths to training files.
            
    Returns:
        `str` or `None`: The dataset name(s) joined by commas, or None if not available.
    """
    if isinstance(train_files, str):
        return os.path.basename(os.path.dirname(train_files))
    elif isinstance(train_files, (list, tuple)):
        dataset_names = [os.path.basename(os.path.dirname(f)) for f in train_files]
        return ", ".join(dataset_names)
    return None


def push_to_hub(
    repo_id: str,
    local_path: str,
    private: bool = False,
    token: Optional[str] = None,
    base_model: Optional[str] = None,
    model_name: Optional[str] = None,
    training_config: Optional[dict] = None,
    tags: Optional[List[str]] = None,
    trainer_name: Optional[str] = None,
):
    """Push a model to the Hugging Face Hub with comprehensive metadata and model card.

    Args:
        repo_id (`str`):
            The name of the repository you want to push your model to. It should be in the format `username/model_name`.
        local_path (`str`):
            Path to the local directory containing the model files.
        private (`bool`, *optional*, defaults to `False`):
            Whether the model repository should be private.
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If None, will use the token generated when
            running `huggingface-cli login` (stored in `~/.huggingface`).
        base_model (`str`, *optional*):
            The name or path of the base model used for training.
        model_name (`str`, *optional*):
            The name of the model for the model card.
        training_config (`dict`, *optional*):
            Training configuration containing dataset information and other metadata.
        tags (`List[str]`, *optional*):
            Additional tags to add to the model card.
        trainer_name (`str`, *optional*):
            Name of the trainer used (e.g., "PPO", "SFT").
    """
    # Extract base model name if provided
    if base_model:
        base_model = get_base_model_name(base_model)
    
    # Extract dataset name from training config if available
    dataset_name = None
    if training_config and "data" in training_config:
        dataset_name = get_dataset_name(training_config["data"].get("train_files"))
    
    # Get wandb URL if available
    wandb_url = None
    if wandb.run is not None:
        wandb_url = wandb.run.get_url()
    
    # Generate model card if we have the necessary information
    if base_model and model_name:
        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=repo_id,
            dataset_name=dataset_name,
            tags=tags or [],
            wandb_url=wandb_url,
            trainer_name=trainer_name or "Unknown",
        )
        
        # Save model card
        model_card.save(os.path.join(local_path, "README.md"))
    
    # Push to hub
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, token=token, exist_ok=True)
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        token=token,
    ) 