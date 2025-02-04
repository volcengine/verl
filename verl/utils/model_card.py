"""Utilities for generating model cards."""

import os
from typing import Optional, List
from huggingface_hub import ModelCard, ModelCardData
from importlib.metadata import version


def generate_model_card(
    base_model: Optional[str],
    model_name: str,
    hub_model_id: str,
    dataset_name: Optional[str],
    tags: List[str],
    wandb_url: Optional[str],
    trainer_name: str,
    trainer_citation: Optional[str] = None,
    paper_title: Optional[str] = None,
    paper_id: Optional[str] = None,
) -> ModelCard:
    """
    Generate a `ModelCard` from a template.

    Args:
        base_model (`str` or `None`):
            Base model name.
        model_name (`str`):
            Model name.
        hub_model_id (`str`):
            Hub model ID as `username/model_id`.
        dataset_name (`str` or `None`):
            Dataset name.
        tags (`list[str]`):
            Tags.
        wandb_url (`str` or `None`):
            Weights & Biases run URL.
        trainer_name (`str`):
            Trainer name.
        trainer_citation (`str` or `None`, defaults to `None`):
            Trainer citation as a BibTeX entry.
        paper_title (`str` or `None`, defaults to `None`):
            Paper title.
        paper_id (`str` or `None`, defaults to `None`):
            ArXiv paper ID as `YYMM.NNNNN`.

    Returns:
        `ModelCard`:
            A ModelCard object.
    """
    card_data = ModelCardData(
        base_model=base_model,
        datasets=dataset_name,
        library_name="transformers",
        license="apache-2.0",
        model_name=model_name,
        tags=["generated_from_trainer", "verl", *tags],
    )

    template = """
# {model_name}

This model is a fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) using VeRL (Versatile Reinforcement Learning).

## Model Details

### Model Description

- **Model Type:** {model_name}
- **Base Model:** [{base_model}](https://huggingface.co/{base_model})
- **Training Method:** {trainer_name}
- **Language(s):** English

## Training Details

- **Training Data:** {dataset_name}
- **Training Procedure:** {trainer_name}
- **Framework Versions:**
  - VeRL: {verl_version}
  - Transformers: {transformers_version}
  - PyTorch: {pytorch_version}
  - Datasets: {datasets_version}
  - Tokenizers: {tokenizers_version}

{wandb_section}

## Citation

If you use this model, please cite:

{trainer_citation}

{paper_section}

## Model Card Authors

This model card was generated automatically using VeRL.
    """

    wandb_section = ""
    if wandb_url:
        wandb_section = f"For more details about the training process, see the [Weights & Biases run]({wandb_url})."

    paper_section = ""
    if paper_title and paper_id:
        paper_section = f"""
## Research Paper

This model is part of the research paper:

**{paper_title}**
ArXiv: [{paper_id}](https://arxiv.org/abs/{paper_id})
"""

    card = ModelCard.from_template(
        card_data,
        template,
        model_name=model_name,
        base_model=base_model,
        hub_model_id=hub_model_id,
        dataset_name=dataset_name,
        wandb_section=wandb_section,
        trainer_name=trainer_name,
        trainer_citation=trainer_citation or "",
        paper_section=paper_section,
        verl_version=version("verl"),
        transformers_version=version("transformers"),
        pytorch_version=version("torch"),
        datasets_version=version("datasets"),
        tokenizers_version=version("tokenizers"),
    )
    return card 