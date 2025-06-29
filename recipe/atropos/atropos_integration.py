"""
Atropos Integration for VeRL - Real Environment Feedback

This module provides the core integration between VeRL and Atropos environments,
enabling real-time advantage computation from environment feedback.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class AtroposEnvironmentClient:
    """
    Client for communicating with Atropos environments.

    This client handles:
    - Submitting generated responses to Atropos
    - Retrieving token-level advantages computed by the environment
    - Managing batch processing and synchronization
    """

    def __init__(self, api_url: str = "http://localhost:8000", timeout: int = 30):
        self.api_url = api_url
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self) -> bool:
        """Check if Atropos server is healthy"""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_prompts(self, batch_size: int) -> Optional[Dict[str, Any]]:
        """
        Get a batch of prompts from Atropos environments.

        Returns:
            Dictionary containing:
            - prompts: List of prompt strings
            - metadata: Environment-specific metadata
            - group_ids: IDs for tracking groups
        """
        try:
            response = self.session.post(f"{self.api_url}/get_prompts", json={"batch_size": batch_size}, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get prompts: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error getting prompts: {e}")
            return None

    def submit_responses_and_get_advantages(self, prompts: List[str], responses: List[str], metadata: Dict[str, Any], tokenizer: AutoTokenizer) -> Optional[Dict[str, Any]]:
        """
        Submit responses to Atropos and get token-level advantages.

        This is the core integration point where:
        1. Generated responses are sent to the environment
        2. The environment evaluates them (e.g., checks math correctness)
        3. Token-level advantages are computed and returned

        Args:
            prompts: List of prompt strings
            responses: List of generated response strings
            metadata: Environment metadata from get_prompts
            tokenizer: Tokenizer for encoding/decoding

        Returns:
            Dictionary containing:
            - advantages: Token-level advantages tensor
            - rewards: Sequence-level rewards
            - env_metrics: Environment-specific metrics
        """
        # Tokenize prompts and responses
        tokenized_data = []
        for prompt, response in zip(prompts, responses):
            # Tokenize full sequence
            full_text = prompt + response
            tokens = tokenizer.encode(full_text, add_special_tokens=True)

            # Create mask (0 for prompt, 1 for response)
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            mask = [0] * len(prompt_tokens) + [1] * (len(tokens) - len(prompt_tokens))

            tokenized_data.append({"tokens": tokens, "mask": mask, "prompt": prompt, "response": response})

        # Prepare request payload
        payload = {"tokenized_data": tokenized_data, "metadata": metadata, "tokenizer_name": tokenizer.name_or_path}

        try:
            # Submit to Atropos
            response = self.session.post(f"{self.api_url}/evaluate_and_compute_advantages", json=payload, timeout=self.timeout)

            if response.status_code == 200:
                result = response.json()

                # Convert advantages to tensor
                advantages_list = result.get("advantages", [])
                if advantages_list:
                    # Pad sequences to same length
                    max_len = max(len(adv) for adv in advantages_list)
                    padded_advantages = []
                    for adv in advantages_list:
                        padded = adv + [0.0] * (max_len - len(adv))
                        padded_advantages.append(padded)

                    advantages_tensor = torch.tensor(padded_advantages, dtype=torch.float32)
                    result["advantages"] = advantages_tensor

                return result
            else:
                logger.error(f"Failed to get advantages: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error submitting responses: {e}")
            return None


class AtroposGRPOComputer:
    """
    GRPO advantage computer that integrates with Atropos environments.

    This implements the GRPO algorithm with token-level advantage overrides
    from Atropos environments.
    """

    def __init__(self, atropos_client: AtroposEnvironmentClient):
        self.client = atropos_client

    def compute_advantages_with_overrides(
        self, prompts: torch.Tensor, responses: torch.Tensor, old_log_probs: torch.Tensor, ref_log_probs: Optional[torch.Tensor], tokenizer: AutoTokenizer, metadata: Dict[str, Any], kl_coef: float = 0.1, use_token_level_overrides: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute GRPO advantages with Atropos environment overrides.

        Args:
            prompts: Prompt token tensor
            responses: Response token tensor
            old_log_probs: Log probs from current policy
            ref_log_probs: Log probs from reference policy (optional)
            tokenizer: Tokenizer for decoding
            metadata: Environment metadata
            kl_coef: KL penalty coefficient
            use_token_level_overrides: Whether to use token-level overrides

        Returns:
            advantages: Token-level advantages
            metrics: Dictionary of metrics
        """
        batch_size = prompts.shape[0]
        device = prompts.device

        # Decode prompts and responses
        prompt_texts = []
        response_texts = []

        for i in range(batch_size):
            prompt = tokenizer.decode(prompts[i], skip_special_tokens=True)
            response = tokenizer.decode(responses[i], skip_special_tokens=True)
            prompt_texts.append(prompt)
            response_texts.append(response)

        # Submit to Atropos and get environment feedback
        result = self.client.submit_responses_and_get_advantages(prompts=prompt_texts, responses=response_texts, metadata=metadata, tokenizer=tokenizer)

        if result is None:
            # Fallback to standard GRPO if Atropos fails
            logger.warning("Atropos evaluation failed, using standard GRPO")
            return self._compute_standard_grpo_advantages(old_log_probs, ref_log_probs, kl_coef), {}

        # Extract advantages and metrics
        advantages = result["advantages"].to(device)
        env_metrics = result.get("env_metrics", {})

        # Apply KL penalty if reference policy provided
        if ref_log_probs is not None:
            kl_div = old_log_probs - ref_log_probs
            kl_penalty = -kl_coef * kl_div

            if use_token_level_overrides:
                # Apply KL penalty only to non-overridden tokens
                # Atropos provides mask indicating which tokens have overrides
                override_mask = result.get("override_mask", torch.zeros_like(advantages))
                advantages = advantages + (1 - override_mask) * kl_penalty
            else:
                advantages = advantages + kl_penalty

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute metrics
        metrics = {"advantages/mean": advantages.mean().item(), "advantages/std": advantages.std().item(), "advantages/min": advantages.min().item(), "advantages/max": advantages.max().item(), **env_metrics}

        return advantages, metrics

    def _compute_standard_grpo_advantages(self, old_log_probs: torch.Tensor, ref_log_probs: Optional[torch.Tensor], kl_coef: float) -> torch.Tensor:
        """Fallback to standard GRPO advantage computation"""
        # Group-relative advantages
        batch_size, seq_len = old_log_probs.shape
        n_groups = batch_size // 8  # Assuming group size of 8

        advantages = torch.zeros_like(old_log_probs)

        for g in range(n_groups):
            start_idx = g * 8
            end_idx = (g + 1) * 8
            group_log_probs = old_log_probs[start_idx:end_idx]

            # Compute group mean
            group_mean = group_log_probs.mean(dim=0, keepdim=True)

            # Group-relative advantages
            group_advantages = group_log_probs - group_mean
            advantages[start_idx:end_idx] = group_advantages

        # Apply KL penalty if reference provided
        if ref_log_probs is not None:
            kl_div = old_log_probs - ref_log_probs
            advantages = advantages - kl_coef * kl_div

        return advantages


def create_atropos_grpo_trainer(config):
    """
    Factory function to create an Atropos-integrated GRPO trainer.

    This sets up the complete integration including:
    - Atropos client connection
    - GRPO advantage computer with overrides
    - Proper configuration for VeRL
    """
    # Initialize Atropos client
    atropos_url = config.get("atropos_url", "http://localhost:8000")
    client = AtroposEnvironmentClient(api_url=atropos_url)

    # Check connectivity
    if not client.health_check():
        raise ConnectionError(f"Cannot connect to Atropos at {atropos_url}. Please ensure Atropos server is running.")

    # Create GRPO computer
    grpo_computer = AtroposGRPOComputer(client)

    # Return configured components
    return {"client": client, "grpo_computer": grpo_computer, "config": {"use_token_level_overrides": config.get("use_token_level_overrides", True), "kl_coef": config.get("kl_coef", 0.1), "fallback_to_standard": config.get("fallback_to_standard", True)}}
