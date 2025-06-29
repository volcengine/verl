"""
GRPO Trainer with Atropos Integration

This implements a GRPO trainer that gets real advantages from Atropos environments
instead of using mock/heuristic values.
"""

import logging
from typing import Dict, List, Optional, Any
import torch
import numpy as np

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.debug import marked_timer

from .atropos_integration import AtroposEnvironmentClient, AtroposGRPOComputer

logger = logging.getLogger(__name__)


class RayGRPOAtroposTrainer(RayPPOTrainer):
    """
    GRPO trainer integrated with Atropos environments.
    
    Key features:
    - Gets prompts from Atropos environments
    - Submits generated responses for evaluation
    - Receives token-level advantages based on task performance
    - Implements GRPO with real environment feedback
    """
    
    def __init__(self, config, role_worker_mapping, resource_pool_manager):
        super().__init__(config, role_worker_mapping, resource_pool_manager)
        
        # Initialize Atropos integration
        atropos_config = config.trainer.get("atropos", {})
        self.atropos_client = AtroposEnvironmentClient(
            api_url=atropos_config.get("api_url", "http://localhost:8000"),
            timeout=atropos_config.get("timeout", 30)
        )
        
        # Check connectivity
        if not self.atropos_client.health_check():
            raise ConnectionError(
                f"Cannot connect to Atropos at {self.atropos_client.api_url}. "
                "Please start Atropos server first."
            )
            
        # Initialize GRPO computer
        self.grpo_computer = AtroposGRPOComputer(self.atropos_client)
        
        # GRPO-specific config
        self.grpo_config = {
            "kl_coef": config.algorithm.get("kl_coef", 0.1),
            "use_token_level_overrides": config.algorithm.get("use_token_level_overrides", True),
            "group_size": config.algorithm.get("group_size", 8),
            "normalize_advantages": config.algorithm.get("normalize_advantages", True)
        }
        
        logger.info(f"Initialized GRPO-Atropos trainer with config: {self.grpo_config}")
        
    def _get_prompts_from_atropos(self, batch_size: int) -> Optional[DataProto]:
        """Get prompts from Atropos environments"""
        result = self.atropos_client.get_prompts(batch_size)
        
        if result is None:
            return None
            
        # Convert to DataProto format
        prompts = result["prompts"]
        metadata = result["metadata"]
        
        # Tokenize prompts
        tokenized = []
        for prompt in prompts:
            tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.data.max_prompt_length
            )
            tokenized.append(tokens)
            
        # Create batch
        batch_dict = {
            "input_ids": torch.cat([t.input_ids for t in tokenized], dim=0),
            "attention_mask": torch.cat([t.attention_mask for t in tokenized], dim=0),
        }
        
        # Add metadata
        non_tensor_batch = {
            "raw_prompts": prompts,
            "atropos_metadata": metadata,
            "group_ids": result.get("group_ids", [])
        }
        
        return DataProto.from_dict(batch_dict, non_tensor_batch)
        
    def _compute_advantages_grpo(self, batch: DataProto) -> torch.Tensor:
        """
        Compute GRPO advantages using Atropos environment feedback.
        
        This is where the real integration happens:
        1. Extract generated responses
        2. Submit to Atropos for evaluation
        3. Get token-level advantages based on task performance
        4. Apply GRPO transformations
        """
        # Extract data
        input_ids = batch.batch["input_ids"]
        responses = batch.batch["responses"]
        old_log_probs = batch.batch.get("old_log_probs")
        ref_log_probs = batch.batch.get("ref_log_probs")
        
        # Get metadata
        metadata = batch.non_tensor_batch.get("atropos_metadata", {})
        
        # Compute advantages with Atropos
        advantages, metrics = self.grpo_computer.compute_advantages_with_overrides(
            prompts=input_ids[:, :-responses.shape[1]],  # Extract prompt portion
            responses=responses,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            tokenizer=self.tokenizer,
            metadata=metadata,
            kl_coef=self.grpo_config["kl_coef"],
            use_token_level_overrides=self.grpo_config["use_token_level_overrides"]
        )
        
        # Log metrics
        for key, value in metrics.items():
            self.metrics[f"grpo/{key}"] = value
            
        return advantages
        
    def training_step(self, batch_dict: Dict) -> Dict:
        """
        Override training step to use Atropos-provided prompts and advantages.
        """
        metrics = {}
        timing_raw = {}
        
        # Option 1: Use Atropos-provided prompts
        if self.config.trainer.get("use_atropos_prompts", True):
            batch_size = self.config.actor_rollout_ref.rollout.batch_size
            batch = self._get_prompts_from_atropos(batch_size)
            if batch is None:
                logger.warning("Failed to get prompts from Atropos, using dataset")
                batch = DataProto.from_single_dict(batch_dict)
        else:
            # Option 2: Use dataset prompts but still get advantages from Atropos
            batch = DataProto.from_single_dict(batch_dict)
            
        with marked_timer("training_step", timing_raw):
            # Generate responses
            with marked_timer("generation", timing_raw):
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask"],
                    non_tensor_batch_keys=["raw_prompts"]
                )
                
                # Generate multiple responses per prompt for GRPO
                gen_output = self.actor_rollout_wg.generate_sequences(
                    gen_batch,
                    n=self.grpo_config["group_size"]
                )
                
                # Merge back
                batch = batch.repeat(self.grpo_config["group_size"], interleave=True)
                batch = batch.union(gen_output)
                
            # Compute advantages using Atropos
            with marked_timer("advantages", timing_raw):
                advantages = self._compute_advantages_grpo(batch)
                batch.batch["advantages"] = advantages
                
            # Compute old log probs
            with marked_timer("old_log_probs", timing_raw):
                old_log_probs = self.actor_rollout_wg.compute_log_prob(batch)
                batch = batch.union(old_log_probs)
                
            # Reference log probs if using
            if self.use_reference_policy:
                with marked_timer("ref_log_probs", timing_raw):
                    ref_log_probs = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_probs)
                    
            # GRPO policy update
            with marked_timer("policy_update", timing_raw):
                actor_output = self.actor_rollout_wg.update_actor(batch)
                metrics.update(actor_output.meta_info.get("metrics", {}))
                
        # Update timing
        metrics["timing/total"] = timing_raw.get("training_step", 0)
        metrics.update({f"timing/{k}": v for k, v in timing_raw.items()})
        
        return metrics
        
    def _log_atropos_metrics(self, batch: DataProto) -> Dict[str, float]:
        """Extract and log Atropos-specific metrics"""
        metrics = {}
        
        # Get Atropos metadata
        metadata = batch.non_tensor_batch.get("atropos_metadata", {})
        
        # Environment-specific metrics
        if "env_name" in metadata:
            env_name = metadata["env_name"]
            
            # Extract scores if available
            if "scores" in batch.batch:
                scores = batch.batch["scores"]
                metrics[f"atropos/{env_name}/mean_score"] = scores.mean().item()
                metrics[f"atropos/{env_name}/correct_rate"] = (scores > 0).float().mean().item()
                
            # Extract advantages statistics by environment
            if "advantages" in batch.batch:
                advantages = batch.batch["advantages"]
                response_mask = batch.batch.get("response_mask", torch.ones_like(advantages))
                masked_advantages = advantages * response_mask
                
                metrics[f"atropos/{env_name}/advantage_mean"] = masked_advantages.sum() / response_mask.sum()
                metrics[f"atropos/{env_name}/advantage_std"] = masked_advantages[response_mask.bool()].std().item()
                
        return metrics


def create_grpo_atropos_trainer(config):
    """
    Factory function to create a GRPO trainer with Atropos integration.
    
    Args:
        config: VeRL configuration
        
    Returns:
        Configured GRPO-Atropos trainer class
    """
    return RayGRPOAtroposTrainer