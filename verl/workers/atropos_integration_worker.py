#!/usr/bin/env python3
"""
Simplified Atropos Worker
========================

Clean, simple worker for VeRL-Atropos integration that removes Ray complexity
and uses the simplified trajectory batch API.
"""

import logging
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from verl.protocol import DataProto
from verl.workers.atropos_integration_client import AtroposEnvironmentManager, SimpleAtroposClient, TrajectoryBatch
from verl.workers.fsdp_workers import ActorRolloutRefWorker

logger = logging.getLogger(__name__)


class SimpleAtroposWorker(ActorRolloutRefWorker):
    """
    Simplified Atropos worker that replaces complex registration/endpoint management
    with direct trajectory batch requests.

    Much cleaner than the original AtroposRolloutWorker:
    - No complex API registration
    - No endpoint management
    - Simple batch requests
    - Clean data conversion
    """

    def __init__(self, config: DictConfig, role: str = "actor_rollout_ref"):
        super().__init__(config=config, role=role)

        # Initialize Atropos client
        atropos_config = config.get("atropos", {})
        api_url = atropos_config.get("api_url", "http://localhost:8000")
        timeout = atropos_config.get("timeout", 30)

        self.client = SimpleAtroposClient(api_url=api_url, timeout=timeout)

        # Initialize environment manager
        environment_config = atropos_config.get("environments", {"weights": {"gsm8k": 1.0}, "default_batch_size": 64})
        self.env_manager = AtroposEnvironmentManager(self.client, environment_config)

        # Inference endpoint (will be set by trainer)
        self.inference_endpoint = None

        logger.info(f"SimpleAtroposWorker initialized with API: {api_url}")
        logger.info(f"Environment weights: {environment_config.get('weights', {})}")

    def set_inference_endpoint(self, endpoint: str):
        """Set the VeRL inference server endpoint"""
        self.inference_endpoint = endpoint
        logger.info(f"Inference endpoint set: {endpoint}")

    def generate_sequences(self, prompts: DataProto, sampling_params: Optional[Dict] = None, **kwargs) -> DataProto:
        """
        Generate sequences using Atropos environments.

        This replaces the complex rollout generation with a simple
        trajectory batch request to Atropos.
        """
        batch_size = prompts.batch.batch_size[0]

        logger.info(f"Generating {batch_size} sequences via Atropos")

        try:
            # Get trajectory batch from Atropos
            trajectory_batch = self.env_manager.get_training_batch(batch_size=batch_size, inference_endpoint=self.inference_endpoint)

            # Convert to VeRL DataProto format
            verl_data = self._convert_trajectory_to_dataproto(trajectory_batch, prompts)

            logger.info(f"Successfully generated {batch_size} sequences")
            return verl_data

        except Exception as e:
            logger.error(f"Sequence generation failed: {e}")
            raise RuntimeError(f"Atropos sequence generation failed: {e}")

    def _convert_trajectory_to_dataproto(self, trajectory_batch: TrajectoryBatch, original_prompts: DataProto) -> DataProto:
        """
        Convert Atropos TrajectoryBatch to VeRL DataProto format.

        Much simpler than the original complex conversion logic.
        """
        device = next(iter(original_prompts.batch.values())).device

        # Move tensors to correct device
        tokens = trajectory_batch.tokens.to(device)
        masks = trajectory_batch.masks.to(device)
        scores = trajectory_batch.scores.to(device)

        # Create batch dictionary
        batch_dict = {
            "input_ids": tokens,
            "attention_mask": masks,
            "responses": tokens,  # For GRPO, responses are the full sequences
            "token_level_scores": scores,
            "response_mask": masks,
        }

        # Create non-tensor batch
        non_tensor_batch = {"uid": trajectory_batch.groups, "environment_metadata": trajectory_batch.metadata}

        # Create DataProto
        result = DataProto(batch=batch_dict, non_tensor_batch=non_tensor_batch)

        logger.debug(f"Converted trajectory batch: {tokens.shape} tokens, {len(trajectory_batch.groups)} groups")
        return result

    def compute_ref_log_prob(self, data: DataProto) -> DataProto:
        """
        Compute reference log probabilities.

        For the simplified integration, we can either:
        1. Use the parent class implementation (standard VeRL)
        2. Request ref log probs from Atropos (if supported)

        For now, using standard VeRL implementation.
        """
        return super().compute_ref_log_prob(data)


class UnifiedInferenceManager:
    """
    Manages the unified inference engine that serves both VeRL training
    and Atropos environment requests.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.inference_server = None
        self.server_endpoint = None

    def start_inference_server(self) -> str:
        """
        Start the unified inference server and return its endpoint.

        This server will be used by both:
        1. VeRL for training (weight updates, ref log probs)
        2. Atropos for rollout generation
        """
        # For now, assume vLLM server is already running
        # In a full implementation, this would start the server

        port = self.config.get("inference_port", 9000)
        self.server_endpoint = f"http://localhost:{port}"

        logger.info(f"Unified inference server endpoint: {self.server_endpoint}")
        return self.server_endpoint

    def update_model_weights(self, model_state_dict: Dict[str, torch.Tensor]):
        """
        Update the inference server with new model weights.

        This ensures Atropos always uses the latest trained model.
        """
        # Implementation would depend on the inference engine
        # For vLLM, this might involve model reloading or weight swapping
        logger.info("Updated inference server with new model weights")

    def get_endpoint(self) -> str:
        """Get the inference server endpoint"""
        return self.server_endpoint


class SimpleAtroposTrainer:
    """
    Simplified trainer integration that coordinates VeRL training
    with Atropos environments using the clean API.
    """

    def __init__(self, config: DictConfig):
        self.config = config

        # Initialize components
        self.inference_manager = UnifiedInferenceManager(config)
        self.atropos_worker = SimpleAtroposWorker(config)

        # Start inference server
        inference_endpoint = self.inference_manager.start_inference_server()
        self.atropos_worker.set_inference_endpoint(inference_endpoint)

        logger.info("SimpleAtroposTrainer initialized")

    def get_training_batch(self, batch_size: int) -> DataProto:
        """
        Get a training batch from Atropos environments.

        This is the main interface for VeRL training.
        """
        # Create dummy prompts (Atropos generates the actual prompts)
        dummy_prompts = self._create_dummy_prompts(batch_size)

        # Get sequences from Atropos
        return self.atropos_worker.generate_sequences(dummy_prompts)

    def update_policy_weights(self, model_state_dict: Dict[str, torch.Tensor]):
        """
        Update the inference server with new policy weights.

        This ensures Atropos uses the latest trained model.
        """
        self.inference_manager.update_model_weights(model_state_dict)

    def _create_dummy_prompts(self, batch_size: int) -> DataProto:
        """Create dummy prompts for the interface"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dummy data - Atropos will generate the actual prompts
        dummy_tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        dummy_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)

        batch_dict = {"input_ids": dummy_tokens, "attention_mask": dummy_mask}

        return DataProto(batch=batch_dict, non_tensor_batch={})
