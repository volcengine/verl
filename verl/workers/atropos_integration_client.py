#!/usr/bin/env python3
"""
Simplified Atropos Client
========================

Clean, simple interface for VeRL-Atropos integration that matches the improved architecture.
Replaces complex registration/endpoint management with direct trajectory batch requests.
"""

import logging
import requests
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryBatch:
    """Clean data structure for trajectory batches"""
    tokens: torch.Tensor
    masks: torch.Tensor  
    scores: torch.Tensor
    groups: List[int]
    metadata: Dict[str, Any]


class SimpleAtroposClient:
    """
    Simplified Atropos client with clean API interface.
    
    No complex registration - just direct trajectory batch requests.
    """
    
    def __init__(self, api_url: str = "http://localhost:8000", timeout: int = 30):
        self.api_url = api_url
        self.timeout = timeout
        self.session = requests.Session()
        
        # Test connectivity on initialization
        self._test_connectivity()
        logger.info(f"SimpleAtroposClient initialized: {api_url}")
    
    def _test_connectivity(self) -> None:
        """Test API connectivity"""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"API health check failed: {response.status_code}")
        except requests.RequestException as e:
            raise ConnectionError(f"Cannot connect to Atropos API at {self.api_url}: {e}")
    
    def get_trajectory_batch(
        self, 
        batch_size: int,
        environment: str = "gsm8k",
        model_config: Optional[Dict] = None,
        inference_endpoint: Optional[str] = None
    ) -> TrajectoryBatch:
        """
        Get a trajectory batch from Atropos environments.
        
        Args:
            batch_size: Number of trajectories to generate
            environment: Environment name (gsm8k, math, code, etc.)
            model_config: Model configuration for generation
            inference_endpoint: VeRL inference server endpoint
            
        Returns:
            TrajectoryBatch with tokens, masks, scores, and groups
        """
        request_data = {
            "batch_size": batch_size,
            "environment": environment,
            "model_config": model_config or self._default_model_config(),
            "inference_endpoint": inference_endpoint
        }
        
        logger.info(f"Requesting trajectory batch: {batch_size} samples from {environment}")
        
        try:
            response = self.session.post(
                f"{self.api_url}/trajectory_batch",
                json=request_data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_trajectory_response(data)
            else:
                raise RuntimeError(f"Trajectory request failed: {response.status_code} - {response.text}")
                
        except requests.RequestException as e:
            raise RuntimeError(f"Trajectory request failed: {e}")
    
    def get_multi_environment_batch(
        self,
        total_batch_size: int,
        environment_weights: Dict[str, float],
        model_config: Optional[Dict] = None,
        inference_endpoint: Optional[str] = None
    ) -> TrajectoryBatch:
        """
        Get trajectories from multiple environments based on weights.
        
        Args:
            total_batch_size: Total number of trajectories
            environment_weights: Dict of environment_name -> weight
            model_config: Model configuration
            inference_endpoint: VeRL inference server endpoint
            
        Returns:
            Combined TrajectoryBatch from multiple environments
        """
        # Calculate batch sizes for each environment
        env_batches = self._calculate_environment_batches(total_batch_size, environment_weights)
        
        # Collect batches from each environment
        all_batches = []
        for env_name, env_batch_size in env_batches.items():
            if env_batch_size > 0:
                batch = self.get_trajectory_batch(
                    batch_size=env_batch_size,
                    environment=env_name,
                    model_config=model_config,
                    inference_endpoint=inference_endpoint
                )
                all_batches.append(batch)
        
        # Combine batches
        return self._combine_trajectory_batches(all_batches)
    
    def _default_model_config(self) -> Dict:
        """Default model configuration for generation"""
        return {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
            "do_sample": True
        }
    
    def _parse_trajectory_response(self, data: Dict) -> TrajectoryBatch:
        """Parse API response into TrajectoryBatch"""
        try:
            # Convert lists to tensors
            tokens = torch.tensor(data["tokens"], dtype=torch.long)
            masks = torch.tensor(data["masks"], dtype=torch.bool)
            scores = torch.tensor(data["scores"], dtype=torch.float32)
            groups = data["groups"]
            metadata = data.get("metadata", {})
            
            logger.info(f"Parsed trajectory batch: {tokens.shape[0]} samples")
            
            return TrajectoryBatch(
                tokens=tokens,
                masks=masks,
                scores=scores,
                groups=groups,
                metadata=metadata
            )
            
        except KeyError as e:
            raise ValueError(f"Invalid trajectory response format: missing {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse trajectory response: {e}")
    
    def _calculate_environment_batches(
        self, 
        total_batch_size: int, 
        environment_weights: Dict[str, float]
    ) -> Dict[str, int]:
        """Calculate batch sizes for each environment based on weights"""
        # Normalize weights
        total_weight = sum(environment_weights.values())
        normalized_weights = {env: weight/total_weight for env, weight in environment_weights.items()}
        
        # Calculate batch sizes
        env_batches = {}
        remaining_batch_size = total_batch_size
        
        for env_name, weight in normalized_weights.items():
            if remaining_batch_size <= 0:
                env_batches[env_name] = 0
            else:
                batch_size = int(total_batch_size * weight)
                env_batches[env_name] = min(batch_size, remaining_batch_size)
                remaining_batch_size -= batch_size
        
        # Distribute any remaining samples
        if remaining_batch_size > 0:
            for env_name in env_batches:
                if remaining_batch_size <= 0:
                    break
                env_batches[env_name] += 1
                remaining_batch_size -= 1
        
        return env_batches
    
    def _combine_trajectory_batches(self, batches: List[TrajectoryBatch]) -> TrajectoryBatch:
        """Combine multiple trajectory batches into one"""
        if not batches:
            raise ValueError("No batches to combine")
        
        if len(batches) == 1:
            return batches[0]
        
        # Combine tensors
        combined_tokens = torch.cat([batch.tokens for batch in batches], dim=0)
        combined_masks = torch.cat([batch.masks for batch in batches], dim=0)
        combined_scores = torch.cat([batch.scores for batch in batches], dim=0)
        
        # Combine groups (adjust indices for concatenation)
        combined_groups = []
        group_offset = 0
        for batch in batches:
            adjusted_groups = [g + group_offset for g in batch.groups]
            combined_groups.extend(adjusted_groups)
            group_offset += max(batch.groups) + 1 if batch.groups else 0
        
        # Combine metadata
        combined_metadata = {
            "source_environments": [batch.metadata.get("environment", "unknown") for batch in batches],
            "batch_sizes": [batch.tokens.shape[0] for batch in batches]
        }
        
        return TrajectoryBatch(
            tokens=combined_tokens,
            masks=combined_masks,
            scores=combined_scores,
            groups=combined_groups,
            metadata=combined_metadata
        )


class AtroposEnvironmentManager:
    """
    Manages multiple Atropos environments with weighted sampling.
    """
    
    def __init__(self, client: SimpleAtroposClient, environment_config: Dict[str, Any]):
        self.client = client
        self.environment_weights = environment_config.get("weights", {"gsm8k": 1.0})
        self.default_batch_size = environment_config.get("default_batch_size", 64)
        
        logger.info(f"AtroposEnvironmentManager initialized with environments: {list(self.environment_weights.keys())}")
    
    def get_training_batch(
        self, 
        batch_size: Optional[int] = None,
        inference_endpoint: Optional[str] = None
    ) -> TrajectoryBatch:
        """Get a training batch from configured environments"""
        batch_size = batch_size or self.default_batch_size
        
        if len(self.environment_weights) == 1:
            # Single environment - direct call
            env_name = list(self.environment_weights.keys())[0]
            return self.client.get_trajectory_batch(
                batch_size=batch_size,
                environment=env_name,
                inference_endpoint=inference_endpoint
            )
        else:
            # Multiple environments - weighted sampling
            return self.client.get_multi_environment_batch(
                total_batch_size=batch_size,
                environment_weights=self.environment_weights,
                inference_endpoint=inference_endpoint
            ) 