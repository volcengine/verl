# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Atropos-VeRL Integration Module

This module provides the core integration between VeRL and Atropos environments,
enabling training with environment feedback and token-level advantages.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
import requests
import torch
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class AtroposAPIError(Exception):
    """Raised when Atropos API operations fail"""
    pass


@dataclass
class AtroposConfig:
    """Configuration for Atropos integration"""
    api_url: str = "http://localhost:9001"
    timeout: int = 30
    retry_attempts: int = 10
    retry_delay: float = 0.5
    max_wait_time: float = 30.0
    use_advantages: bool = True
    fallback_to_standard: bool = True


class AtroposEnvironmentClient:
    """
    Client for interacting with Atropos environments.
    
    This client communicates with the Atropos API server to:
    - Submit generated responses for evaluation
    - Retrieve token-level advantages from environments
    - Handle retry logic and error cases
    """
    
    def __init__(self, config: AtroposConfig):
        self.config = config
        self.session = requests.Session()
        self._test_connectivity()
        
    def _test_connectivity(self):
        """Test if Atropos API is reachable"""
        try:
            response = self.session.get(f"{self.config.api_url}/health", timeout=5)
            if response.status_code != 200:
                raise AtroposAPIError(f"Atropos API health check failed: {response.status_code}")
            logger.info(f"Connected to Atropos API at {self.config.api_url}")
        except requests.exceptions.RequestException as e:
            raise AtroposAPIError(f"Cannot connect to Atropos API at {self.config.api_url}: {e}")
    
    def submit_responses_and_get_advantages(
        self,
        prompts: List[str],
        responses: List[str], 
        metadata: Optional[Dict[str, Any]] = None,
        tokenizer = None
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Submit responses to Atropos and get token-level advantages.
        
        Args:
            prompts: List of prompt strings
            responses: List of response strings
            metadata: Optional metadata for the batch
            tokenizer: Tokenizer for converting advantages to token level
            
        Returns:
            Tuple of (advantages tensor, metrics dict)
        """
        # Prepare submission data
        submission_data = {
            "prompts": prompts,
            "responses": responses,
            "metadata": metadata or {}
        }
        
        try:
            # Submit to Atropos
            response = self.session.post(
                f"{self.config.api_url}/evaluate_and_compute_advantages",
                json=submission_data,
                timeout=self.config.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Atropos API error: {response.status_code} - {response.text}")
                return None, {}
            
            # Parse response
            result = response.json()
            advantages = result.get("advantages", [])
            metrics = result.get("metrics", {})
            
            # Convert to tensor if we have tokenizer
            if tokenizer and advantages:
                advantage_tensor = self._convert_to_token_level_advantages(
                    advantages, responses, tokenizer
                )
                return advantage_tensor, metrics
            
            return advantages, metrics
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to communicate with Atropos: {e}")
            return None, {}
    
    def _convert_to_token_level_advantages(
        self, 
        advantages: List[float],
        responses: List[str],
        tokenizer
    ) -> torch.Tensor:
        """Convert response-level advantages to token-level"""
        token_advantages = []
        
        for adv, response in zip(advantages, responses):
            # Tokenize response
            tokens = tokenizer.encode(response, add_special_tokens=False)
            # Broadcast advantage to all tokens
            token_adv = torch.full((len(tokens),), adv, dtype=torch.float32)
            token_advantages.append(token_adv)
        
        # Pad to same length
        max_len = max(len(adv) for adv in token_advantages)
        padded = []
        for adv in token_advantages:
            if len(adv) < max_len:
                padding = torch.zeros(max_len - len(adv))
                adv = torch.cat([adv, padding])
            padded.append(adv)
        
        return torch.stack(padded)


class AtroposGRPOComputer:
    """
    Computes GRPO advantages with optional Atropos environment overrides.
    
    This class integrates with Atropos to get environment-specific advantages
    while maintaining compatibility with standard GRPO computation.
    """
    
    def __init__(self, config: AtroposConfig):
        self.config = config
        self.client = AtroposEnvironmentClient(config)
        
    def compute_advantages_with_overrides(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        scores: torch.Tensor,
        tokenizer,
        response_mask: torch.Tensor,
        fallback_estimator = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute advantages with Atropos environment overrides.
        
        Args:
            prompts: Prompt token tensor
            responses: Response token tensor  
            scores: Initial scores from model
            tokenizer: Tokenizer for decoding
            response_mask: Mask for valid response tokens
            fallback_estimator: Fallback advantage estimator
            
        Returns:
            Tuple of (advantages tensor, metrics dict)
        """
        # Decode prompts and responses
        prompt_texts = [tokenizer.decode(p, skip_special_tokens=True) for p in prompts]
        response_texts = [tokenizer.decode(r, skip_special_tokens=True) for r in responses]
        
        # Try to get advantages from Atropos
        advantages, metrics = self.client.submit_responses_and_get_advantages(
            prompt_texts, response_texts, tokenizer=tokenizer
        )
        
        if advantages is not None and self.config.use_advantages:
            # Use Atropos advantages
            logger.info("Using advantages from Atropos environments")
            # Ensure correct shape and device
            if advantages.shape[0] != responses.shape[0]:
                logger.warning("Advantage shape mismatch, using fallback")
                return self._compute_fallback_advantages(
                    scores, response_mask, fallback_estimator
                )
            return advantages, metrics
        
        # Fallback to standard computation
        if self.config.fallback_to_standard:
            logger.info("Using fallback advantage computation")
            return self._compute_fallback_advantages(
                scores, response_mask, fallback_estimator
            )
        
        raise AtroposAPIError("Failed to get advantages from Atropos and fallback disabled")
    
    def _compute_fallback_advantages(
        self,
        scores: torch.Tensor,
        response_mask: torch.Tensor,
        fallback_estimator
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute advantages using fallback method"""
        if fallback_estimator is None:
            # Simple score-based advantages
            advantages = scores.unsqueeze(-1) * response_mask
            return advantages, {"fallback": True}
        
        # Use provided estimator
        return fallback_estimator(scores, response_mask), {"fallback": True}