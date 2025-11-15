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
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests
import torch

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
            raise AtroposAPIError(f"Cannot connect to Atropos API at {self.config.api_url}: {e}") from e
    
    def submit_responses_and_get_advantages(
        self,
        prompts: list[str],
        responses: list[str], 
        metadata: Optional[dict[str, Any]] = None,
        response_mask: Optional[torch.Tensor] = None
    ) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
        """
        Submit responses to Atropos and get token-level advantages.
        
        Args:
            prompts: List of prompt strings
            responses: List of response strings
            metadata: Optional metadata for the batch
            response_mask: Mask tensor for valid response tokens
            
        Returns:
            Tuple of (advantages tensor, metrics dict)
        """
        # Prepare submission data
        submission_data = {
            "prompts": prompts,
            "responses": responses,
            "metadata": metadata or {}
        }
        
        last_error = None
        cumulative_wait_time = 0.0
        
        # Retry loop with exponential backoff and total wait time cap
        for attempt in range(self.config.retry_attempts):
            try:
                # Submit to Atropos
                response = self.session.post(
                    f"{self.config.api_url}/evaluate_and_compute_advantages",
                    json=submission_data,
                    timeout=self.config.timeout
                )
                
                if response.status_code != 200:
                    error_msg = f"Atropos API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    last_error = error_msg
                    
                    # Don't retry on client errors (4xx)
                    if 400 <= response.status_code < 500:
                        break
                    
                    # Wait before retrying on server errors
                    if attempt < self.config.retry_attempts - 1:
                        base_wait_time = self.config.retry_delay * (2 ** attempt)
                        # Add random jitter (0-1s) to prevent thundering herd
                        jitter = random.uniform(0, 1)
                        wait_time = base_wait_time + jitter
                        
                        # Check if we would exceed max_wait_time
                        if cumulative_wait_time + wait_time > self.config.max_wait_time:
                            logger.warning(
                                f"Would exceed max_wait_time ({self.config.max_wait_time}s), stopping retries"
                            )
                            break
                            
                        logger.info(
                            f"Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{self.config.retry_attempts})"
                        )
                        time.sleep(wait_time)
                        cumulative_wait_time += wait_time
                    continue
                
                # Parse response
                result = response.json()
                advantages = result.get("advantages", [])
                metrics = result.get("metrics", {})
                
                # Convert to tensor if we have response_mask  
                if response_mask is not None and advantages:
                    advantage_tensor = self._convert_to_token_level_advantages(
                        advantages, response_mask
                    )
                    return advantage_tensor, metrics
                
                return advantages, metrics
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Failed to communicate with Atropos: {e}"
                logger.error(error_msg)
                last_error = error_msg
                
                # Wait before retrying
                if attempt < self.config.retry_attempts - 1:
                    base_wait_time = self.config.retry_delay * (2 ** attempt)
                    # Add random jitter (0-1s) to prevent thundering herd
                    jitter = random.uniform(0, 1)
                    wait_time = base_wait_time + jitter
                    
                    # Check if we would exceed max_wait_time
                    if cumulative_wait_time + wait_time > self.config.max_wait_time:
                        logger.warning(
                            f"Would exceed max_wait_time ({self.config.max_wait_time}s), stopping retries"
                        )
                        break
                        
                    logger.info(
                        f"Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{self.config.retry_attempts})"
                    )
                    time.sleep(wait_time)
                    cumulative_wait_time += wait_time
        
        logger.error(f"All {self.config.retry_attempts} attempts failed. Last error: {last_error}")
        return None, {}
    
    def _convert_to_token_level_advantages(
        self, 
        advantages: list[float],
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """Convert response-level advantages to token-level using response mask"""
        batch_size, seq_len = response_mask.shape

        token_advantages = torch.zeros(
            batch_size,
            seq_len,
            dtype=torch.float32,
            device=response_mask.device,
        )

        for i, adv in enumerate(advantages):
            mask = response_mask[i].float()
            if isinstance(adv, (list, tuple)):
                adv_tensor = torch.as_tensor(adv, dtype=torch.float32, device=response_mask.device)
                if adv_tensor.ndim != 1:
                    raise ValueError(f"Unexpected advantage shape for sample {i}: {adv_tensor.shape}")
                length = min(seq_len, adv_tensor.shape[0])
                token_advantages[i, :length] = adv_tensor[:length]
                token_advantages[i] *= mask
            else:
                token_advantages[i] = float(adv) * mask

        return token_advantages


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
    ) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
        """
        Compute advantages with Atropos environment overrides.
        
        Args:
            prompts: Prompt token tensor
            responses: Response token tensor  
            scores: Initial scores from model
            tokenizer: Tokenizer for decoding
            response_mask: Mask for valid response tokens
            
        Returns:
            Tuple of (optional advantages tensor, metrics dict)
        """
        # Decode prompts and responses
        prompt_texts = [
            tokenizer.decode(p.detach().cpu().tolist(), skip_special_tokens=True)
            for p in prompts
        ]
        response_texts = []
        for resp, mask in zip(responses, response_mask):
            valid_length = int(mask.sum().item())
            tokens = resp[:valid_length] if valid_length > 0 else resp
            response_texts.append(
                tokenizer.decode(tokens.detach().cpu().tolist(), skip_special_tokens=True)
            )
        
        # Try to get advantages from Atropos
        advantages, metrics = self.client.submit_responses_and_get_advantages(
            prompt_texts, response_texts, response_mask=response_mask
        )
        
        if advantages is not None and self.config.use_advantages:
            # Use Atropos advantages
            logger.info("Using advantages from Atropos environments")
            # Ensure correct shape and device
            if advantages.shape[0] != responses.shape[0]:
                logger.warning(
                    f"Advantage batch size mismatch: expected {responses.shape[0]}, got {advantages.shape[0]}"
                )
                return self._compute_fallback_advantages()
            if advantages.shape[1] != response_mask.shape[1]:
                logger.warning(
                    f"Advantage sequence length mismatch: expected {response_mask.shape[1]}, "
                    f"got {advantages.shape[1]}"
                )
                return self._compute_fallback_advantages()
            
            # Move to correct device and dtype
            target_dtype = scores.dtype if scores is not None else torch.float32
            advantages = advantages.to(device=responses.device, dtype=target_dtype)
            return advantages, metrics
        
        # Fallback to standard computation
        if self.config.fallback_to_standard:
            logger.info("Using fallback advantage computation")
            return self._compute_fallback_advantages()
        
        raise AtroposAPIError("Failed to get advantages from Atropos and fallback disabled")
    
    def _compute_fallback_advantages(
        self,
    ) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
        """Signal that standard GRPO computation should be used instead."""
        return None, {"fallback": True}
