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
Atropos API Client for VeRL Integration

This module provides a clean interface to interact with Atropos environments
through the Atropos API server.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union
import requests
from dataclasses import dataclass
import numpy as np
import torch

logger = logging.getLogger(__name__)


class AtroposAPIError(Exception):
    """Raised when Atropos API operations fail"""
    pass


@dataclass
class AtroposConfig:
    """Configuration for Atropos API client"""
    api_url: str = "http://localhost:9001"
    timeout: int = 30
    batch_size: int = 4
    max_token_len: int = 512
    wandb_group: str = "verl_atropos_integration"
    wandb_project: str = "verl_grpo_atropos"
    checkpoint_dir: str = "/tmp/verl_checkpoints"
    save_checkpoint_interval: int = 100
    retry_attempts: int = 10
    retry_delay: float = 0.5
    max_wait_time: float = 30.0


class AtroposAPIClient:
    """
    Client for interacting with Atropos API server.
    
    This client handles:
    - Trainer registration
    - Data submission to environments
    - Batch retrieval with retry logic
    - Environment coordination
    """
    
    def __init__(self, config: AtroposConfig):
        self.config = config
        self.trainer_uuid = None
        self.registered = False
        self.current_step = 0
        
    def test_connectivity(self) -> bool:
        """Test if Atropos API server is reachable"""
        try:
            response = requests.get(
                f"{self.config.api_url}/status",
                timeout=self.config.timeout
            )
            if response.status_code == 200:
                logger.info(f"Atropos API server is reachable at {self.config.api_url}")
                return True
            else:
                logger.error(f"Atropos API returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Atropos API: {e}")
            return False
            
    def register_trainer(self, starting_step: int = 0, num_steps: int = 1000) -> bool:
        """Register this trainer with Atropos API"""
        if self.registered:
            return True
            
        registration_data = {
            "wandb_group": self.config.wandb_group,
            "wandb_project": self.config.wandb_project,
            "batch_size": self.config.batch_size,
            "max_token_len": self.config.max_token_len,
            "checkpoint_dir": self.config.checkpoint_dir,
            "save_checkpoint_interval": self.config.save_checkpoint_interval,
            "starting_step": starting_step,
            "num_steps": num_steps
        }
        
        try:
            response = requests.post(
                f"{self.config.api_url}/register",
                json=registration_data,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                self.trainer_uuid = data.get('uuid')
                self.registered = True
                self.current_step = starting_step
                logger.info(f"Trainer registered with UUID: {self.trainer_uuid}")
                return True
            else:
                raise AtroposAPIError(f"Registration failed: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise AtroposAPIError(f"Registration request failed: {e}")
            
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about registered environments"""
        try:
            response = requests.get(
                f"{self.config.api_url}/envs",
                timeout=self.config.timeout
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get environment info: {response.status_code}")
                return {"envs": []}
        except:
            return {"envs": []}
            
    def submit_scored_data(
        self,
        tokens: List[List[int]],
        masks: List[List[int]],
        scores: List[float],
        ref_logprobs: Optional[List[List[float]]] = None,
        messages: Optional[List[List[Dict[str, Any]]]] = None,
        advantages: Optional[List[List[float]]] = None,
        overrides: Optional[List[dict]] = None,
        group_overrides: Optional[dict] = None,
        images: Optional[Any] = None
    ) -> bool:
        """
        Submit scored data to Atropos for processing by environments.
        
        Args:
            tokens: List of token sequences
            masks: List of masks (0 for prompt, 1 for response)
            scores: List of scores for each sequence
            ref_logprobs: Optional reference log probabilities
            messages: Optional chat messages
            advantages: Optional pre-computed advantages
            overrides: Optional per-sequence advantage overrides
            group_overrides: Optional group-level advantage overrides
            images: Optional image data
            
        Returns:
            True if submission successful
        """
        scored_data = {
            "tokens": tokens,
            "masks": masks,
            "scores": scores,
            "ref_logprobs": ref_logprobs,
            "messages": messages,
            "advantages": advantages,
            "overrides": overrides,
            "group_overrides": group_overrides,
            "images": images
        }
        
        # Remove None values
        scored_data = {k: v for k, v in scored_data.items() if v is not None}
        
        try:
            response = requests.post(
                f"{self.config.api_url}/scored_data",
                json=scored_data,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully submitted {len(tokens)} sequences to Atropos")
                return True
            else:
                logger.error(f"Data submission failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Data submission request failed: {e}")
            return False
            
    def retrieve_batch(self) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve a processed batch from Atropos with retry logic.
        
        Returns:
            Batch data if available, None otherwise
        """
        start_time = time.time()
        delay = self.config.retry_delay
        
        for attempt in range(self.config.retry_attempts):
            elapsed = time.time() - start_time
            if elapsed > self.config.max_wait_time:
                logger.warning(f"Max wait time exceeded ({self.config.max_wait_time}s)")
                break
                
            try:
                # Check queue status first
                status_response = requests.get(
                    f"{self.config.api_url}/status",
                    timeout=5
                )
                if status_response.status_code == 200:
                    status = status_response.json()
                    queue_size = status.get('queue_size', 0)
                    logger.debug(f"Queue status - Size: {queue_size}, Step: {status.get('current_step', 0)}")
                
                # Try to get batch
                response = requests.get(
                    f"{self.config.api_url}/batch",
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    batch = data.get('batch')
                    
                    if batch is not None and len(batch) > 0:
                        logger.info(f"Retrieved batch with {len(batch)} items (attempt {attempt + 1})")
                        self.current_step += 1
                        return batch
                    else:
                        logger.debug(f"Batch unavailable (attempt {attempt + 1})")
                        
                else:
                    logger.warning(f"Batch request failed with status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Batch retrieval error on attempt {attempt + 1}: {e}")
                
            # Wait before retry (except on last attempt)
            if attempt < self.config.retry_attempts - 1:
                time.sleep(delay)
                delay = min(delay * 1.5, 3.0)  # Exponential backoff with cap
                
        logger.warning(f"No batch available after {self.config.retry_attempts} attempts")
        return None
        
    def process_rollout_data(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        tokenizer,
        scores: Optional[List[float]] = None,
        log_probs: Optional[torch.Tensor] = None,
        ref_model=None
    ) -> Optional[Dict[str, Any]]:
        """
        Process rollout data and submit to Atropos, then retrieve advantages.
        
        Args:
            prompts: Prompt token tensor
            responses: Response token tensor
            tokenizer: Tokenizer for decoding
            scores: Optional pre-computed scores
            log_probs: Optional log probabilities from generation
            ref_model: Optional reference model for computing ref_logprobs
            
        Returns:
            Dictionary containing advantages and other data, or None if failed
        """
        batch_size = prompts.shape[0]
        prompt_len = prompts.shape[1]
        response_len = responses.shape[1]
        
        # Prepare token sequences and masks
        tokens_list = []
        masks_list = []
        ref_logprobs_list = []
        
        for i in range(batch_size):
            # Combine prompt and response
            full_sequence = torch.cat([prompts[i], responses[i]], dim=0)
            tokens = full_sequence.tolist()
            
            # Create mask (0 for prompt, 1 for response)
            mask = [0] * prompt_len + [1] * response_len
            
            tokens_list.append(tokens)
            masks_list.append(mask)
            
            # Compute reference log probabilities if model provided
            if ref_model is not None:
                ref_logprobs = self._compute_reference_logprobs(full_sequence, ref_model)
                ref_logprobs_list.append(ref_logprobs)
        
        # If no scores provided, compute them
        if scores is None:
            scores = self._compute_default_scores(responses, log_probs, tokenizer)
            
        # Submit to Atropos
        success = self.submit_scored_data(
            tokens=tokens_list,
            masks=masks_list,
            scores=scores,
            ref_logprobs=ref_logprobs_list if ref_model is not None else None
        )
        
        if not success:
            logger.error("Failed to submit data to Atropos")
            return None
            
        # Wait a bit for processing
        time.sleep(1.0)
        
        # Retrieve batch with advantages
        batch = self.retrieve_batch()
        
        if batch is None:
            logger.warning("No batch received from Atropos")
            return None
            
        # Extract advantages from batch
        return self._extract_advantages_from_batch(batch, batch_size, prompt_len + response_len)
        
    def _compute_reference_logprobs(self, tokens: torch.Tensor, ref_model) -> List[float]:
        """Compute reference log probabilities using reference model"""
        ref_model.eval()
        device = next(ref_model.parameters()).device
        
        with torch.no_grad():
            input_ids = tokens.unsqueeze(0).to(device)
            outputs = ref_model(input_ids=input_ids)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Get log probs for actual tokens
            ref_logprobs = []
            for j in range(len(tokens) - 1):
                token_id = tokens[j + 1]
                log_prob = log_probs[0, j, token_id].item()
                ref_logprobs.append(log_prob)
            ref_logprobs.append(0.0)  # Placeholder for last token
            
        return ref_logprobs
        
    def _compute_default_scores(
        self,
        responses: torch.Tensor,
        log_probs: Optional[torch.Tensor],
        tokenizer
    ) -> List[float]:
        """Compute default scores based on response quality heuristics"""
        batch_size = responses.shape[0]
        scores = []
        
        for i in range(batch_size):
            response = responses[i]
            non_pad_mask = response != tokenizer.pad_token_id
            valid_response = response[non_pad_mask]
            
            if len(valid_response) == 0:
                scores.append(0.0)
                continue
                
            # Simple scoring based on length and diversity
            unique_tokens = len(torch.unique(valid_response))
            diversity = unique_tokens / len(valid_response)
            
            # Length preference (prefer moderate length)
            ideal_length = 15
            length_score = 1.0 - min(abs(len(valid_response) - ideal_length) / ideal_length, 1.0)
            
            # If log probs available, use them
            if log_probs is not None:
                valid_log_probs = log_probs[i][non_pad_mask]
                avg_log_prob = valid_log_probs.mean().item()
                confidence = min(1.0, max(0.0, 1.0 + avg_log_prob))
                score = 0.5 * confidence + 0.3 * diversity + 0.2 * length_score
            else:
                score = 0.6 * diversity + 0.4 * length_score
                
            scores.append(float(score))
            
        return scores
        
    def _extract_advantages_from_batch(
        self,
        batch: List[Dict[str, Any]],
        expected_batch_size: int,
        seq_len: int
    ) -> Dict[str, Any]:
        """Extract advantages and other data from Atropos batch"""
        # Initialize result tensors
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        advantages = torch.zeros(expected_batch_size, seq_len, device=device)
        
        # Process each item in batch
        for i, item in enumerate(batch):
            if i >= expected_batch_size:
                break
                
            # Check if advantages are provided
            if 'advantages' in item and item['advantages'] is not None:
                item_advantages = torch.tensor(item['advantages'], device=device)
                advantages[i, :len(item_advantages)] = item_advantages
            elif 'scores' in item:
                # Use scores as advantages if no token-level advantages
                score = item['scores']
                advantages[i, :] = score
                
        return {
            'advantages': advantages,
            'batch_data': batch,
            'processed_count': min(len(batch), expected_batch_size)
        }