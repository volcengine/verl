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
Atropos Integration Utilities

This module provides utility functions for integrating VeRL with Atropos environments,
including API validation, data conversion, and endpoint management.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import requests
from requests.exceptions import ConnectionError, RequestException, Timeout

import torch
import numpy as np

logger = logging.getLogger(__name__)


class AtroposAPIValidator:
    """Validates Atropos API connectivity and endpoints"""
    
    @staticmethod
    def validate_api_connectivity(api_url: str, timeout: int = 10) -> bool:
        """
        Test Atropos API connectivity
        
        Args:
            api_url: Atropos API base URL
            timeout: Request timeout in seconds
            
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            response = requests.get(f"{api_url}/status", timeout=timeout)
            if response.status_code == 200:
                logger.info(f"Atropos API connectivity confirmed at {api_url}")
                return True
            else:
                logger.error(f"Atropos API returned status {response.status_code}")
                return False
        except Timeout:
            logger.error(f"Connection timeout to Atropos API at {api_url}")
            return False
        except ConnectionError:
            logger.error(f"Cannot connect to Atropos API at {api_url}")
            return False
        except RequestException as e:
            logger.error(f"Network error connecting to Atropos API: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Atropos API: {e}")
            return False
    
    @staticmethod
    def validate_registration_data(registration_data: Dict[str, Any]) -> bool:
        """
        Validate registration data format for Atropos API
        
        Args:
            registration_data: Data to be sent to /register endpoint
            
        Returns:
            True if data is valid, False otherwise
        """
        required_fields = [
            "wandb_group", "wandb_project", "batch_size", 
            "max_token_len", "checkpoint_dir", "starting_step", "num_steps"
        ]
        
        for field in required_fields:
            if field not in registration_data:
                logger.error(f"Missing required registration field: {field}")
                return False
        
        # Validate data types
        if not isinstance(registration_data["batch_size"], int) or registration_data["batch_size"] <= 0:
            logger.error("batch_size must be a positive integer")
            return False
        
        if not isinstance(registration_data["max_token_len"], int) or registration_data["max_token_len"] <= 0:
            logger.error("max_token_len must be a positive integer")
            return False
        
        return True


class AtroposDataConverter:
    """Converts data between VeRL and Atropos formats"""
    
    @staticmethod
    def verl_to_atropos_batch(
        tokens: torch.Tensor,
        masks: torch.Tensor,
        scores: torch.Tensor,
        ref_logprobs: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Convert VeRL batch data to Atropos submission format
        
        Args:
            tokens: Token IDs [batch_size, seq_len]
            masks: Attention masks [batch_size, seq_len]
            scores: Response scores [batch_size]
            ref_logprobs: Reference log probabilities [batch_size, seq_len]
            
        Returns:
            Dictionary in Atropos submission format
        """
        submission_data = {
            "tokens": tokens.tolist() if torch.is_tensor(tokens) else tokens,
            "masks": masks.tolist() if torch.is_tensor(masks) else masks,
            "scores": scores.tolist() if torch.is_tensor(scores) else scores,
        }
        
        if ref_logprobs is not None:
            submission_data["ref_logprobs"] = (
                ref_logprobs.tolist() if torch.is_tensor(ref_logprobs) else ref_logprobs
            )
        
        return submission_data
    
    @staticmethod
    def atropos_to_verl_batch(atropos_batch: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        """
        Convert Atropos batch to VeRL tensors
        
        Args:
            atropos_batch: Batch data from Atropos API
            
        Returns:
            Tuple of (tokens, masks, advantages, scores)
        """
        batch_items = atropos_batch['batch']
        
        all_tokens = []
        all_masks = []
        all_advantages = []
        all_scores = []
        
        for item in batch_items:
            tokens = torch.tensor(item['tokens'], dtype=torch.long)
            masks = torch.tensor(item['masks'], dtype=torch.float)
            
            # Handle advantages - can be token-level or response-level
            if 'advantages' in item and item['advantages'] is not None:
                if isinstance(item['advantages'][0], list):
                    # Token-level advantages
                    advantages = torch.tensor(item['advantages'], dtype=torch.float)
                else:
                    # Response-level advantages - broadcast to token level
                    adv_val = item['advantages'][0]
                    advantages = torch.full_like(masks, adv_val)
            else:
                # Use scores as advantages if no explicit advantages
                score = item.get('scores', [0.0])[0]
                advantages = torch.full_like(masks, score)
            
            all_tokens.append(tokens)
            all_masks.append(masks)
            all_advantages.append(advantages)
            all_scores.append(item.get('scores', [0.0]))
        
        # Pad sequences to same length
        max_len = max(tokens.shape[-1] for tokens in all_tokens)
        
        padded_tokens = torch.stack([
            torch.nn.functional.pad(tokens, (0, max_len - tokens.shape[-1]), value=0)
            for tokens in all_tokens
        ])
        
        padded_masks = torch.stack([
            torch.nn.functional.pad(masks, (0, max_len - masks.shape[-1]), value=0.0)
            for masks in all_masks
        ])
        
        padded_advantages = torch.stack([
            torch.nn.functional.pad(advantages, (0, max_len - advantages.shape[-1]), value=0.0)
            for advantages in all_advantages
        ])
        
        scores_tensor = torch.tensor([scores[0] for scores in all_scores], dtype=torch.float)
        
        return padded_tokens, padded_masks, padded_advantages, scores_tensor


class AtroposEndpointManager:
    """Manages inference server endpoints for Atropos integration"""
    
    def __init__(self, base_port: int = 9000):
        self.base_port = base_port
        self.active_endpoints = []
    
    def generate_endpoints(self, num_servers: int, host: str = "localhost") -> List[str]:
        """
        Generate inference server endpoint URLs
        
        Args:
            num_servers: Number of inference servers to create endpoints for
            host: Host address for the servers
            
        Returns:
            List of endpoint URLs
        """
        endpoints = []
        for i in range(num_servers):
            port = self.base_port + i
            endpoint = f"http://{host}:{port}"
            endpoints.append(endpoint)
        
        self.active_endpoints = endpoints
        logger.info(f"Generated {len(endpoints)} inference endpoints")
        return endpoints
    
    def validate_endpoints(self, endpoints: List[str], timeout: int = 5) -> List[str]:
        """
        Validate that inference endpoints are accessible
        
        Args:
            endpoints: List of endpoint URLs to validate
            timeout: Request timeout in seconds
            
        Returns:
            List of accessible endpoints
        """
        valid_endpoints = []
        
        for endpoint in endpoints:
            try:
                # Try to reach the health endpoint
                response = requests.get(f"{endpoint}/health", timeout=timeout)
                if response.status_code == 200:
                    valid_endpoints.append(endpoint)
                    logger.debug(f"Endpoint {endpoint} is accessible")
                else:
                    logger.warning(f"Endpoint {endpoint} returned status {response.status_code}")
            except Exception as e:
                logger.warning(f"Endpoint {endpoint} is not accessible: {e}")
        
        logger.info(f"Validated {len(valid_endpoints)}/{len(endpoints)} endpoints")
        return valid_endpoints


class AtroposRetryHandler:
    """Handles retry logic for Atropos API calls"""
    
    @staticmethod
    def retry_with_backoff(
        func,
        max_attempts: int = 8,
        initial_delay: float = 0.3,
        max_delay: float = 2.0,
        backoff_factor: float = 1.5,
        max_wait_time: float = 12.0
    ):
        """
        Retry a function with exponential backoff
        
        Args:
            func: Function to retry
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay between retries
            max_delay: Maximum delay between retries
            backoff_factor: Factor to multiply delay by each attempt
            max_wait_time: Maximum total time to spend retrying
            
        Returns:
            Result of successful function call, or None if all attempts failed
        """
        start_time = time.time()
        current_delay = initial_delay
        
        for attempt in range(max_attempts):
            try:
                # Check if we've exceeded maximum wait time
                elapsed_time = time.time() - start_time
                if elapsed_time > max_wait_time:
                    logger.warning(f"Maximum wait time ({max_wait_time}s) exceeded")
                    break
                
                logger.debug(f"Attempt {attempt + 1}/{max_attempts} "
                           f"(elapsed: {elapsed_time:.1f}s)")
                
                # Try the function
                result = func()
                if result is not None:
                    return result
                
                # Exponential backoff
                time.sleep(current_delay)
                current_delay = min(current_delay * backoff_factor, max_delay)
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(current_delay)
                current_delay = min(current_delay * backoff_factor, max_delay)
        
        logger.warning("All retry attempts failed")
        return None


def create_atropos_registration_data(config) -> Dict[str, Any]:
    """
    Create registration data for Atropos API from VeRL config
    
    Args:
        config: VeRL configuration object
        
    Returns:
        Registration data dictionary
    """
    return {
        "wandb_group": config.trainer.get("wandb_group", "verl_atropos_integration"),
        "wandb_project": config.trainer.get("project_name", "verl_atropos_grpo"),
        "batch_size": config.data.train_batch_size,
        "max_token_len": config.data.max_prompt_length + config.data.max_response_length,
        "checkpoint_dir": config.trainer.get("output_dir", "/tmp/verl_checkpoints"),
        "save_checkpoint_interval": config.trainer.get("save_freq", 100),
        "starting_step": 0,
        "num_steps": config.trainer.get("total_epochs", 1000)
    }


def validate_atropos_config(config) -> bool:
    """
    Validate Atropos configuration in VeRL config
    
    Args:
        config: VeRL configuration object
        
    Returns:
        True if configuration is valid, False otherwise
    """
    if "atropos" not in config:
        logger.error("Missing 'atropos' section in configuration")
        return False
    
    atropos_config = config.atropos
    
    required_fields = ["api_url"]
    for field in required_fields:
        if field not in atropos_config:
            logger.error(f"Missing required Atropos config field: {field}")
            return False
    
    # Validate API URL format
    api_url = atropos_config.api_url
    if not api_url.startswith(("http://", "https://")):
        logger.error(f"Invalid API URL format: {api_url}")
        return False
    
    logger.info("Atropos configuration validation passed")
    return True 