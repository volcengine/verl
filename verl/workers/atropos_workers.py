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
Atropos Workers for VERL Integration

This module provides workers that integrate VERL's training pipeline with Atropos environments.
The main component is AtroposRolloutWorker which handles:
- Registration with Atropos API
- Inference server endpoint provision
- Data collection and submission
- Batch retrieval with token-level advantages
- GPU device management and tensor placement
"""

import asyncio
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from requests.exceptions import ConnectionError, RequestException, Timeout

from verl import DataProto
from verl.utils.logger import HFTBLogger
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.atropos_utils import (
    AtroposAPIValidator, AtroposDataConverter, AtroposEndpointManager,
    AtroposRetryHandler, create_atropos_registration_data, validate_atropos_config,
    get_device_for_tensor_ops
)

logger = logging.getLogger(__name__)


class AtroposAPIError(Exception):
    """Raised when Atropos API operations fail"""
    pass


class AtroposRolloutWorker(ActorRolloutRefWorker):
    """
    Atropos integration worker that replaces standard rollout generation
    with Atropos API-based environment coordination.
    
    This worker:
    1. Registers with Atropos API as a trainer
    2. Provides inference server endpoints to Atropos
    3. Coordinates data collection through Atropos environments
    4. Retrieves processed batches with token-level advantages
    5. Manages policy weight updates to inference servers
    6. Handles GPU device placement and memory management
    """
    
    def __init__(self, config: DictConfig, device: str = "cuda:0"):
        super().__init__(config=config, device=device)
        
        # Setup GPU device for tensor operations
        self.tensor_device = get_device_for_tensor_ops(device)
        
        # Initialize CUDA if available and using GPU
        if self.tensor_device.type == "cuda":
            torch.cuda.init()
            torch.cuda.set_device(self.tensor_device)
            logger.info(f"Initialized CUDA device: {self.tensor_device}")
        else:
            logger.warning(f"Using CPU device: {self.tensor_device}")
        
        # Atropos configuration
        self.atropos_config = config.get("atropos", {})
        self.api_url = self.atropos_config.get("api_url", "http://localhost:8000")
        self.api_timeout = self.atropos_config.get("timeout", 30)
        self.batch_retry_attempts = self.atropos_config.get("batch_retry_attempts", 8)
        self.batch_retry_delay = self.atropos_config.get("batch_retry_delay", 0.3)
        self.batch_max_wait_time = self.atropos_config.get("batch_max_wait_time", 12.0)
        
        # Registration state
        self.registered = False
        self.trainer_uuid = None
        self.step_count = 0
        
        # Inference server endpoints (provided by VeRL)
        self.inference_endpoints = []
        
        logger.info(f"AtroposRolloutWorker initialized with API URL: {self.api_url}")
        logger.info(f"Using device: {self.tensor_device} for tensor operations")
    
    def set_inference_endpoints(self, endpoints: List[str]):
        """Set inference server endpoints that will be provided to Atropos"""
        self.inference_endpoints = endpoints
        logger.info(f"Configured {len(endpoints)} inference endpoints for Atropos")
    
    def _test_api_connectivity(self) -> None:
        """Test Atropos API connectivity and raise error if unreachable"""
        if not AtroposAPIValidator.validate_api_connectivity(self.api_url, self.api_timeout):
            raise AtroposAPIError(f"Cannot connect to Atropos API at {self.api_url} - ensure server is running")
    
    def _register_with_atropos(self) -> bool:
        """Register this trainer with the Atropos API"""
        if self.registered:
            return True
        
        try:
            logger.info("Registering trainer with Atropos API...")
            
            # Create registration data using utility function
            registration_data = create_atropos_registration_data(self.config)
            registration_data["starting_step"] = self.step_count
            
            # Add GPU/device information to registration
            registration_data["device_info"] = {
                "device_type": self.tensor_device.type,
                "device_index": self.tensor_device.index if self.tensor_device.type == "cuda" else None,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            # Validate registration data
            if not AtroposAPIValidator.validate_registration_data(registration_data):
                raise AtroposAPIError("Invalid registration data format")
            
            response = requests.post(
                f"{self.api_url}/register", 
                json=registration_data, 
                timeout=self.api_timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                self.trainer_uuid = data.get('uuid')
                logger.info(f"Trainer successfully registered with UUID: {self.trainer_uuid}")
                self.registered = True
                return True
            else:
                raise AtroposAPIError(f"Registration failed: HTTP {response.status_code} - {response.text}")
                
        except RequestException as e:
            raise AtroposAPIError(f"Registration request failed: {e}")
        except Exception as e:
            raise AtroposAPIError(f"Registration failed: {e}")
    
    def _provide_endpoints_to_atropos(self):
        """Provide inference server endpoints to Atropos environments"""
        if not self.inference_endpoints:
            logger.warning("No inference endpoints configured for Atropos")
            return
        
        try:
            # Provide endpoints to Atropos via the /endpoints API
            endpoint_data = {
                "trainer_uuid": self.trainer_uuid,
                "endpoints": self.inference_endpoints,
                "model_info": {
                    "max_tokens": self.config.data.max_response_length,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "device": str(self.tensor_device)
                }
            }
            
            response = requests.post(
                f"{self.api_url}/endpoints",
                json=endpoint_data,
                timeout=self.api_timeout
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully provided {len(self.inference_endpoints)} endpoints to Atropos")
            else:
                logger.warning(f"Failed to provide endpoints to Atropos: HTTP {response.status_code}")
                
        except RequestException as e:
            logger.error(f"Failed to provide endpoints to Atropos: {e}")
            raise AtroposAPIError(f"Endpoint provision failed: {e}")
    
    def _retrieve_batch_from_atropos(self) -> Optional[Dict[str, Any]]:
        """Retrieve processed batch from Atropos API with retry logic"""
        logger.info("Retrieving batch from Atropos API...")
        
        def _try_get_batch():
            """Single attempt to get batch from Atropos"""
            # Check queue status first
            try:
                status_response = requests.get(f"{self.api_url}/status", timeout=5)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    queue_size = status_data.get('queue_size', 0)
                    current_step = status_data.get('current_step', 0)
                    logger.debug(f"Queue status - Size: {queue_size}, Step: {current_step}")
            except Exception as e:
                logger.debug(f"Status check failed: {e}")
            
            # Try to get the batch
            response = requests.get(f"{self.api_url}/batch", timeout=10)
            
            if response.status_code == 200:
                batch_data = response.json()
                if 'batch' in batch_data and batch_data['batch']:
                    logger.info(f"Successfully retrieved batch with {len(batch_data['batch'])} items")
                    return batch_data
                else:
                    logger.debug("No batch available yet")
                    return None
            elif response.status_code == 204:
                logger.debug("No batch ready (HTTP 204)")
                return None
            else:
                logger.warning(f"Batch request failed: HTTP {response.status_code}")
                return None
        
        # Use retry handler
        return AtroposRetryHandler.retry_with_backoff(
            _try_get_batch,
            max_attempts=self.batch_retry_attempts,
            initial_delay=self.batch_retry_delay,
            max_wait_time=self.batch_max_wait_time
        )
    
    def _submit_scored_data_to_atropos(self, tokens, masks, scores, ref_logprobs=None):
        """Submit scored data to Atropos for processing with proper device handling"""
        try:
            # Ensure tensors are properly placed before conversion
            if torch.is_tensor(tokens) and tokens.device != self.tensor_device:
                tokens = tokens.to(self.tensor_device)
            if torch.is_tensor(masks) and masks.device != self.tensor_device:
                masks = masks.to(self.tensor_device)
            if torch.is_tensor(scores) and scores.device != self.tensor_device:
                scores = scores.to(self.tensor_device)
            if ref_logprobs is not None and torch.is_tensor(ref_logprobs) and ref_logprobs.device != self.tensor_device:
                ref_logprobs = ref_logprobs.to(self.tensor_device)
            
            # Convert data using utility function (handles GPU->CPU conversion internally)
            data_payload = AtroposDataConverter.verl_to_atropos_batch(
                tokens, masks, scores, ref_logprobs
            )
            
            # Prepare submission with metadata
            submission_data = {
                "trainer_uuid": self.trainer_uuid,
                "step": self.step_count,
                "data": data_payload
            }
            
            response = requests.post(
                f"{self.api_url}/scored_data",
                json=submission_data,
                timeout=self.api_timeout
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully submitted scored data for step {self.step_count}")
                return True
            else:
                logger.warning(f"Failed to submit scored data: HTTP {response.status_code} - {response.text}")
                return False
                
        except RequestException as e:
            logger.error(f"Failed to submit scored data to Atropos: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error submitting scored data: {e}")
            return False
    
    def _convert_atropos_to_verl_data(self, atropos_batch: Dict[str, Any]) -> DataProto:
        """Convert Atropos batch data to VeRL DataProto format with proper GPU device placement"""
        # Use utility function for conversion with target device
        tokens, masks, advantages, scores = AtroposDataConverter.atropos_to_verl_batch(
            atropos_batch, device=self.tensor_device
        )
        
        # Ensure all tensors are on the correct device
        tokens = tokens.to(self.tensor_device)
        masks = masks.to(self.tensor_device)
        advantages = advantages.to(self.tensor_device)
        scores = scores.to(self.tensor_device)
        
        # Create UIDs for GRPO grouping
        batch_size = tokens.shape[0]
        uids = np.arange(batch_size)
        
        # Create DataProto with properly placed tensors
        batch_data = {
            'input_ids': tokens,
            'attention_mask': masks,
            'response_mask': masks,  # Assume all tokens are response for now
            'token_level_rewards': advantages,
            'advantages': advantages,
            'returns': advantages,  # For GRPO, returns = advantages
        }
        
        non_tensor_batch = {
            'uid': uids
        }
        
        return DataProto(batch=batch_data, non_tensor_batch=non_tensor_batch)
    
    def generate_sequences(
        self,
        prompts: DataProto,
        sampling_params: Optional[Dict] = None,
        **kwargs
    ) -> DataProto:
        """
        Override the standard sequence generation to use Atropos API coordination.
        
        This method implements the full Atropos integration workflow:
        1. Ensures registration with Atropos API
        2. Provides inference endpoints to Atropos environments
        3. Generates responses using VeRL's inference engines
        4. Submits scored data to Atropos for environment processing
        5. Retrieves processed batches with token-level advantages from Atropos
        
        All tensor operations are performed on the configured GPU device.
        """
        
        # Ensure prompts are on the correct device
        if prompts.batch is not None:
            for key, tensor in prompts.batch.items():
                if torch.is_tensor(tensor) and tensor.device != self.tensor_device:
                    prompts.batch[key] = tensor.to(self.tensor_device)
        
        # Clear GPU cache before heavy operations
        if self.tensor_device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Ensure API connectivity
        try:
            self._test_api_connectivity()
        except AtroposAPIError as e:
            logger.error(f"Atropos API connectivity failed: {e}")
            raise
        
        # Register with Atropos if not already done
        if not self.registered:
            try:
                self._register_with_atropos()
            except AtroposAPIError as e:
                logger.error(f"Atropos registration failed: {e}")
                raise
        
        # Provide endpoints to Atropos environments
        self._provide_endpoints_to_atropos()
        
        # Step 1: Generate responses using VeRL's standard inference
        # This uses the actual model to generate responses
        logger.info("Generating responses using VeRL inference engines...")
        generated_data = super().generate_sequences(prompts, sampling_params, **kwargs)
        
        # Step 2: Extract data for Atropos submission and ensure proper device placement
        tokens = generated_data.batch['input_ids'].to(self.tensor_device)
        attention_mask = generated_data.batch.get('attention_mask', 
                                                 torch.ones_like(tokens).to(self.tensor_device))
        
        # Compute initial scores (these will be refined by Atropos environments)
        # For now, use simple length-based scoring as placeholder
        response_lengths = attention_mask.sum(dim=-1)
        initial_scores = response_lengths.float() / tokens.shape[-1]  # Normalize by max length
        initial_scores = initial_scores.to(self.tensor_device)
        
        # Step 3: Submit scored data to Atropos for environment processing
        logger.info("Submitting scored data to Atropos environments...")
        submission_success = self._submit_scored_data_to_atropos(
            tokens=tokens,
            masks=attention_mask,
            scores=initial_scores,
            ref_logprobs=generated_data.batch.get('ref_log_prob')
        )
        
        if not submission_success:
            logger.warning("Failed to submit data to Atropos, using generated data as-is")
            self.step_count += 1
            # Ensure return data is on correct device
            return self._ensure_data_on_device(generated_data)
        
        # Step 4: Retrieve processed batch with advantages from Atropos
        logger.info("Retrieving processed batch with advantages from Atropos...")
        self.step_count += 1
        
        atropos_batch = self._retrieve_batch_from_atropos()
        
        if atropos_batch is None:
            logger.warning("Failed to retrieve batch from Atropos, using generated data")
            return self._ensure_data_on_device(generated_data)
        
        # Step 5: Convert Atropos batch to VeRL format with token-level advantages
        verl_data = self._convert_atropos_to_verl_data(atropos_batch)
        
        # Clear GPU cache after processing
        if self.tensor_device.type == "cuda":
            torch.cuda.empty_cache()
        
        logger.info(f"Successfully processed Atropos batch for step {self.step_count}")
        return verl_data
    
    def _ensure_data_on_device(self, data: DataProto) -> DataProto:
        """Ensure all tensors in DataProto are on the correct device"""
        if data.batch is not None:
            for key, tensor in data.batch.items():
                if torch.is_tensor(tensor) and tensor.device != self.tensor_device:
                    data.batch[key] = tensor.to(self.tensor_device)
        return data
    
    @contextmanager
    def policy_weight_sync(self):
        """Context manager for policy weight synchronization with GPU memory management"""
        # This will be called before rollout to ensure inference servers have latest weights
        logger.debug("Syncing policy weights to inference servers")
        
        # Clear GPU cache before weight sync
        if self.tensor_device.type == "cuda":
            torch.cuda.empty_cache()
        
        try:
            yield
        finally:
            # Clear GPU cache after weight sync
            if self.tensor_device.type == "cuda":
                torch.cuda.empty_cache()
            logger.debug("Policy weight sync complete")


class AtroposShardingManager:
    """
    Sharding manager for Atropos integration that handles weight synchronization
    between training and inference engines with GPU support.
    """
    
    def __init__(self, training_model, inference_engine, device: Union[str, torch.device] = None):
        self.training_model = training_model
        self.inference_engine = inference_engine
        self.device = get_device_for_tensor_ops(device)
        
        # Initialize CUDA if using GPU
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
    
    def __enter__(self):
        """Sync latest training weights to inference engine"""
        logger.debug(f"Syncing training weights to inference engine on device {self.device}")
        
        # Clear GPU cache before weight sync
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        # TODO: Implement weight synchronization with device placement
        # This would use VeRL's existing weight update mechanisms
        # Ensure all weight tensors are moved to the correct device
        
        return self
    
    def __exit__(self, *args):
        """Clean up after inference with GPU memory management"""
        # Clear GPU cache after inference
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.debug("Inference complete, released GPU resources") 