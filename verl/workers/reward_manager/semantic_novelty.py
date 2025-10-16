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
Semantic Novelty Reward Manager

This module implements a reward adjustment mechanism based on semantic novelty for RLHF training,
encouraging the model to generate both correct and innovative answers while avoiding repetitive or mediocre responses.

Core ideas:
1. For correct answers: Calculate maximum and average similarity with correct solution group, reward range 0.5-1
2. For incorrect answers: Only consider global similarity (average and maximum), reward range -1 to -0.5
3. \\boxed invalid answers: Directly give -1 reward

Uses remote vLLM API to compute embeddings, supporting efficient batch processing.
"""

from collections import defaultdict
import numpy as np
import torch
import time
import random
import os
import re
import requests
import json

from verl import DataProto
from verl.workers.reward_manager import register
from verl.utils.reward_score.ttrl.auto_verify import auto_verify
from verl.utils.reward_score.ttrl.ttt_metrics import (
    post_test_time_train_metrics, test_time_train_metrics)

# Try to import wandb, use empty implementation if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    class MockWandb:
        @staticmethod
        def log(*args, **kwargs):
            pass
    wandb = MockWandb()

# Optional import of HF tokenizer for token-level truncation of embedding text to avoid exceeding embedding model context
try:
    from transformers import AutoTokenizer  # type: ignore
    HF_TRANSFORMERS_AVAILABLE = True
except Exception:
    HF_TRANSFORMERS_AVAILABLE = False

@register("semantic")
class SemanticTTRLRewardManager:
    """
    TTRL Reward Manager with Semantic Novelty Adjustment
    
    This class adds semantic novelty evaluation mechanism on top of the original TTRL reward:
    1. Novelty calculation: Average similarity calculated within groups, maximum similarity calculated globally
    2. Novelty normalization: Normalized within correct answer groups, normalized within incorrect answer groups
    3. For correct answers: All correct answers receive positive rewards
       - All correct answers: reward range [0.5,1] (higher novelty = higher reward)
    4. For incorrect answers: All incorrect answers receive negative rewards
       - All incorrect answers: reward range [-1,-0.5] (higher novelty = less punishment)
    5. For \\boxed invalid answers: Directly give -1 reward, not participating in semantic similarity calculation
    
    Uses remote vLLM API to compute embeddings, supporting efficient batch processing.
    
    Simplified reward mechanism:
    - Correct answers: Uniformly rescaled to [0.5,1] range, higher novelty = higher reward
    - Incorrect answers: Uniformly rescaled to [-1,-0.5] range, higher novelty = less punishment
    - Invalid answers: Directly give -1 reward
    """

    def __init__(self, tokenizer, num_examine, reward_fn_key="data_source", compute_score=None, 
                 n_votes_per_prompt=1, n_samples_per_prompt=1, mode="eval", eval_n_samples=1,
                 # === Semantic novelty related parameters ===
                 use_semantic_novelty: bool = True,
                 embedding_dim: int = 2560,
                 # === Remote vLLM API parameters ===
                 vllm_api_url: str = "http://30.159.163.76:2341",
                 vllm_model_name: str = "Qwen/Qwen3-Embedding-4B",
                 # === API retry parameters ===
                 max_retries: int = 5,
                 retry_delay: float = 1.0
                 ) -> None:
        # === Basic parameter setup ===
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.n_votes_per_prompt = n_votes_per_prompt
        self.n_samples_per_prompt = n_samples_per_prompt
        self.mode = mode
        self.eval_n_samples = eval_n_samples
        
        # Parameter validity check
        assert n_votes_per_prompt >= n_samples_per_prompt, \
            f"TTRL requirement: n_votes_per_prompt({n_votes_per_prompt}) >= n_samples_per_prompt({n_samples_per_prompt})"

        # === Semantic novelty parameter setup and validation ===
        self.use_semantic_novelty = use_semantic_novelty
        self.embedding_dim = embedding_dim
        

        
        # === Remote vLLM API parameters ===
        self.vllm_api_url = vllm_api_url.rstrip('/')
        self.vllm_model_name = vllm_model_name
        
        # Semantic novelty parameter validation
        if self.use_semantic_novelty:
            assert self.embedding_dim > 0, f"embedding_dim must be greater than 0, current value: {self.embedding_dim}"

        # === API retry parameter setup ===
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # === Debug and performance control ===
        self.debug_mode = num_examine > 0
        
        # Initialize related components
        if self.use_semantic_novelty:
            self._init_novelty_components()

        # === Initialize log output ===
        self._print_initialization_info()

    def _init_novelty_components(self):
        """
        Initialize semantic novelty related components
        """
        # Statistics
        self._retry_stats = {"total_calls": 0, "total_retries": 0, "failed_calls": 0}
        self._internal_step = 0
        
        # Novelty statistics (for wandb logging)
        self._novelty_stats = {}

        # Embedding-side tokenizer and safe token limit (leave safety margin to avoid overflow)
        # Note: vLLM server sets max_model_len=16384, here we pre-truncate on client side with slightly smaller threshold
        # to absorb tokenization differences and avoid server errors.
        self.embedding_max_token_len = 16000
        self._embedding_tokenizer = None
        if HF_TRANSFORMERS_AVAILABLE:
            try:
                # Use the same model name as embedding service for consistent tokenization behavior
                self._embedding_tokenizer = AutoTokenizer.from_pretrained(
                    self.vllm_model_name, trust_remote_code=True, use_fast=True
                )
                print("  - Embedding tokenizer: loaded, used for safe truncation")
            except Exception as e:
                self._embedding_tokenizer = None
                print(f"  - Embedding tokenizer: loading failed, fallback to character truncation ({e})")
        
        # Initialize wandb logging
        if WANDB_AVAILABLE:
            print(f"  - WandB logging: enabled")
        else:
            print(f"  - WandB logging: wandb not installed, skipping logging")

    def _print_initialization_info(self):
        """
        Print initialization information
        """
        print(f"SemanticTTRLRewardManager initialization completed:")
        print(f"  - Votes/samples: {self.n_votes_per_prompt}/{self.n_samples_per_prompt}")
        print(f"  - Evaluation samples: {self.eval_n_samples}")
        print(f"  - Running mode: {self.mode}")
        
        if self.use_semantic_novelty:
            print(f"  - Semantic novelty: enabled")
            print(f"    * Remote vLLM API: {self.vllm_api_url}")
            print(f"    * Model used: {self.vllm_model_name}")
            print(f"    * Embedding dimension: {self.embedding_dim}")
            print(f"    * Reward range:")
            print(f"      - Correct answers: All answers rescaled to [0.5,1]")
            print(f"      - Incorrect answers: All answers rescaled to [-1,-0.5]")
            print(f"      - Ensure correct answers always get positive rewards, incorrect answers always get negative rewards")
        else:
            print(f"  - Semantic novelty: disabled (using original TTRL)")

    def _truncate_for_embedding(self, text: str) -> str:
        """
        Use embedding model's tokenizer to safely truncate text to no more than embedding_max_token_len tokens.
        If tokenizer cannot be used, fall back to conservative character truncation strategy (approximately 4 characters/token).
        """
        try:
            if self._embedding_tokenizer is not None:
                token_ids = self._embedding_tokenizer.encode(text, add_special_tokens=False)
                if len(token_ids) <= self.embedding_max_token_len:
                    return text
                # Keep the end part, discard the front (consistent with user requirements)
                truncated_ids = token_ids[-self.embedding_max_token_len:]
                truncated_text = self._embedding_tokenizer.decode(truncated_ids, skip_special_tokens=True)
                return truncated_text
            else:
                # Fallback: character truncation (roughly assuming ~4 characters/token)
                max_chars = self.embedding_max_token_len * 4
                # Keep end characters, discard front
                return text[-max_chars:]
        except Exception:
            # Any exception falls back to character truncation for robustness
            max_chars = self.embedding_max_token_len * 4
            return text[-max_chars:]

    def _retry_with_exponential_backoff(self, func, *args, **kwargs):
        """
        Retry function calls using exponential backoff strategy
        """
        last_exception = None
        self._retry_stats["total_calls"] += 1
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter
                    
                    print(f"    Retry attempt {attempt}, waiting {total_delay:.1f} seconds...")
                    time.sleep(total_delay)
                    self._retry_stats["total_retries"] += 1
                
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                error_msg = str(e)
                
                if self._is_retryable_error(e):
                    if attempt < self.max_retries:
                        print(f"    Attempt {attempt + 1} failed (retryable): {error_msg}")
                        continue
                    else:
                        print(f"    Reached maximum retry count ({self.max_retries}), giving up")
                        break
                else:
                    print(f"    Encountered non-retryable error: {error_msg}")
                    break
        
        self._retry_stats["failed_calls"] += 1
        raise last_exception
    
    def _is_retryable_error(self, exception) -> bool:
        """
        Determine if exception is retryable (applicable to HTTP request errors)
        """
        error_str = str(exception).lower()
        
        # Network-related errors
        network_errors = [
            'connection', 'timeout', 'network', 'dns', 'socket',
            'connection reset', 'connection refused', 'connection timeout',
            'read timeout', 'connect timeout'
        ]
        
        # Retryable HTTP status codes
        retryable_http_errors = ['429', '500', '502', '503', '504']
        
        # Non-retryable HTTP status codes
        non_retryable_http_errors = ['400', '401', '403', '404', '413', '422']
        
        # Check if it's a requests exception
        if hasattr(exception, 'response') and exception.response is not None:
            status_code = str(exception.response.status_code)
            if status_code in non_retryable_http_errors:
                return False
            if status_code in retryable_http_errors:
                return True
        
        # Check if error message contains non-retryable error codes
        for error_code in non_retryable_http_errors:
            if error_code in error_str:
                return False
        
        # Check if error message contains network errors or retryable error codes
        for error_pattern in network_errors + retryable_http_errors:
            if error_pattern in error_str:
                return True
        
        # Default to retryable
        return True

    def _get_semantic_embeddings_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Batch get semantic embedding vectors for multiple texts (using remote vLLM API)
        """
        if not texts:
            return []

        # First perform safe truncation to avoid exceeding embedding server max_model_len
        original_count = len(texts)
        truncated_texts = []
        truncated_num = 0
        for t in texts:
            t2 = self._truncate_for_embedding(t)
            if t2 is not t and len(t2) < len(t):
                truncated_num += 1
            truncated_texts.append(t2)
        if truncated_num > 0:
            print(f"    Safely truncated {truncated_num}/{original_count} texts to adapt to embedding model length limit")

        MAX_BATCH_SIZE = 128

        if len(truncated_texts) > MAX_BATCH_SIZE:
            print(f"    Text count ({len(truncated_texts)}) exceeds API limit ({MAX_BATCH_SIZE}), processing in batches...")
            all_embeddings = []
            for i in range(0, len(truncated_texts), MAX_BATCH_SIZE):
                batch_texts = truncated_texts[i:i + MAX_BATCH_SIZE]
                batch_embeddings = self._get_embeddings_batch_direct(batch_texts)
                all_embeddings.extend(batch_embeddings)
            return all_embeddings
            
        return self._get_embeddings_batch_direct(truncated_texts)
    
    def _get_embeddings_batch_direct(self, texts: list[str]) -> list[np.ndarray]:
        """
        Directly call remote vLLM API to get embedding vectors
        """
        try:
            def _api_call():
                # Build API request - use /embed endpoint
                api_url = f"{self.vllm_api_url}/embed"
                headers = {
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "texts": texts
                }
                
                response = requests.post(api_url, headers=headers, json=payload, timeout=300)
                response.raise_for_status()
                
                return response.json()
            
            response_data = self._retry_with_exponential_backoff(_api_call)
            if "embeddings" not in response_data:
                print(f"API returned abnormal content: {response_data}")
                raise KeyError("API returned content missing 'embeddings' field")
            
            if len(response_data["embeddings"]) != len(texts):
                print(f"Warning: returned embedding count ({len(response_data['embeddings'])}) does not match input text count ({len(texts)})!")
            
            embeddings = []
            for i, embedding_array in enumerate(response_data["embeddings"]):
                embedding_array = np.array(embedding_array, dtype=np.float32)
                if len(embedding_array) != self.embedding_dim:
                    print(f"Warning: text {i+1} embedding dimension mismatch! Expected {self.embedding_dim}, got {len(embedding_array)}")
                embeddings.append(embedding_array)
            
            return embeddings
            
        except Exception as e:
            print(f"Batch remote vLLM API call ultimately failed: {str(e)}")
            print(f"Falling back to single text processing...")
            return [self._get_semantic_embedding_single(text) for text in texts]
    
    def _get_semantic_embedding_single(self, text: str) -> np.ndarray:
        """
        Get semantic embedding vector for a single text (fallback solution)
        """
        try:
            def _api_call():
                api_url = f"{self.vllm_api_url}/embed"
                headers = {
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "texts": [text]
                }
                
                response = requests.post(api_url, headers=headers, json=payload, timeout=300)
                response.raise_for_status()
                
                return response.json()
            
            response_data = self._retry_with_exponential_backoff(_api_call)
            if "embeddings" not in response_data:
                print(f"API returned abnormal content: {response_data}")
                raise KeyError("API returned content missing 'embeddings' field")
            
            embedding = np.array(response_data["embeddings"][0], dtype=np.float32)
            if len(embedding) != self.embedding_dim:
                print(f"Warning: embedding dimension mismatch! Expected {self.embedding_dim}, got {len(embedding)}")
            return embedding
            
        except Exception as e:
            print(f"Single remote vLLM API call ultimately failed: {str(e)}")
            print(f"Falling back to random embedding vector...")
            np.random.seed(hash(text) % (2**32 - 1))
            return np.random.rand(self.embedding_dim).astype(np.float32)

    def get_retry_stats(self) -> dict:
        """
        Get API call statistics
        """
        if not self.use_semantic_novelty:
            return {"novelty_enabled": False}
        
        stats = {
            "novelty_enabled": True,
            "api_stats": self._retry_stats.copy()
        }
        
        if stats["api_stats"]["total_calls"] > 0:
            stats["api_stats"]["success_rate"] = 1 - (stats["api_stats"]["failed_calls"] / stats["api_stats"]["total_calls"])
            stats["api_stats"]["avg_retries_per_call"] = stats["api_stats"]["total_retries"] / stats["api_stats"]["total_calls"]
        else:
            stats["api_stats"]["success_rate"] = 0.0
            stats["api_stats"]["avg_retries_per_call"] = 0.0
            
        return stats
    
    def reset_retry_stats(self):
        """Reset API retry statistics"""
        if self.use_semantic_novelty:
            self._retry_stats = {"total_calls": 0, "total_retries": 0, "failed_calls": 0}
            print("API retry statistics reset")
    
    def reset_internal_step(self):
        """Reset internal step counter"""
        if self.use_semantic_novelty:
            self._internal_step = 0
            print("Internal step counter reset")
    
    def get_internal_step(self) -> int:
        """Get current internal step counter value"""
        return self._internal_step if self.use_semantic_novelty else 0
    
    def log_api_stats_to_wandb(self, step: int = None, prefix: str = "semantic_novelty"):
        """
        Log API call statistics to wandb
        """
        if not self.use_semantic_novelty or not WANDB_AVAILABLE:
            return
        
        stats = self.get_retry_stats()
        if not stats["novelty_enabled"]:
            return
        
        # Prepare wandb log data
        log_data = {}
        api_stats = stats["api_stats"]
        
        log_data[f"{prefix}/api_total_calls"] = api_stats["total_calls"]
        log_data[f"{prefix}/api_success_rate"] = api_stats["success_rate"]
        
        # Log to wandb
        if step is not None:
            wandb.log(log_data, step=step)
        else:
            wandb.log(log_data)
        
        print(f"Logged API statistics to wandb: success_rate={api_stats['success_rate']:.3f}, total_calls={api_stats['total_calls']}")

    def log_novelty_stats_to_wandb(self, step: int = None, prefix: str = "semantic_novelty"):
        """
        Log novelty statistics to wandb
        """
        if not self.use_semantic_novelty or not WANDB_AVAILABLE or not self._novelty_stats:
            return
        
        # Prepare wandb log data
        log_data = {}
        stats = self._novelty_stats
        
        # Answer group statistics
        if "answer_groups" in stats:
            groups = stats["answer_groups"]
            log_data[f"{prefix}/correct_answers_count"] = groups.get("n_correct", 0)
            log_data[f"{prefix}/incorrect_answers_count"] = groups.get("n_incorrect", 0)
            log_data[f"{prefix}/invalid_boxed_count"] = groups.get("n_invalid_boxed", 0)
            log_data[f"{prefix}/positive_reward_count"] = groups.get("n_positive_rewards", 0)
            log_data[f"{prefix}/negative_reward_count"] = groups.get("n_negative_rewards", 0)
            log_data[f"{prefix}/zero_reward_count"] = groups.get("n_zero_rewards", 0)
        
        # Novelty score statistics
        if "novelty_scores" in stats:
            novelty = stats["novelty_scores"]
            log_data[f"{prefix}/novelty_mean"] = novelty.get("mean", 0.0)
            log_data[f"{prefix}/novelty_std"] = novelty.get("std", 0.0)
            log_data[f"{prefix}/novelty_min"] = novelty.get("min", 0.0)
            log_data[f"{prefix}/novelty_max"] = novelty.get("max", 0.0)
            
            # Group novelty statistics
            if "correct" in novelty:
                correct_stats = novelty["correct"]
                log_data[f"{prefix}/novelty_correct_mean"] = correct_stats.get("mean", 0.0)
                log_data[f"{prefix}/novelty_correct_std"] = correct_stats.get("std", 0.0)
            if "incorrect" in novelty:
                incorrect_stats = novelty["incorrect"]
                log_data[f"{prefix}/novelty_incorrect_mean"] = incorrect_stats.get("mean", 0.0)
                log_data[f"{prefix}/novelty_incorrect_std"] = incorrect_stats.get("std", 0.0)
        
        # Reward distribution statistics
        if "rewards" in stats:
            rewards = stats["rewards"]
            log_data[f"{prefix}/reward_mean"] = rewards.get("mean", 0.0)
            log_data[f"{prefix}/reward_std"] = rewards.get("std", 0.0)
            log_data[f"{prefix}/reward_min"] = rewards.get("min", 0.0)
            log_data[f"{prefix}/reward_max"] = rewards.get("max", 0.0)
            
            # Group reward statistics
            if "positive_rewards" in rewards:
                pos_rewards = rewards["positive_rewards"]
                log_data[f"{prefix}/positive_reward_mean"] = pos_rewards.get("mean", 0.0)
                log_data[f"{prefix}/positive_reward_std"] = pos_rewards.get("std", 0.0)
            if "negative_rewards" in rewards:
                neg_rewards = rewards["negative_rewards"]
                log_data[f"{prefix}/negative_reward_mean"] = neg_rewards.get("mean", 0.0)
                log_data[f"{prefix}/negative_reward_std"] = neg_rewards.get("std", 0.0)
            if "incorrect_novel_rewards" in rewards:
                novel_rewards = rewards["incorrect_novel_rewards"]
                log_data[f"{prefix}/incorrect_novel_reward_mean"] = novel_rewards.get("mean", 0.0)
            if "incorrect_not_novel_rewards" in rewards:
                not_novel_rewards = rewards["incorrect_not_novel_rewards"]
                log_data[f"{prefix}/incorrect_not_novel_reward_mean"] = not_novel_rewards.get("mean", 0.0)
        
        # Similarity statistics
        if "similarity" in stats:
            sim = stats["similarity"]
            log_data[f"{prefix}/similarity_mean"] = sim.get("mean", 0.0)
            log_data[f"{prefix}/similarity_std"] = sim.get("std", 0.0)
            log_data[f"{prefix}/similarity_min"] = sim.get("min", 0.0)
            log_data[f"{prefix}/similarity_max"] = sim.get("max", 0.0)
        

        
        # Log to wandb
        if step is not None:
            wandb.log(log_data, step=step)
        else:
            wandb.log(log_data)
        
        print(f"Logged novelty statistics to wandb: {len(log_data)} metrics")


    def _check_boxed_content_has_numbers(self, text: str) -> bool:
        """
        Check if \\boxed{} content in text contains numbers
        """
        boxed_pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(boxed_pattern, text)
        
        if not matches:
            return False
        
        for match in matches:
            if re.search(r'\d', match):
                return True
        
        return False

    def _decode_data_item(self, data_item):
        """
        Decode prompt and response of a single data item
        """
        prompt_idx = data_item.batch["prompts"]
        prompt_length = prompt_idx.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_idx = prompt_idx[-valid_prompt_length:]
        
        response_idx = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_idx = response_idx[:valid_response_length]
        
        prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
        response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
        
        return prompt_str, response_str, valid_response_length

    def _compute_novelty_rewards(self, pred_outputs: list[str], base_rewards: list[float]) -> list[float]:
        """
        Core method for semantic novelty reward calculation
        
        New reward calculation logic:
        1. Novelty calculation: Average similarity calculated within groups (correct vs correct, incorrect vs incorrect), maximum similarity calculated globally
        2. Novelty normalization: Normalized within correct answer groups, normalized within incorrect answer groups to [0,1] range
        3. For correct answers: Control positive reward ratio based on positive_reward_ratio
           - Top ratio answers: Secondary rescale to [0.5,1] range (fully utilize positive reward space)
           - Remaining answers: Secondary rescale to [-1,-0.5] range (fully utilize negative reward space)
        4. For incorrect answers: All incorrect answers handled uniformly
           - All incorrect answers: rescaled to [-1,-0.5] range (higher novelty = less punishment)
        5. \\boxed invalid answers: Directly give -1 reward, not participating in semantic similarity calculation
        6. Simplified reward strategy: Correct answers [0.5,1], incorrect answers [-1,-0.5], adjusted based on novelty
        """
        n_answers = len(pred_outputs)
        original_pred_outputs = pred_outputs.copy()
        
        # Edge case: single answer cannot calculate novelty
        if n_answers <= 1:
            final_rewards = base_rewards
            # Set basic statistics
            self._novelty_stats = {
                "answer_groups": {
                    "n_correct": sum(1 for r in base_rewards if r > 0),
                    "n_incorrect": sum(1 for r in base_rewards if r <= 0),
                    "n_invalid_boxed": 0,
                    "n_positive_rewards": sum(1 for r in base_rewards if r > 0),
                    "n_negative_rewards": sum(1 for r in base_rewards if r <= 0),
                    "n_zero_rewards": 0
                }
            }
        else:
            # === Step 1: Extract content after </think> for semantic comparison ===
            # print(f"    Extracting final answer parts from {n_answers} answers...")  # Simplified output
            processed_outputs = []
            has_think_tag = []
            has_valid_boxed = []
            invalid_boxed_indices = []
            found_think_count = 0
            found_boxed_count = 0
            found_valid_boxed_count = 0
            
            for i, output in enumerate(pred_outputs):
                after_think = self._extract_after_think(output)
                processed_outputs.append(after_think)
                
                # Check if format requirements are met
                has_think = after_think != output
                has_boxed = "\\boxed" in after_think
                
                # Check if \boxed content contains numbers
                boxed_has_numbers = self._check_boxed_content_has_numbers(after_think)
                has_valid_boxed.append(boxed_has_numbers)
                
                # If no \boxed or \boxed contains no numbers, mark as invalid answer
                if not boxed_has_numbers:
                    invalid_boxed_indices.append(i)
                
                meets_requirement = True
                has_think_tag.append(meets_requirement)
                
                if has_think:
                    found_think_count += 1
                if has_boxed:
                    found_boxed_count += 1
                if boxed_has_numbers:
                    found_valid_boxed_count += 1
                
                # Debug: simplified sample output
                # if i < 2 and self.debug_mode:
                #     if self.force_think_format:
                #         print(f"      Sample {i+1}: </think>={'✓' if has_think else '✗'}, "
                #               f"\\boxed={'✓' if has_boxed else '✗'}, "
                #               f"\\boxed has numbers={'✓' if boxed_has_numbers else '✗'}, "
                #               f"meets requirements={'✓' if meets_requirement else '✗'}")
                #     else:
                #         print(f"      Sample {i+1}: \\boxed has numbers={'✓' if boxed_has_numbers else '✗'}")
                #         if boxed_has_numbers:
                #             print(f"      Extracted content: {after_think[:100]}...")
            
            # if self.debug_mode:
            #     print(f"    Format check statistics: </think> tags={found_think_count}/{n_answers}, "
            #           f"\\boxed has numbers={found_valid_boxed_count}/{n_answers}")
            
            # === Handle \boxed invalid answers: directly give -1 reward ===
            if invalid_boxed_indices:
                # print(f"    Found {len(invalid_boxed_indices)} \\boxed invalid answers, directly giving -1 reward...")
                
                final_rewards_with_invalid = [-1.0] * n_answers
                
                # Record invalid boxed statistics and include complete necessary keys
                self._novelty_stats = {
                    "answer_groups": {
                        "n_correct": sum(1 for r in base_rewards if r > 0),
                        "n_incorrect": sum(1 for r in base_rewards if r <= 0),
                        "n_invalid_boxed": len(invalid_boxed_indices),
                        "n_positive_rewards": 0,
                        "n_negative_rewards": len(invalid_boxed_indices),
                        "n_zero_rewards": 0
                    }
                }
                
                if len(invalid_boxed_indices) == n_answers:
                    # print(f"    All answers are \\boxed invalid, all given -1 reward")
                    final_rewards = final_rewards_with_invalid
                else:
                    # Filter out valid answers to continue similarity calculation
                    valid_indices = [i for i in range(n_answers) if i not in invalid_boxed_indices]
                    # print(f"    Calculating novelty for {len(valid_indices)} \\boxed valid answers...")
                    
                    # Rebuild data containing only valid answers
                    processed_outputs = [processed_outputs[i] for i in valid_indices]
                    base_rewards = [base_rewards[i] for i in valid_indices]
                    n_answers = len(processed_outputs)
                    
                    if n_answers <= 1:
                        # print(f"    Insufficient valid answers (<2), cannot calculate novelty")
                        final_rewards = final_rewards_with_invalid
                    else:
                        # Calculate novelty rewards
                        computed_rewards = self._compute_similarity_based_rewards(processed_outputs, base_rewards)
                        
                        # Map calculated novelty rewards back to original positions
                        for i, valid_idx in enumerate(valid_indices):
                            final_rewards_with_invalid[valid_idx] = computed_rewards[i]
                        
                        final_rewards = final_rewards_with_invalid
            else:
                # Normal mode: calculate novelty for all answers
                final_rewards = self._compute_similarity_based_rewards(processed_outputs, base_rewards)
        
        
        return final_rewards

    def _compute_similarity_based_rewards(self, processed_outputs: list[str], base_rewards: list[float]) -> list[float]:
        """
        Core logic for calculating novelty rewards based on similarity
        
        New reward calculation logic:
        1. Novelty calculation: Average similarity calculated within groups, maximum similarity calculated globally
        2. Novelty normalization: Normalized within correct answer groups, normalized within incorrect answer groups to [0,1]
        3. For correct answers: All correct answers receive positive rewards
           - All correct answers rescaled to [0.5,1] range (higher novelty = higher reward)
        4. For incorrect answers: All incorrect answers handled uniformly
           - All incorrect answers: rescaled to [-1,-0.5] range (higher novelty = less punishment)
        5. Simplified reward strategy: Correct answers [0.5,1], incorrect answers [-1,-0.5], adjusted based on novelty
        """
        n_answers = len(processed_outputs)
        
        # Initialize novelty statistics collection
        novelty_stats = {
            "answer_groups": {},
            "novelty_scores": {},
            "rewards": {},
            "similarity": {}
        }
        
        # === Step 2: Batch get semantic embeddings for processed texts ===
        # print(f"    Batch calculating semantic embeddings for {n_answers} final answers...")  # Simplified output
        embeddings = self._get_semantic_embeddings_batch(processed_outputs)

        # === Step 3: Calculate similarity matrix ===
        # print(f"    Calculating {n_answers}x{n_answers} similarity matrix...")  # Simplified output
        
        # Convert embedding list to matrix
        embedding_matrix = np.array(embeddings, dtype=np.float32)
        
        # Calculate L2 norm for normalization
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        
        # Normalize embedding matrix
        normalized_embeddings = embedding_matrix / norms
        
        # Calculate cosine similarity matrix
        cosine_sim_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Clip to [0,1] range
        sim_matrix = np.clip(cosine_sim_matrix, 0.0, 1.0)
        
        # Record similarity statistics
        upper_triangle_mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
        similarity_values = sim_matrix[upper_triangle_mask]
        novelty_stats["similarity"]["mean"] = float(np.mean(similarity_values))
        novelty_stats["similarity"]["std"] = float(np.std(similarity_values))
        novelty_stats["similarity"]["min"] = float(np.min(similarity_values))
        novelty_stats["similarity"]["max"] = float(np.max(similarity_values))
        
        # === Step 4: Group by correctness ===
        correct_indices = [i for i, r in enumerate(base_rewards) if r > 0]
        incorrect_indices = [i for i, r in enumerate(base_rewards) if r <= 0]
        
        print(f"    Semantic novelty calculation: {len(correct_indices)} correct answers, {len(incorrect_indices)} incorrect answers")
        
        # Record answer group statistics
        novelty_stats["answer_groups"]["n_correct"] = len(correct_indices)
        novelty_stats["answer_groups"]["n_incorrect"] = len(incorrect_indices)
        
        # === Step 5: Calculate normalized novelty scores ===
        # print(f"    Calculating normalized novelty scores (average: within groups, maximum: globally)...")  # Simplified output
        
        novelty_scores = np.zeros(n_answers)
        
        for i in range(n_answers):
            is_correct = base_rewards[i] > 0
            
            # Average similarity: only calculated within same group
            if is_correct:
                # Correct answers: only calculate average similarity with other correct answers
                group_indices = [j for j in correct_indices if j != i]
            else:
                # Incorrect answers: only calculate average similarity with other incorrect answers
                group_indices = [j for j in incorrect_indices if j != i]
            
            # Maximum similarity: calculated with all other answers
            all_other_indices = [j for j in range(n_answers) if j != i]
            
            if group_indices:
                # Intra-group average similarity
                group_similarities = [sim_matrix[i, j] for j in group_indices]
                avg_similarity = np.mean(group_similarities)
            else:
                # If only one answer in group, use global average similarity
                if all_other_indices:
                    all_similarities = [sim_matrix[i, j] for j in all_other_indices]
                    avg_similarity = np.mean(all_similarities)
                else:
                    avg_similarity = 0.5  # Global single answer case
            
            if all_other_indices:
                # Global maximum similarity
                all_similarities = [sim_matrix[i, j] for j in all_other_indices]
                max_similarity = np.max(all_similarities)
            else:
                max_similarity = 0.5  # Global single answer case
            
            # Use weighted combination: 50% average similarity + 50% maximum similarity
            combined_similarity = 0.5 * avg_similarity + 0.5 * max_similarity
            novelty_scores[i] = 1.0 - combined_similarity
            
            # Debug info - simplified output, removed detailed similarity information
            # if self.debug_mode and i < 2:
            #     group_type = "correct" if is_correct else "incorrect"
            #     print(f"      Answer {i+1}({group_type}): intra-group avg similarity={avg_similarity:.3f}, "
            #           f"global max similarity={max_similarity:.3f}, novelty={novelty_scores[i]:.3f}")
        
        # === Step 6: Normalize novelty scores by group to [0,1] range ===
        # print(f"    Normalizing novelty scores by group...")  # Simplified output
        
        # Normalize correct answer group and incorrect answer group separately
        if correct_indices and len(correct_indices) > 1:
            correct_novelty_scores = novelty_scores[correct_indices]
            min_correct = np.min(correct_novelty_scores)
            max_correct = np.max(correct_novelty_scores)
            if max_correct > min_correct:
                normalized_correct = (correct_novelty_scores - min_correct) / (max_correct - min_correct)
                for i, idx in enumerate(correct_indices):
                    novelty_scores[idx] = normalized_correct[i]
            # if self.debug_mode:
            #     print(f"      Correct answer group normalization: [{min_correct:.3f}, {max_correct:.3f}] -> [0, 1]")
        elif correct_indices:
            # Only one correct answer, set to medium novelty
            for idx in correct_indices:
                novelty_scores[idx] = 0.5
            # if self.debug_mode:
            #     print(f"      Correct answer group: only 1 answer, set to 0.5")
        
        if incorrect_indices and len(incorrect_indices) > 1:
            incorrect_novelty_scores = novelty_scores[incorrect_indices]
            min_incorrect = np.min(incorrect_novelty_scores)
            max_incorrect = np.max(incorrect_novelty_scores)
            if max_incorrect > min_incorrect:
                normalized_incorrect = (incorrect_novelty_scores - min_incorrect) / (max_incorrect - min_incorrect)
                for i, idx in enumerate(incorrect_indices):
                    novelty_scores[idx] = normalized_incorrect[i]
            # if self.debug_mode:
            #     print(f"      Incorrect answer group normalization: [{min_incorrect:.3f}, {max_incorrect:.3f}] -> [0, 1]")
        elif incorrect_indices:
            # Only one incorrect answer, set to medium novelty
            for idx in incorrect_indices:
                novelty_scores[idx] = 0.5
            # if self.debug_mode:
            #     print(f"      Incorrect answer group: only 1 answer, set to 0.5")
        
        # Record novelty score statistics
        novelty_stats["novelty_scores"]["mean"] = float(np.mean(novelty_scores))
        novelty_stats["novelty_scores"]["std"] = float(np.std(novelty_scores))
        novelty_stats["novelty_scores"]["min"] = float(np.min(novelty_scores))
        novelty_stats["novelty_scores"]["max"] = float(np.max(novelty_scores))
        
        # Group novelty statistics
        if correct_indices:
            correct_novelty = novelty_scores[correct_indices]
            novelty_stats["novelty_scores"]["correct"] = {
                "mean": float(np.mean(correct_novelty)),
                "std": float(np.std(correct_novelty))
            }
        
        if incorrect_indices:
            incorrect_novelty = novelty_scores[incorrect_indices]
            novelty_stats["novelty_scores"]["incorrect"] = {
                "mean": float(np.mean(incorrect_novelty)),
                "std": float(np.std(incorrect_novelty))
            }
        
        # === Step 7: Assign rewards for correct answers ===
        # Simplified output - remove detailed reward assignment information

        final_rewards = np.zeros(n_answers)

        if correct_indices:
            correct_novelty_scores = novelty_scores[correct_indices]
            n_correct = len(correct_indices)

            # All correct answers assigned to [0.5,1] range, rescaled based on novelty
            if len(correct_indices) > 1:
                min_correct = min(correct_novelty_scores)
                max_correct = max(correct_novelty_scores)
                if max_correct > min_correct:
                    # Rescale to [0.5,1] range
                    for i, idx in enumerate(correct_indices):
                        novelty = novelty_scores[idx]
                        rescaled_novelty = (novelty - min_correct) / (max_correct - min_correct)
                        final_rewards[idx] = 0.5 + 0.5 * rescaled_novelty  # Map to [0.5,1]
                else:
                    # All correct answers have same novelty, give medium reward
                    for idx in correct_indices:
                        final_rewards[idx] = 0.75  # Midpoint of [0.5,1]
            else:
                # Only one correct answer, give highest reward
                final_rewards[correct_indices[0]] = 1.0

            # if self.debug_mode:
            #     print(f"      Correct answers: {n_correct} answers all rescaled to [0.5,1] range")
            #     for i, idx in enumerate(correct_indices[:2]):  # Only show first 2
            #         print(f"      Correct answer {idx+1}: original novelty={novelty_scores[idx]:.3f}, final reward={final_rewards[idx]:.3f}")
            
            # Record correct answer reward statistics
            novelty_stats["answer_groups"]["n_positive_rewards"] = len(correct_indices)
            novelty_stats["answer_groups"]["n_negative_rewards"] = 0
            novelty_stats["answer_groups"]["n_zero_rewards"] = 0
            
            correct_rewards = [final_rewards[idx] for idx in correct_indices]
            novelty_stats["rewards"]["positive_rewards"] = {
                "mean": float(np.mean(correct_rewards)),
                "std": float(np.std(correct_rewards))
            }
            
            # Clear negative reward statistics
            novelty_stats["rewards"]["negative_rewards"] = {
                "mean": 0.0,
                "std": 0.0
            }
        
        # === Step 8: Assign rewards for incorrect answers ===
        # print(f"    Assigning rewards for incorrect answers (all incorrect answers in [-1,-0.5] range)...")  # Simplified output

        if incorrect_indices:
            incorrect_novelty_scores = novelty_scores[incorrect_indices]
            n_incorrect = len(incorrect_indices)

            # All incorrect answers assigned to [-1,-0.5] range, rescaled based on novelty
            if len(incorrect_indices) > 1:
                min_incorrect = min(incorrect_novelty_scores)
                max_incorrect = max(incorrect_novelty_scores)
                if max_incorrect > min_incorrect:
                    # Rescale to [-1,-0.5] range
                    for i, idx in enumerate(incorrect_indices):
                        novelty = novelty_scores[idx]
                        rescaled_novelty = (novelty - min_incorrect) / (max_incorrect - min_incorrect)
                        final_rewards[idx] = -1.0 + 0.5 * rescaled_novelty  # Map to [-1,-0.5]
                else:
                    # All incorrect answers have same novelty, give medium reward
                    for idx in incorrect_indices:
                        final_rewards[idx] = -0.75  # Midpoint of [-1,-0.5]
            else:
                # Only one incorrect answer, give relatively high error reward
                final_rewards[incorrect_indices[0]] = -0.5

            
            
            # Record incorrect answer reward statistics
            incorrect_rewards = [final_rewards[idx] for idx in incorrect_indices]
            novelty_stats["rewards"]["incorrect_novel_rewards"] = {
                "mean": float(np.mean(incorrect_rewards))
            }
            
            # Clear group statistics
            novelty_stats["rewards"]["incorrect_not_novel_rewards"] = {
                "mean": 0.0
            }
        

        
        # Record overall reward statistics
        all_rewards = final_rewards.flatten() if isinstance(final_rewards, np.ndarray) else final_rewards
        novelty_stats["rewards"]["mean"] = float(np.mean(all_rewards))
        novelty_stats["rewards"]["std"] = float(np.std(all_rewards))
        novelty_stats["rewards"]["min"] = float(np.min(all_rewards))
        novelty_stats["rewards"]["max"] = float(np.max(all_rewards))
        
        # Save statistics to class member variable
        self._novelty_stats = novelty_stats
        
        # Add concise summary output
        if self.debug_mode:
            avg_correct_reward = np.mean([final_rewards[i] for i in correct_indices]) if correct_indices else 0
            avg_incorrect_reward = np.mean([final_rewards[i] for i in incorrect_indices]) if incorrect_indices else 0
            print(f"    Semantic novelty completed: correct answer avg reward={avg_correct_reward:.3f}, incorrect answer avg reward={avg_incorrect_reward:.3f}")

        return final_rewards.tolist()


    def _data_source_to_task(self, data_source):
        """
        Infer task type based on data source name
        """
        if data_source in ["MATH-TTT", "AIME-TTT", "AMC-TTT", "AIME25"]:
            return "math"
        elif data_source in ["GPQA-TTT"]:
            return "gpqa"
        else:
            return data_source
            # raise NotImplementedError(f"Data source {data_source} not supported for TTRLRewardManager")

    def _compute_similarity_rewards(self, processed_outputs: list[str]):
        n_answers = len(processed_outputs)
        
        # Initialize novelty statistics collection
        novelty_stats = {
            "answer_groups": {},
            "novelty_scores": {},
            "rewards": {},
            "similarity": {}
        }
        
        # === Step 2: Batch get semantic embeddings for processed texts ===
        # print(f"    Batch calculating semantic embeddings for {n_answers} final answers...")  # Simplified output
        embeddings = self._get_semantic_embeddings_batch(processed_outputs)

        # === Step 3: Calculate similarity matrix ===
        # print(f"    Calculating {n_answers}x{n_answers} similarity matrix...")  # Simplified output
        
        # Convert embedding list to matrix
        embedding_matrix = np.array(embeddings, dtype=np.float32)
        
        # Calculate L2 norm for normalization
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        
        # Normalize embedding matrix
        normalized_embeddings = embedding_matrix / norms
        
        # Calculate cosine similarity matrix
        cosine_sim_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Clip to [0,1] range
        sim_matrix = np.clip(cosine_sim_matrix, 0.0, 1.0)

        novelty_scores = np.zeros(n_answers)
        
        for i in range(n_answers):
            all_other_indices = [j for j in range(n_answers) if j != i]
            all_similarities = [sim_matrix[i, j] for j in all_other_indices]
            avg_similarity = np.std(all_similarities)
            novelty_scores[i] = 1 - avg_similarity
        final_rewards = np.zeros(n_answers)
        # todo: apply reward to correct answer only?
        min_score = np.min(novelty_scores)
        max_score = np.max(novelty_scores)
        if max_score > min_score:
            for i in range(n_answers):
                novelty = novelty_scores[i]
                rescaled = (novelty - min_score)/(max_score-min_score)
                final_rewards[i] = rescaled
        else:
            final_rewards = 0.5

        

        return final_rewards.tolist()
    

    def _compute_ttrl_reward(self, data: DataProto):
        """
        Calculate TTRL rewards during training (with semantic novelty adjustment)
        
        This is the core method for training mode, responsible for:
        1. Processing data by prompt groups
        2. Using test_time_train_metrics to calculate base rewards (based on voting)
        3. Applying semantic novelty adjustment (if enabled)
        4. Building reward tensor for training use
        
        Args:
            data (DataProto): Training data containing prompts, responses, etc.
            
        Returns:
            tuple: (reward_tensor, reward_extra_info, ttrl_info)
                - reward_tensor: Reward tensor with shape (batch_size, seq_len)
                - reward_extra_info: Additional reward information
                - ttrl_info: TTRL training metrics
        """
        print(f"Starting TTRL reward calculation during training (semantic novelty: {'enabled' if self.use_semantic_novelty else 'disabled'})...")
        
        reward_extra_info = defaultdict(list)
        ttrl_info = {}
        if len(data) % self.n_samples_per_prompt != 0: # eval data pass for now
            return torch.zeros_like(
            data.batch["responses"], 
            dtype=torch.float32
        ), reward_extra_info, ttrl_info
        
        # Data validity check
        assert len(data) % self.n_samples_per_prompt == 0, \
            f"Data length ({len(data)}) must be divisible by n_votes_per_prompt ({self.n_samples_per_prompt})"
        
        prompt_num = len(data) // self.n_samples_per_prompt
        print(f"  Processing {prompt_num} prompts, each with {self.n_samples_per_prompt} candidate answers")
        
        # Initialize reward tensor: only allocate space for samples actually used for training
        reward_tensor = torch.zeros_like(
            data.batch["responses"][:prompt_num * self.n_samples_per_prompt], 
            dtype=torch.float32
        )
        
        already_print_data_sources = {}  # Counter to control debug output
        all_ttrl_metrics = defaultdict(list)  # Collect all TTRL metrics
        scores = [0.0 for _ in range(len(data))]  # Scores for all samples
        seen = {}
        prompts_list = []
        for i, data_item in enumerate(data):
            prompt_idx = data_item.batch["prompts"]
            prompt_length = prompt_idx.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_idx = prompt_idx[-valid_prompt_length:]
            prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)    

            response_idx = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            if prompt_str in seen:
                seen[prompt_str].append((i,data_item, valid_response_length))
            else:
                prompts_list.append(prompt_str)
                seen[prompt_str] = [(i, data_item , valid_response_length)]
        
        # === Process by prompt groups ===
        for prompt_i in range(prompt_num):
            print(f"  Processing prompt {prompt_i+1}/{prompt_num}...")
            
            group_pred_outputs = []  # All predicted outputs for current prompt
            group_labels = []        # All ground truth for current prompt
            group_extra_info = []    # Extra information for current prompt
            task = None
            
            # === Collect all candidate answers for current prompt ===
            for i in range(self.n_samples_per_prompt):
                # data_item = data[prompt_i * self.n_samples_per_prompt + i]
                data_item = seen[prompts_list[prompt_i]][i][1]

                
                # Parse token sequences
                prompt_idx = data_item.batch["prompts"]
                prompt_length = prompt_idx.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_idx = prompt_idx[-valid_prompt_length:]
                
                response_idx = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_idx = response_idx[:valid_response_length]

                # Decode to text
                prompt_str = self.tokenizer.decode(valid_prompt_idx, skip_special_tokens=False)
                response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                print(prompt_i, i, prompt_str)
                # Get metadata
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch["extra_info"]
                
                # Task type check
                if task is None:
                    task = self._data_source_to_task(data_source)
                else:
                    if task != self._data_source_to_task(data_source):
                        raise NotImplementedError(f"Task type inconsistent: {task} vs {self._data_source_to_task(data_source)}")

                # Collect current group data (consistent with ttrl.py, directly use ground_truth)
                group_labels.append(ground_truth)
                group_pred_outputs.append(response_str)
                group_extra_info.append(extra_info)
            
            # === Calculate base rewards (correctness judgment based on voting) ===
            # base_rewards, ttrl_metrics = test_time_train_metrics(
            #     group_pred_outputs, group_labels, task=task, extra_info=group_extra_info
            # )

            # === Calculate strategy entropy ===
            # current_group_data = data[prompt_i * self.n_samples_per_prompt:(prompt_i + 1) * self.n_samples_per_prompt]
            # strategy_entropy = self._compute_strategy_entropy(current_group_data)
            # ttrl_metrics["neg_log_likelihood"] = strategy_entropy
            # if self.debug_mode and strategy_entropy > 0:
            #     print(f"    Strategy entropy: H_ttrl={strategy_entropy:.3f} (normalized negative log-likelihood)")

            # === Apply semantic novelty adjustment ===
            # if self.use_semantic_novelty:
            #     print(f"    Applying semantic novelty adjustment...")
            final_rewards = self._compute_similarity_rewards(group_pred_outputs)
            # else:
            #     print(f"    Using original TTRL rewards...")
            #     final_rewards = base_rewards

            # Accumulate TTRL metrics
            # for k, v in ttrl_metrics.items():
            #     all_ttrl_metrics[k].append(v)

            # === Fill reward tensor and score list ===  
            for i in range(self.n_samples_per_prompt):
                current_reward = final_rewards[i]
                
                # Only fill reward tensor for samples used for training
                if i < self.n_samples_per_prompt:
                    # Place reward at the last valid token position of response
                    # reward_tensor[prompt_i * self.n_samples_per_prompt + i, valid_response_length - 1] = current_reward
                    reward_tensor[seen[prompts_list[prompt_i]][i][0], seen[prompts_list[prompt_i]][i][2] - 1] = current_reward


                # Record scores for all samples (including those not used for training)
                # scores[prompt_i * self.n_samples_per_prompt + i] = current_reward
                scores[seen[prompts_list[prompt_i]][i][0]] = current_reward

                # === Debug output ===
                # data_item = data[prompt_i * self.n_samples_per_prompt + i]
                data_item = seen[prompts_list[prompt_i]][i][1]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                    
                # Control output count for each data source
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                        
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                        
                    print(f"\n    === Sample Debug Output ===")
                    print(f"    [prompt] {self.tokenizer.decode(data_item.batch['prompts'][-data_item.batch['attention_mask'][:data_item.batch['prompts'].shape[-1]].sum():], skip_special_tokens=False)}")
                    print(f"    [response] {group_pred_outputs[i]}")
                    print(f"    [final_score] {current_reward:.4f}")
                    # if self.use_semantic_novelty:
                    #     print(f"    [base_reward] {base_rewards[i]:.4f}")

        # === Update accuracy field in data ===
        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=data.batch["prompts"].device)
        
        # === Calculate and output average metrics ===
        print(f"\n=== TTRL Training Metrics Summary ===")
        for k, v in all_ttrl_metrics.items():
            if isinstance(v, list):
                avg_v = np.mean(v)
                print(f"[{k}] {avg_v:.4f}")
                ttrl_info[k] = avg_v
                
        return reward_tensor, reward_extra_info, ttrl_info

    def _compute_eval_reward(self, data: DataProto):
        """
        Calculate rewards during evaluation (with semantic novelty adjustment)
        
        Main differences between evaluation mode and training mode:
        1. Use auto_verify instead of test_time_train_metrics to judge correctness
        2. No need to distinguish votes and samples, all data used for evaluation
        3. Still group by prompt to support semantic novelty calculation
        
        Args:
            data (DataProto): Evaluation data
            
        Returns:
            tuple: (reward_tensor, reward_extra_info, ttrl_info)
        """
        print(f"Starting reward calculation during evaluation (semantic novelty: {'enabled' if self.use_semantic_novelty else 'disabled'})...")
        
        reward_extra_info = defaultdict(list)
        ttrl_info = {}
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print_data_sources = {}

        # Data validity check
        assert len(data) % self.eval_n_samples == 0, \
            f"Evaluation data length ({len(data)}) must be divisible by eval_n_samples ({self.eval_n_samples})"

        prompt_num = len(data) // self.eval_n_samples
        print(f"  Processing {prompt_num} prompts, each with {self.eval_n_samples} samples")

        # === Collect all data and group by task to avoid mixed task crashes ===
        group_pred_outputs = []
        group_labels = []
        group_extra_info = []
        sample_valid_resp_len: dict[int, int] = {}
        task_groups = {}

        for i in range(len(data)):
            data_item = data[i]
            # Decode and get valid length
            prompt_str, response_str, valid_response_length = self._decode_data_item(data_item)
            sample_valid_resp_len[i] = int(valid_response_length)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch["extra_info"]

            group_labels.append(ground_truth)
            group_pred_outputs.append(response_str)
            group_extra_info.append(extra_info)

            # Debug output
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)

            task_key = self._data_source_to_task(data_source)
            if task_key not in task_groups:
                task_groups[task_key] = {"indices": [], "outputs": [], "labels": [], "extra": []}
            task_groups[task_key]["indices"].append(i)
            task_groups[task_key]["outputs"].append(response_str)
            task_groups[task_key]["labels"].append(ground_truth)
            task_groups[task_key]["extra"].append(extra_info)

        # === Verify by task separately and merge results ===
        verify_bool_results = [False] * len(data)
        for task_key, group in task_groups.items():
            task_verify, verify_extra = auto_verify(task_key, group["outputs"], group["labels"], extra_info=group["extra"])
            # Aggregate extra information
            for k, v in verify_extra.items():
                if isinstance(v, list):
                    reward_extra_info[k] += v
            # Backfill verification results
            for local_idx, sample_idx in enumerate(group["indices"]):
                verify_bool_results[sample_idx] = bool(task_verify[local_idx])

        # Convert boolean results to +1/-1 rewards
        base_rewards = [1.0 if ok else -1.0 for ok in verify_bool_results]

        # === Apply semantic novelty adjustment by prompt groups ===
        if self.use_semantic_novelty:
            print(f"  Applying semantic novelty adjustment (by prompt groups)...")
            final_rewards = []
            
            for prompt_i in range(prompt_num):
                start_idx = prompt_i * self.eval_n_samples
                end_idx = start_idx + self.eval_n_samples
                
                prompt_pred_outputs = group_pred_outputs[start_idx:end_idx]
                prompt_base_rewards = base_rewards[start_idx:end_idx]
                
                prompt_final_rewards = self._compute_novelty_rewards(prompt_pred_outputs, prompt_base_rewards)
                final_rewards.extend(prompt_final_rewards)
        else:
            print(f"  Using original verification rewards...")
            final_rewards = base_rewards

        # === Fill reward tensor (using valid length of each sample) ===
        for i in range(len(data)):
            vlen = sample_valid_resp_len.get(i, 0)
            if vlen > 0:
                reward_tensor[i, vlen - 1] = final_rewards[i]

        # === Calculate TTRL metrics (for consistency evaluation) ===
        print(f"\n=== Calculating TTRL Evaluation Metrics ===")
        all_ttrl_metrics = defaultdict(list)
        
        for prompt_i in range(prompt_num):
            group_pred_outputs_ttrl = []
            group_labels_ttrl = []
            group_extra_info_ttrl = []
            
            # Re-collect data (for calculating TTRL metrics)
            for i in range(self.eval_n_samples):
                idx = prompt_i * self.eval_n_samples + i
                data_item = data[idx]
                
                # Decode response
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_idx = data_item.batch["responses"][:valid_response_length]
                response_str = self.tokenizer.decode(valid_response_idx, skip_special_tokens=False)
                
                # Get ground truth and extra information
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                extra_info = data_item.non_tensor_batch["extra_info"]
                
                group_pred_outputs_ttrl.append(response_str)
                group_labels_ttrl.append(ground_truth)  # Consistent with ttrl.py, directly use ground_truth
                group_extra_info_ttrl.append(extra_info)
            
            # Determine task type for this prompt
            first_ds = data[prompt_i * self.eval_n_samples].non_tensor_batch[self.reward_fn_key]
            group_task = self._data_source_to_task(first_ds)
            
            # Calculate TTRL metrics for current prompt
            _, ttrl_metrics = test_time_train_metrics(
                group_pred_outputs_ttrl, group_labels_ttrl, task=group_task, extra_info=group_extra_info_ttrl
            )

            # === Calculate strategy entropy ===
            current_group_data = data[prompt_i * self.eval_n_samples:(prompt_i + 1) * self.eval_n_samples]
            strategy_entropy = self._compute_strategy_entropy(current_group_data)
            ttrl_metrics["neg_log_likelihood"] = strategy_entropy
            
            for k, v in ttrl_metrics.items():
                all_ttrl_metrics[k].append(v)
        
        # Output average TTRL metrics
        for k, v in all_ttrl_metrics.items():
            if isinstance(v, list):
                avg_v = np.mean(v)
                print(f"[{k}] {avg_v:.4f}")
                ttrl_info[k] = avg_v
        
        return reward_tensor, reward_extra_info, ttrl_info

    def __call__(self, data: DataProto, return_dict=False):
        """
        Main calling interface for reward manager
        
        Automatically selects appropriate reward calculation method based on current mode (train/eval).
        This is the main entry point for external calls. Each call automatically increments internal step counter for wandb logging.
        
        Args:
            data (DataProto): Input data containing prompts, responses, etc.
            return_dict (bool): Whether to return detailed information dictionary
                - False: Only return reward tensor
                - True: Return dictionary containing reward tensor, extra information and metrics
                
        Returns:
            torch.Tensor or dict: 
                - If return_dict=False: Return reward tensor
                - If return_dict=True: Return dictionary with the following keys:
                    * "reward_tensor": Reward tensor
                    * "reward_extra_info": Additional reward information
                    * "ttrl_info": TTRL training/evaluation metrics
                    
        Example:
            >>> manager = TTRLRewardManager(tokenizer, ...)
            >>> 
            >>> # Simple call, only get reward tensor (automatically logged to wandb)
            >>> rewards = manager(data)
            >>> 
            >>> # Detailed call, get complete information (automatically logged to wandb)
            >>> result = manager(data, return_dict=True)
            >>> rewards = result["reward_tensor"]
            >>> metrics = result["ttrl_info"]
        """
        # Increment internal step counter
        if self.use_semantic_novelty:
            self._internal_step += 1
        
        print(f"\n{'='*50}")
        print(f"TTRLRewardManager execution started")
        print(f"Mode: {self.mode}")
        print(f"Data size: {len(data)}")
        print(f"Semantic novelty: {'enabled' if self.use_semantic_novelty else 'disabled'}")
        if self.use_semantic_novelty:
            print(f"Internal step: {self._internal_step}")
        print(f"{'='*50}")
        print("First data: ",data[0])
        # Select appropriate calculation method based on mode
        if self.mode == "train":
            reward_tensor, reward_extra_info, ttrl_info = self._compute_ttrl_reward(data)
        elif self.mode == "eval":
            reward_tensor, reward_extra_info, ttrl_info = self._compute_ttrl_reward(data)
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}, please use 'train' or 'eval'")

        print(f"\n{'='*50}")
        print(f"TTRLRewardManager execution completed")
        print(f"Reward tensor shape: {reward_tensor.shape}")
        print(f"Average reward: {reward_tensor.mean().item():.4f}")
        print(f"Reward range: [{reward_tensor.min().item():.4f}, {reward_tensor.max().item():.4f}]")
        
        # Temporarily disable additional wandb logging to avoid step conflicts
        # TODO: Use different metric names or synchronize step with main training loop
        # if self.use_semantic_novelty:
        #     self.log_api_stats_to_wandb(step=self._internal_step)
        #     self.log_novelty_stats_to_wandb(step=self._internal_step)
        
        print(f"{'='*50}\n")

        # Return result based on return format requirements
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "ttrl_info": ttrl_info,
            }
        else:
            return reward_tensor 

"""
Usage Example - Semantic Novelty Reward Manager
==============================================

# Simplified reward allocation logic:
# - Novelty calculation: Average similarity calculated within groups, maximum similarity calculated globally, normalized to [0,1] within each group
# - Correct answers: All correct answers rescaled to [0.5,1] range (higher novelty = higher reward)
# - Incorrect answers: All incorrect answers rescaled to [-1,-0.5] range (higher novelty = less punishment)
# - \\boxed invalid answers: Directly give -1 reward
# - Simplified reward strategy: Correct answers [0.5,1], incorrect answers [-1,-0.5], adjusted based on novelty

# Create reward manager example:
reward_manager = SemanticTTRLRewardManager(
    tokenizer=tokenizer,
    num_examine=2,
    use_semantic_novelty=True,
    embedding_dim=2560,
    vllm_api_url="http://30.159.163.76:2341",
    vllm_model_name="Qwen/Qwen3-Embedding-4B"
)

# Use reward manager:
rewards = reward_manager(data)
""" 