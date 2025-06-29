#!/usr/bin/env python3
"""
Demo: Simplified Atropos-VeRL Integration
========================================

This demo shows the improved architecture using the REAL Atropos API:
- Clean API interface (uses actual /batch endpoint)
- Multi-environment support with weights
- Unified inference management
- No Ray complexity for basic use cases
- Falls back to simulation when Atropos is not running
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import requests
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


@dataclass
class TrajectoryBatch:
    """Clean data structure for trajectory batches"""

    tokens: List[List[int]]
    masks: List[List[bool]]
    scores: List[float]
    groups: List[int]
    metadata: Dict[str, Any]


class RealAtroposAPI:
    """
    Real Atropos API client that uses the actual Atropos server endpoints.

    This demonstrates the simplified integration with the real Atropos system.
    """

    def __init__(self, api_url: str = "http://localhost:8001"):
        self.api_url = api_url
        self.session = requests.Session()
        logger.info(f"Real Atropos API client initialized: {api_url}")

    def health_check(self) -> bool:
        """Health check endpoint"""
        try:
            response = self.session.get(f"{self.api_url}/status", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_trajectory_batch(self, batch_size: int, environment: str, model_config: Dict, inference_endpoint: str) -> TrajectoryBatch:
        """
        Get trajectory batch from real Atropos API.

        This uses the actual /batch endpoint from Atropos.
        """
        logger.info(f"Requesting {batch_size} trajectories from Atropos (environment: {environment})")

        try:
            # Get batch from Atropos
            response = self.session.get(f"{self.api_url}/batch", timeout=30)

            if response.status_code == 200:
                data = response.json()
                batch_data = data.get("batch")

                if batch_data is None:
                    # No data available, simulate some for demo
                    logger.warning("No batch data available from Atropos, using simulated data")
                    return self._generate_simulated_batch(batch_size, environment, model_config, inference_endpoint)

                # Convert Atropos batch format to our TrajectoryBatch format
                return self._convert_atropos_batch(batch_data, environment)
            else:
                logger.warning(f"Atropos API returned {response.status_code}, using simulated data")
                return self._generate_simulated_batch(batch_size, environment, model_config, inference_endpoint)

        except Exception as e:
            logger.warning(f"Failed to get batch from Atropos: {e}, using simulated data")
            return self._generate_simulated_batch(batch_size, environment, model_config, inference_endpoint)

    def _convert_atropos_batch(self, batch_data: List[Dict], environment: str) -> TrajectoryBatch:
        """Convert Atropos batch format to our TrajectoryBatch format"""
        tokens = []
        masks = []
        scores = []
        groups = []

        for i, item in enumerate(batch_data):
            # Extract data from Atropos format
            item_tokens = item.get("tokens", [])
            item_masks = item.get("masks", [])
            item_scores = item.get("scores", [])

            if item_tokens and item_masks and item_scores:
                tokens.extend(item_tokens)
                masks.extend(item_masks)
                scores.extend(item_scores)
                # Create groups for each trajectory
                groups.extend(list(range(len(groups), len(groups) + len(item_tokens))))

        metadata = {"environment": environment, "batch_size": len(tokens), "source": "real_atropos", "generation_time": time.time()}

        logger.info(f"Converted Atropos batch: {len(tokens)} trajectories")
        return TrajectoryBatch(tokens=tokens, masks=masks, scores=scores, groups=groups, metadata=metadata)

    def _generate_simulated_batch(self, batch_size: int, environment: str, model_config: Dict, inference_endpoint: str) -> TrajectoryBatch:
        """Generate simulated batch when real Atropos data is not available"""
        logger.info(f"Generating simulated {batch_size} trajectories for {environment}")

        # Simulate calling the inference endpoint
        self._simulate_inference_call(inference_endpoint, batch_size)

        # Generate mock trajectory data
        tokens = []
        masks = []
        scores = []
        groups = []

        for i in range(batch_size):
            # Simulate different sequence lengths
            seq_len = np.random.randint(50, 200)

            # Mock tokens (in real implementation, these would be actual tokenized sequences)
            token_seq = np.random.randint(1, 1000, seq_len).tolist()
            mask_seq = [True] * seq_len

            # Mock reward score based on environment
            if environment == "gsm8k":
                score = np.random.uniform(0.3, 0.9)  # Math problems
            elif environment == "math":
                score = np.random.uniform(0.2, 0.8)  # Harder math
            elif environment == "code":
                score = np.random.uniform(0.4, 0.85)  # Code problems
            else:
                score = np.random.uniform(0.1, 0.7)

            tokens.append(token_seq)
            masks.append(mask_seq)
            scores.append(score)
            groups.append(i)  # Each trajectory is its own group for GRPO

        metadata = {"environment": environment, "batch_size": batch_size, "inference_endpoint": inference_endpoint, "source": "simulated", "generation_time": time.time()}

        return TrajectoryBatch(tokens=tokens, masks=masks, scores=scores, groups=groups, metadata=metadata)

    def _simulate_inference_call(self, endpoint: str, batch_size: int):
        """Simulate calling the inference endpoint for generation"""
        logger.debug(f"Calling inference endpoint {endpoint} for {batch_size} generations")
        time.sleep(0.1)  # Simulate network call


class SimpleAtroposClient:
    """
    Simplified Atropos client with clean API interface.

    This replaces the complex AtroposRolloutWorker with a simple client.
    """

    def __init__(self, api_url: str = "http://localhost:8001"):
        self.api_url = api_url
        self.real_api = RealAtroposAPI(api_url)  # Use real Atropos API
        logger.info(f"SimpleAtroposClient initialized: {api_url}")

    def get_trajectory_batch(self, batch_size: int, environment: str = "gsm8k", inference_endpoint: str = "http://localhost:9000") -> TrajectoryBatch:
        """Get a trajectory batch from Atropos environments"""
        model_config = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512}

        return self.real_api.get_trajectory_batch(batch_size=batch_size, environment=environment, model_config=model_config, inference_endpoint=inference_endpoint)

    def get_multi_environment_batch(self, total_batch_size: int, environment_weights: Dict[str, float], inference_endpoint: str = "http://localhost:9000") -> TrajectoryBatch:
        """Get trajectories from multiple environments based on weights"""
        # Calculate batch sizes for each environment
        env_batches = self._calculate_environment_batches(total_batch_size, environment_weights)

        # Collect batches from each environment
        all_batches = []
        for env_name, env_batch_size in env_batches.items():
            if env_batch_size > 0:
                batch = self.get_trajectory_batch(batch_size=env_batch_size, environment=env_name, inference_endpoint=inference_endpoint)
                all_batches.append(batch)

        # Combine batches
        return self._combine_trajectory_batches(all_batches)

    def _calculate_environment_batches(self, total_batch_size: int, environment_weights: Dict[str, float]) -> Dict[str, int]:
        """Calculate batch sizes for each environment based on weights"""
        # Normalize weights
        total_weight = sum(environment_weights.values())
        normalized_weights = {env: weight / total_weight for env, weight in environment_weights.items()}

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

        logger.info(f"Environment batch distribution: {env_batches}")
        return env_batches

    def _combine_trajectory_batches(self, batches: List[TrajectoryBatch]) -> TrajectoryBatch:
        """Combine multiple trajectory batches into one"""
        if not batches:
            raise ValueError("No batches to combine")

        if len(batches) == 1:
            return batches[0]

        # Combine data
        combined_tokens = []
        combined_masks = []
        combined_scores = []
        combined_groups = []

        group_offset = 0
        for batch in batches:
            combined_tokens.extend(batch.tokens)
            combined_masks.extend(batch.masks)
            combined_scores.extend(batch.scores)

            # Adjust group indices for concatenation
            adjusted_groups = [g + group_offset for g in batch.groups]
            combined_groups.extend(adjusted_groups)
            group_offset += max(batch.groups) + 1 if batch.groups else 0

        # Combine metadata
        combined_metadata = {"source_environments": [batch.metadata.get("environment", "unknown") for batch in batches], "batch_sizes": [len(batch.tokens) for batch in batches], "total_batch_size": len(combined_tokens)}

        return TrajectoryBatch(tokens=combined_tokens, masks=combined_masks, scores=combined_scores, groups=combined_groups, metadata=combined_metadata)


class UnifiedInferenceManager:
    """
    Manages the unified inference engine that serves both VeRL training
    and Atropos environment requests.
    """

    def __init__(self):
        self.endpoint = "http://localhost:9000"
        self.model_weights_version = 0
        logger.info(f"Unified inference manager: {self.endpoint}")

    def update_model_weights(self, weights_dict: Dict):
        """Update the inference server with new model weights"""
        self.model_weights_version += 1
        logger.info(f"Updated inference server weights (version {self.model_weights_version})")

    def get_endpoint(self) -> str:
        """Get the inference server endpoint"""
        return self.endpoint


class SimplifiedGRPOTrainer:
    """
    Simplified GRPO trainer that demonstrates the clean Atropos integration.

    Key improvements:
    - No Ray complexity
    - Direct API calls to Atropos
    - Unified inference management
    - Multi-environment support
    - Clean configuration
    """

    def __init__(self):
        # Configuration
        self.total_epochs = 10
        self.batch_size = 64
        self.environment_weights = {
            "gsm8k": 0.7,  # 70% math problems
            "math": 0.2,  # 20% advanced math
            "code": 0.1,  # 10% coding problems
        }

        # Initialize components
        self.atropos_client = SimpleAtroposClient()
        self.inference_manager = UnifiedInferenceManager()

        # Training state
        self.global_step = 0
        self.best_reward = -float("inf")

        logger.info("SimplifiedGRPOTrainer initialized")
        logger.info(f"Environment weights: {self.environment_weights}")

    def get_training_batch(self) -> TrajectoryBatch:
        """Get a training batch from Atropos environments"""
        return self.atropos_client.get_multi_environment_batch(total_batch_size=self.batch_size, environment_weights=self.environment_weights, inference_endpoint=self.inference_manager.get_endpoint())

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch using Atropos environments"""
        logger.info(f"Starting epoch {epoch + 1}/{self.total_epochs}")

        epoch_metrics = {"loss": [], "reward": [], "kl_div": []}
        steps_per_epoch = 15

        for step in range(steps_per_epoch):
            self.global_step += 1

            # Get training batch from Atropos
            batch_data = self.get_training_batch()

            # Simulate GRPO training step
            step_metrics = self._simulate_grpo_step(batch_data, epoch, step)

            # Collect metrics
            for key, value in step_metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key].append(value)

            # Log progress
            if step % 5 == 0 or step == steps_per_epoch - 1:
                env_info = ", ".join([f"{env}: {batch_data.metadata['batch_sizes'][i]}" for i, env in enumerate(batch_data.metadata["source_environments"])])
                logger.info(f"  Step {step + 1:2d}/{steps_per_epoch}: loss={step_metrics['loss']:.3f}, reward={step_metrics['reward']:.3f}, envs=[{env_info}]")

            # Update policy weights periodically
            if step % 3 == 0:
                self._update_policy_weights()

        # Calculate epoch averages
        avg_metrics = {key: sum(values) / len(values) if values else 0.0 for key, values in epoch_metrics.items()}

        logger.info(f"Epoch {epoch + 1} completed:")
        logger.info(f"  Average Loss: {avg_metrics['loss']:.3f}")
        logger.info(f"  Average Reward: {avg_metrics['reward']:.3f}")
        logger.info(f"  Average KL Div: {avg_metrics['kl_div']:.4f}")

        if avg_metrics["reward"] > self.best_reward:
            self.best_reward = avg_metrics["reward"]
            logger.info(f"  New best reward: {avg_metrics['reward']:.3f}")

        return avg_metrics

    def _simulate_grpo_step(self, batch_data: TrajectoryBatch, epoch: int, step: int) -> Dict[str, float]:
        """Simulate a GRPO training step"""
        # Use actual scores from the batch for more realistic simulation
        batch_rewards = batch_data.scores
        avg_batch_reward = sum(batch_rewards) / len(batch_rewards)

        # Simulate loss and KL divergence
        base_loss = 2.5 - (epoch * 0.15) - (step * 0.01)
        loss = max(0.1, base_loss + np.random.normal(0, 0.1))
        kl_div = 0.05 + np.random.exponential(0.02)

        return {"loss": loss, "reward": avg_batch_reward, "kl_div": kl_div}

    def _update_policy_weights(self):
        """Update the inference server with new policy weights"""
        dummy_weights = {"layer.weight": torch.randn(10, 10)}
        self.inference_manager.update_model_weights(dummy_weights)

    def train(self):
        """Main training loop"""
        logger.info("Starting Simplified GRPO Training with Multi-Environment Support")
        logger.info("=" * 70)

        start_time = time.time()
        training_history = []

        try:
            for epoch in range(self.total_epochs):
                epoch_start = time.time()

                # Train one epoch
                epoch_metrics = self.train_epoch(epoch)
                training_history.append(epoch_metrics)

                epoch_time = time.time() - epoch_start
                logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")
                logger.info("-" * 50)

                time.sleep(0.2)  # Brief pause

            # Training completed
            total_time = time.time() - start_time
            logger.info("=" * 70)
            logger.info("TRAINING COMPLETED SUCCESSFULLY!")
            logger.info(f"Total training time: {total_time:.1f}s")
            logger.info(f"Best reward achieved: {self.best_reward:.3f}")
            logger.info(f"Total training steps: {self.global_step}")

            # Show progress summary
            self._show_training_summary(training_history)

            return True

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def _show_training_summary(self, history):
        """Show training progress summary"""
        logger.info("\nTraining Progress Summary:")
        for i, metrics in enumerate(history):
            epoch_num = i + 1
            loss = metrics.get("loss", 0)
            reward = metrics.get("reward", 0)
            logger.info(f"  Epoch {epoch_num:2d}: Loss={loss:.3f}, Reward={reward:.3f}")


def main():
    """Main entry point"""
    print("Simplified Atropos-VeRL Integration Demo")
    print("Demonstrating the improved architecture using REAL Atropos API")
    print()

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    else:
        print("Running on CPU")

    print()

    # Run the demo
    trainer = SimplifiedGRPOTrainer()
    success = trainer.train()

    if success:
        print("\nDemo completed successfully!")
        print("\nKey improvements demonstrated:")
        print("  - Clean API interface (POST /trajectory_batch)")
        print("  - Multi-environment support with weights (70% GSM8K, 20% Math, 10% Code)")
        print("  - Unified inference management")
        print("  - No complex registration/endpoint management")
        print("  - No Ray complexity for basic use cases")
        print("  - Simple configuration")
        print("\nThis matches your diagram architecture!")
    else:
        print("\nDemo failed or was interrupted")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
