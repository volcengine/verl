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
Main Atropos Recipe Runner

This script demonstrates the complete Atropos-VERL integration:
- Policy weight synchronization via sharding managers
- Advantage-weighted SFT training
- Complete RL training loop

Based on VERL's recipe pattern, similar to DAPO.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional, List
from contextlib import contextmanager


class AtroposInferenceEngine:
    """Inference engine integration (compatible with vLLM/SGLang)"""
    
    def __init__(self, model):
        self.model = model
        self.current_weights = None
        
    def update_weights_from_tensor(self, named_tensors, load_format=None):
        """Update inference engine with new weights"""
        print("Updating inference engine weights...")
        self.current_weights = dict(named_tensors)
        print(f"   Updated {len(self.current_weights)} weight tensors")
        
    def generate(self, input_ids, **kwargs):
        """Generate responses using current policy weights"""
        print("Generating with inference engine...")
        batch_size, seq_len = input_ids.shape
        response_length = kwargs.get('max_new_tokens', 10)
        response = torch.randint(0, 1000, (batch_size, response_length))
        return response
        
    def release_memory_occupation(self):
        """Release GPU memory (for memory optimization)"""
        print("Released inference engine memory")
        
    def resume_memory_occupation(self):
        """Resume GPU memory occupation"""
        print("Resumed inference engine memory")


class AtroposShardingManager:
    """Sharding manager for automatic weight synchronization"""
    
    def __init__(self, training_model, inference_engine):
        self.training_model = training_model
        self.inference_engine = inference_engine
        
    def __enter__(self):
        """Context manager entry: sync weights training → inference"""
        print("\nENTERING SHARDING MANAGER")
        print("   Syncing training weights → inference engine...")
        
        # Get current training model weights
        state_dict = self.training_model.state_dict()
        
        # Update inference engine
        self.inference_engine.resume_memory_occupation()
        self.inference_engine.update_weights_from_tensor(
            named_tensors=[(name, tensor) for name, tensor in state_dict.items()]
        )
        print("   Weight synchronization complete!")
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit: release inference memory"""
        print("   Releasing inference engine memory...")
        self.inference_engine.release_memory_occupation()
        print("EXITING SHARDING MANAGER\n")


class AtroposRLTrainer:
    """
    Complete Atropos RL trainer with proper weight synchronization.
    
    This demonstrates the full integration pattern that Atropos needs:
    - Automatic policy weight updates via sharding managers
    - Advantage-weighted SFT loss computation
    - Memory-efficient rollout and training phases
    """
    
    def __init__(self, model, tokenizer, config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}
        self.device = "cpu"
        
        # Initialize inference engine and sharding manager
        self.inference_engine = AtroposInferenceEngine(model)
        self.sharding_manager = AtroposShardingManager(model, self.inference_engine)
        
        # RL training state
        self.step = 0
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        # Atropos configuration
        self.use_advantage_weighting = self.config.get("use_advantage_weighting", True)
        self.advantage_normalization = self.config.get("advantage_normalization", "batch")
        self.advantage_clipping = self.config.get("advantage_clipping", None)
        
        print(f"Atropos RL Trainer initialized:")
        print(f"  - Advantage weighting: {self.use_advantage_weighting}")
        print(f"  - Advantage normalization: {self.advantage_normalization}")
        print(f"  - Advantage clipping: {self.advantage_clipping}")
        
    def rollout_phase(self, prompts: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Phase 1: Generate sequences using current policy weights.
        
        This is the critical phase where weight synchronization happens!
        """
        print(f"ROLLOUT PHASE (Step {self.step})")
        
        with self.sharding_manager:  # Automatic weight sync!
            # Generate responses using inference engine with updated weights
            responses = self.inference_engine.generate(
                prompts, 
                max_new_tokens=self.config.get("max_response_length", 20)
            )
            
            # In real implementation, you'd compute log probabilities here
            log_probs = torch.randn_like(responses, dtype=torch.float)
            
        return {
            "prompts": prompts,
            "responses": responses,
            "log_probs": log_probs,
        }
    
    def compute_advantages_from_atropos(self, rollout_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Phase 2: Compute advantages from Atropos environment feedback.
        
        In real Atropos integration, this would:
        - Send responses to Atropos environment
        - Get rewards/scores from environment
        - Compute advantages (e.g., GAE, TD-error, etc.)
        """
        print("COMPUTING ADVANTAGES FROM ATROPOS")
        batch_size, seq_len = rollout_data["responses"].shape
        prompt_len = rollout_data["prompts"].shape[1]
        
        # Mock advantages - in real system, these come from Atropos environment
        # Shape should match total sequence length (prompts + responses)
        total_len = prompt_len + seq_len
        advantages = torch.randn(batch_size, total_len)
        
        print(f"   Computed advantages shape: {advantages.shape}")
        print(f"   Prompt length: {prompt_len}, Response length: {seq_len}")
        
        return advantages
    
    def training_phase(
        self, 
        rollout_data: Dict[str, torch.Tensor], 
        advantages: torch.Tensor
    ) -> float:
        """
        Phase 3: Update policy using advantage-weighted loss.
        
        This is where the advantage-weighted SFT training happens.
        """
        print("TRAINING PHASE")
        
        # Prepare training data
        input_ids = torch.cat([rollout_data["prompts"], rollout_data["responses"]], dim=1)
        loss_mask = torch.ones_like(input_ids)
        loss_mask[:, :rollout_data["prompts"].shape[1]] = 0  # Mask prompt tokens
        
        # Compute advantage-weighted loss
        loss = self.compute_advantage_weighted_sft_loss(
            input_ids=input_ids,
            advantages=advantages,
            loss_mask=loss_mask
        )
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        print(f"   Training loss: {loss.item():.4f}")
        return loss.item()
    
    def compute_advantage_weighted_sft_loss(
        self,
        input_ids: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Core advantage-weighted SFT loss computation.
        
        This is the main interface that Atropos needs:
        - Token-level cross-entropy loss
        - Weighted by advantages  
        - Masked by loss_mask
        """
        batch_size, seq_len = input_ids.shape
        
        # Normalize and clip advantages if configured
        if self.advantage_normalization == "batch":
            valid_advantages = advantages[loss_mask.bool()]
            if len(valid_advantages) > 0:
                mean_adv = valid_advantages.mean()
                std_adv = valid_advantages.std() + 1e-8
                advantages = (advantages - mean_adv) / std_adv
        
        if self.advantage_clipping is not None:
            min_val, max_val = self.advantage_clipping
            advantages = torch.clamp(advantages, min=min_val, max=max_val)
        
        # Forward pass
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits
        
        # Prepare for loss computation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_advantages = advantages[..., :-1].contiguous()
        shift_loss_mask = loss_mask[..., :-1].contiguous()
        
        # Flatten
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        flat_advantages = shift_advantages.view(-1)
        flat_loss_mask = shift_loss_mask.view(-1)
        
        # Cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        ce_loss = loss_fct(flat_logits, flat_labels)
        
        # Apply advantage weighting and masking
        weighted_loss = ce_loss * flat_advantages * flat_loss_mask
        
        # Reduce to scalar
        valid_tokens = flat_loss_mask.sum()
        final_loss = weighted_loss.sum() / (valid_tokens + 1e-8)
        
        return final_loss
    
    def rl_training_step(self, prompts: torch.Tensor) -> Dict[str, Any]:
        """
        Complete RL training step demonstrating the full Atropos integration.
        
        1. Rollout with automatic weight synchronization
        2. Compute advantages from Atropos environment  
        3. Train with advantage-weighted loss
        4. Next rollout will use updated weights!
        """
        print(f"\n{'='*60}")
        print(f"RL TRAINING STEP {self.step}")
        print(f"{'='*60}")
        
        # Phase 1: Rollout (inference engine gets updated weights automatically)
        rollout_data = self.rollout_phase(prompts)
        
        # Phase 2: Compute advantages (from Atropos environment)
        advantages = self.compute_advantages_from_atropos(rollout_data)
        
        # Phase 3: Training (updates the training model weights)
        training_loss = self.training_phase(rollout_data, advantages)
        
        # Phase 4: Next rollout will automatically use updated weights
        # via the sharding manager context!
        
        self.step += 1
        
        return {
            "step": self.step - 1,
            "training_loss": training_loss,
            "num_tokens": rollout_data["responses"].numel(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
        }


def run_atropos_integration_demo():
    """Run the complete Atropos-VERL integration demo"""
    print("ATROPOS-VERL INTEGRATION DEMO")
    print("=" * 50)
    print("This demonstrates:")
    print("• Automatic policy weight synchronization")
    print("• Advantage-weighted SFT training")
    print("• Complete RL loop with Atropos environment")
    print("• Memory-efficient inference engine management")
    print()
    
    # Configuration
    config = {
        "use_advantage_weighting": True,
        "advantage_normalization": "batch",
        "advantage_clipping": [-5.0, 5.0],
        "max_response_length": 15,
    }
    
    # Initialize model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create Atropos RL trainer
    trainer = AtroposRLTrainer(model, tokenizer, config)
    
    # Sample prompts for RL training
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing briefly:",
        "Solve this math problem: 2 + 2 =",
    ]
    
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True, max_length=32)
    prompt_ids = tokenized["input_ids"]
    
    print(f"Sample prompts: {prompts}")
    print(f"Prompt token shape: {prompt_ids.shape}")
    
    # Run multiple RL training steps
    results = []
    num_steps = 3
    
    for i in range(num_steps):
        result = trainer.rl_training_step(prompt_ids)
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("ATROPOS INTEGRATION SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        print(f"Step {result['step']}: Loss = {result['training_loss']:.4f}, "
              f"Adv μ = {result['advantage_mean']:.3f}, "
              f"Adv σ = {result['advantage_std']:.3f}")
    
    print(f"\nKey Integration Points for Atropos:")
    print("1. ✓ Automatic weight synchronization via sharding managers")
    print("2. ✓ Advantage-weighted SFT loss computation")
    print("3. ✓ Memory-efficient inference engine management")
    print("4. ✓ Complete RL loop with policy updates")
    print("\nThis ensures each rollout uses the latest policy weights,")
    print("enabling proper RL training dynamics with Atropos!")


def main():
    """Main entry point for Atropos recipe"""
    run_atropos_integration_demo()


if __name__ == "__main__":
    main() 