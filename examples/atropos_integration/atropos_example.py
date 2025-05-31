#!/usr/bin/env python3
"""
Atropos Integration Example for VERL
Demonstrates the complete RL loop including policy weight synchronization:
- Rollout with current policy
- Advantage-weighted SFT training  
- Policy weight synchronization to inference engine
- Repeat with updated policy
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional
from contextlib import contextmanager

# Mock classes to demonstrate the complete RL interface
class MockInferenceEngine:
    """Mock inference engine (represents vLLM/SGLang)"""
    
    def __init__(self, model):
        self.model = model
        self.current_weights = None
        
    def update_weights_from_tensor(self, named_tensors, load_format=None):
        """Update inference engine with new weights"""
        print("Updating inference engine weights...")
        self.current_weights = dict(named_tensors)
        print(f"   Updated {len(self.current_weights)} weight tensors")
        
    def generate(self, input_ids, **kwargs):
        """Mock generation using current weights"""
        print("Generating with inference engine...")
        # Simulate generation
        batch_size, seq_len = input_ids.shape
        response_length = 10
        response = torch.randint(0, 1000, (batch_size, response_length))
        return response
        
    def release_memory_occupation(self):
        """Release GPU memory (for memory optimization)"""
        print("Released inference engine memory")
        
    def resume_memory_occupation(self):
        """Resume GPU memory occupation"""
        print("Resumed inference engine memory")


class MockShardingManager:
    """Mock sharding manager to demonstrate weight sync interface"""
    
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


class MockAtroposRLTrainer:
    """
    Complete RL trainer showing the full loop with policy weight synchronization.
    This demonstrates the core interface that Atropos would need to implement.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cpu"
        
        # Initialize inference engine and sharding manager
        self.inference_engine = MockInferenceEngine(model)
        self.sharding_manager = MockShardingManager(model, self.inference_engine)
        
        # RL training state
        self.step = 0
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
    def rollout_phase(self, prompts: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Phase 1: Generate sequences using current policy weights
        This is where the magic happens - inference engine has the latest weights!
        """
        print(f"ROLLOUT PHASE (Step {self.step})")
        
        with self.sharding_manager:  # This syncs weights automatically!
            # Generate responses using inference engine with updated weights
            responses = self.inference_engine.generate(prompts)
            
            # In real implementation, you'd also compute log probabilities
            # and other needed data for RL
            
        return {
            "prompts": prompts,
            "responses": responses,
            "log_probs": torch.randn_like(responses, dtype=torch.float),  # Mock log probs
        }
    
    def compute_advantages(self, rollout_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Phase 2: Compute advantages from Atropos environment feedback
        In real Atropos integration, this would come from environment scoring
        """
        print("COMPUTING ADVANTAGES")
        batch_size, seq_len = rollout_data["responses"].shape
        
        # Mock advantages - in real system, these come from Atropos
        advantages = torch.randn(batch_size, seq_len + rollout_data["prompts"].shape[1])
        print(f"   Computed advantages shape: {advantages.shape}")
        return advantages
    
    def training_phase(
        self, 
        rollout_data: Dict[str, torch.Tensor], 
        advantages: torch.Tensor
    ) -> float:
        """
        Phase 3: Update policy using advantage-weighted loss
        This is the SFT training part shown in the original example
        """
        print("TRAINING PHASE")
        
        # Prepare training data
        input_ids = torch.cat([rollout_data["prompts"], rollout_data["responses"]], dim=1)
        loss_mask = torch.ones_like(input_ids)
        loss_mask[:, :rollout_data["prompts"].shape[1]] = 0  # Mask prompt tokens
        
        # Compute advantage-weighted loss (from original example)
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
        """Core advantage-weighted SFT loss computation (from original example)"""
        batch_size, seq_len = input_ids.shape
        
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
        Complete RL training step demonstrating the full loop:
        1. Rollout with current policy weights
        2. Compute advantages 
        3. Train with advantage-weighted loss
        4. Next rollout will automatically use updated weights!
        """
        print(f"\n{'='*60}")
        print(f"RL TRAINING STEP {self.step}")
        print(f"{'='*60}")
        
        # Phase 1: Rollout (inference engine gets updated weights automatically)
        rollout_data = self.rollout_phase(prompts)
        
        # Phase 2: Compute advantages (from Atropos environment)
        advantages = self.compute_advantages(rollout_data)
        
        # Phase 3: Training (updates the training model weights)
        training_loss = self.training_phase(rollout_data, advantages)
        
        # Phase 4: Next rollout will automatically use updated weights
        # via the sharding manager context!
        
        self.step += 1
        
        return {
            "step": self.step - 1,
            "training_loss": training_loss,
            "num_tokens": rollout_data["responses"].numel(),
        }


def demonstrate_complete_rl_loop():
    """Demonstrate the complete RL training loop with policy weight synchronization"""
    print("COMPLETE ATROPOS-VERL RL INTEGRATION DEMO")
    print("=" * 50)
    print("This demonstrates the full RL loop:")
    print("• Rollout with current policy weights")
    print("• Advantage computation from environment")  
    print("• Advantage-weighted training")
    print("• Automatic policy weight synchronization")
    print("• Repeat with updated policy")
    print()
    
    # Initialize
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create RL trainer
    trainer = MockAtroposRLTrainer(model, tokenizer)
    
    # Sample prompts
    prompts = ["What is the capital of France?", "Explain quantum computing"]
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True, max_length=32)
    prompt_ids = tokenized["input_ids"]
    
    print(f"Sample prompts: {prompts}")
    print(f"Prompt token shape: {prompt_ids.shape}")
    
    # Run multiple RL training steps
    results = []
    for i in range(3):
        result = trainer.rl_training_step(prompt_ids)
        results.append(result)
    
    print(f"\n{'='*60}")
    print("RL TRAINING SUMMARY")
    print(f"{'='*60}")
    for result in results:
        print(f"Step {result['step']}: Loss = {result['training_loss']:.4f}")
    
    print("\nKey Integration Points for Atropos:")
    print("1. Sharding Manager handles automatic weight synchronization")
    print("2. Inference engine always has latest policy weights")
    print("3. Advantage computation integrates with environment feedback")
    print("4. Training updates are immediately available for next rollout")
    print("\nThis ensures proper RL training where each rollout uses")
    print("the most recent policy weights!")


def main():
    demonstrate_complete_rl_loop()


if __name__ == "__main__":
    main() 