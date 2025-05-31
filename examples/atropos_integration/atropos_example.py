#!/usr/bin/env python3
"""
Atropos Integration Example for VERL
Demonstrates loss-masked weighted SFT interface:
- Given a batch of tokens, advantages, and loss masks (all same shape)
- Compute token-level CE and scale by advantages
- Reduce and backprop
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Mock trainer class to demonstrate the interface
class MockAtroposSFTTrainer:
    """Mock trainer to demonstrate the advantage-weighted SFT interface."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cpu"  # Use CPU for demo
        
    def compute_advantage_weighted_sft_loss(
        self,
        input_ids: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Core interface: loss-masked weighted SFT
        
        Args:
            input_ids: Batch of tokens, shape (batch_size, seq_len)
            advantages: Batch of advantages, same shape as input_ids
            loss_mask: Batch of loss masks, same shape as input_ids
            
        Returns:
            Scalar loss tensor ready for backprop
        """
        print(f"Processing batch:")
        print(f"  Input tokens shape: {input_ids.shape}")
        print(f"  Advantages shape: {advantages.shape}")
        print(f"  Loss mask shape: {loss_mask.shape}")
        
        # Forward pass through model (WITH gradients for realistic demo)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits
        
        # Prepare for loss computation
        shift_logits = logits[..., :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
        shift_labels = input_ids[..., 1:].contiguous()   # (batch_size, seq_len-1)
        shift_advantages = advantages[..., :-1].contiguous()  # (batch_size, seq_len-1)
        shift_loss_mask = loss_mask[..., :-1].contiguous()    # (batch_size, seq_len-1)
        
        # Flatten for cross-entropy computation
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))  # (batch_size*(seq_len-1), vocab_size)
        flat_labels = shift_labels.view(-1)                    # (batch_size*(seq_len-1),)
        flat_advantages = shift_advantages.view(-1)            # (batch_size*(seq_len-1),)
        flat_loss_mask = shift_loss_mask.view(-1)              # (batch_size*(seq_len-1),)
        
        print(f"  Flattened logits shape: {flat_logits.shape}")
        print(f"  Valid tokens: {flat_loss_mask.sum().item()}")
        
        # Compute token-level cross-entropy loss (no reduction)
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        ce_loss = loss_fct(flat_logits, flat_labels)  # (batch_size*(seq_len-1),)
        
        # Apply advantage weighting and loss masking
        weighted_loss = ce_loss * flat_advantages * flat_loss_mask
        
        # Reduce to scalar
        valid_tokens = flat_loss_mask.sum()
        final_loss = weighted_loss.sum() / (valid_tokens + 1e-8)
        
        print(f"  Token-level CE loss (sample): {ce_loss[:5].tolist()}")
        print(f"  Advantages (sample): {flat_advantages[:5].tolist()}")
        print(f"  Weighted loss (sample): {weighted_loss[:5].tolist()}")
        print(f"  Final loss: {final_loss.item():.4f}")
        
        return final_loss


def demonstrate_interface():
    """Demonstrate the core advantage-weighted SFT interface."""
    print("LOSS-MASKED WEIGHTED SFT INTERFACE DEMONSTRATION")
    print("=" * 55)
    
    # Load a small model for demonstration
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create mock trainer
    trainer = MockAtroposSFTTrainer(model, tokenizer)
    
    print("Creating sample data...")
    
    # Sample conversations (what would come from Atropos)
    conversations = [
        "User: What is 2+2? Assistant: 2+2 equals 4.",
        "User: Write a poem. Assistant: Roses are red, violets are blue.",
    ]
    
    # Tokenize conversations
    tokenized = tokenizer(
        conversations,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    )
    
    input_ids = tokenized["input_ids"]
    batch_size, seq_len = input_ids.shape
    
    # Create token-level advantages (what would come from Atropos RL scoring)
    # Add randomness to make each run different
    advantages = torch.randn(batch_size, seq_len) * 0.5 + 1.0 + torch.rand(1).item() * 0.2
    
    # Create loss masks (1 = compute loss, 0 = ignore)
    # Typically mask out user prompts, compute loss on assistant responses
    loss_mask = torch.ones(batch_size, seq_len)
    # For demo, mask out first few tokens (user prompts)
    loss_mask[:, :8] = 0  # Mask first 8 tokens
    
    print(f"\nSample conversation: '{conversations[0]}'")
    print(f"Tokenized shape: {input_ids.shape}")
    print(f"Sample advantages: {advantages[0, :10].tolist()}")
    print(f"Sample loss mask: {loss_mask[0, :10].tolist()}")
    
    print("\n" + "="*55)
    print("COMPUTING ADVANTAGE-WEIGHTED LOSS")
    print("="*55)
    
    # Demonstrate the core interface
    loss = trainer.compute_advantage_weighted_sft_loss(
        input_ids=input_ids,
        advantages=advantages,
        loss_mask=loss_mask
    )
    
    print(f"\nInterface demonstration complete!")
    print(f"Loss requires grad: {loss.requires_grad}")
    print(f"Ready for: loss.backward()")
    
    return loss


def demonstrate_backprop():
    """Demonstrate the full pipeline with backpropagation."""
    print("\n" + "="*55)
    print("FULL PIPELINE WITH BACKPROPAGATION")
    print("="*55)
    
    # Get the loss from interface demonstration
    loss = demonstrate_interface()
    
    print(f"\nPerforming backpropagation...")
    print(f"Loss value: {loss.item():.4f}")
    print(f"Loss requires grad: {loss.requires_grad}")
    
    # Actually perform backpropagation to show it works
    if loss.requires_grad:
        print("Computing gradients...")
        loss.backward()
        print("Backpropagation complete - gradients computed!")
    else:
        print("Note: Model in eval mode, gradients not computed")
    
    print("Interface working correctly!")


def main():
    """Main demonstration function."""
    print("ATROPOS-VERL ADVANTAGE-WEIGHTED SFT DEMO")
    print("="*45)
    print("Demonstrating the core interface:")
    print("• Given: batch of tokens, advantages, loss masks (same shape)")
    print("• Compute: token-level CE scaled by advantages")
    print("• Output: reduced loss ready for backprop")
    print()
    
    demonstrate_backprop()


if __name__ == "__main__":
    main() 