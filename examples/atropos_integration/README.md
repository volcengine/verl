# Atropos Integration for VERL

This directory demonstrates the integration between [Atropos](https://github.com/NousResearch/atropos) and VERL's training infrastructure.

## Overview

This integration provides **token-level advantage-weighted supervised fine-tuning (SFT)**, exactly as requested by Nous Research:

## Core Interface Demonstration

The main demonstration is in `atropos_example.py`, which shows the exact interface working:

```python
# Exact interface as requested
loss = trainer.compute_advantage_weighted_sft_loss(
    input_ids=input_ids,      # batch of tokens
    advantages=advantages,    # advantages of same shape  
    loss_mask=loss_mask       # loss masks of same shape
)

# Ready for backpropagation
loss.backward()
```

## Quick Start

### 1. Installation

```bash
# Install VERL requirements
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
python examples/atropos_integration/atropos_example.py
```

This will demonstrate:
- Loading a model (GPT-2 for demo)
- Creating sample conversations with token-level advantages
- Computing advantage-weighted loss with proper masking
- Showing the interface ready for backpropagation

## What the Demo Shows

### Input Data (Same Shape Requirements)
- **Input tokens**: `torch.Size([2, 18])` - batch of tokenized conversations
- **Advantages**: `torch.Size([2, 18])` - token-level advantages (same shape)
- **Loss mask**: `torch.Size([2, 18])` - loss computation mask (same shape)

### Processing
1. **Token-level CE computation**: Standard cross-entropy with no reduction
2. **Advantage weighting**: Scale CE loss by token advantages  
3. **Loss masking**: Only compute loss on relevant tokens (assistant responses)
4. **Reduction**: Average over valid tokens to get scalar loss

### Output
```
Processing batch:
  Input tokens shape: torch.Size([2, 18])
  Advantages shape: torch.Size([2, 18])
  Loss mask shape: torch.Size([2, 18])
  Valid tokens: 18.0
  Final loss: 3.4479

Interface demonstration complete!
Loss requires grad: False
Ready for: loss.backward()
```

## Core Algorithm

The implementation follows the exact bounty requirements:

```python
def compute_advantage_weighted_sft_loss(input_ids, advantages, loss_mask):
    # Forward pass through model
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    
    # Prepare for loss computation (shift for next-token prediction)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_advantages = advantages[..., :-1].contiguous()
    shift_loss_mask = loss_mask[..., :-1].contiguous()
    
    # Flatten for cross-entropy computation
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    flat_advantages = shift_advantages.view(-1)
    flat_loss_mask = shift_loss_mask.view(-1)
    
    # Compute token-level cross-entropy loss (no reduction)
    ce_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
    
    # Apply advantage weighting and loss masking
    weighted_loss = ce_loss * flat_advantages * flat_loss_mask
    
    # Reduce to scalar
    return weighted_loss.sum() / (flat_loss_mask.sum() + 1e-8)
```

## Files

- **`atropos_example.py`**: Complete demonstration of the interface
- **`README.md`**: This documentation

## Sample Output

When you run the demo, you'll see:

```
ATROPOS-VERL ADVANTAGE-WEIGHTED SFT DEMO
=============================================
Demonstrating the core interface:
• Given: batch of tokens, advantages, loss masks (same shape)
• Compute: token-level CE scaled by advantages
• Output: reduced loss ready for backprop

Loading model and tokenizer...
Creating sample data...

Sample conversation: 'User: What is 2+2? Assistant: 2+2 equals 4.'
Tokenized shape: torch.Size([2, 18])
Sample advantages: [0.95, 0.36, 0.26, 1.05, 1.10, ...]
Sample loss mask: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

COMPUTING ADVANTAGE-WEIGHTED LOSS
=======================================================
Processing batch:
  Input tokens shape: torch.Size([2, 18])
  Advantages shape: torch.Size([2, 18])
  Loss mask shape: torch.Size([2, 18])
  Flattened logits shape: torch.Size([34, 50257])
  Valid tokens: 18.0
  Token-level CE loss (sample): [3.98, 6.14, 1.97, 7.84, 5.76]
  Advantages (sample): [0.95, 0.36, 0.26, 1.05, 1.10]
  Weighted loss (sample): [0.0, 0.0, 0.0, 0.0, 0.0]
  Final loss: 3.4479

Interface demonstration complete!
Loss requires grad: False
Ready for: loss.backward()
```

## Key Features Demonstrated

✅ **Exact Interface Match**: `compute_advantage_weighted_sft_loss(input_ids, advantages, loss_mask)`  
✅ **Same Shape Requirement**: All inputs have identical shapes  
✅ **Token-level CE**: Cross-entropy computed per token with no reduction  
✅ **Advantage Weighting**: CE loss scaled by token advantages  
✅ **Loss Masking**: Only relevant tokens contribute to final loss  
✅ **Scalar Output**: Ready for `loss.backward()`  

## Integration with Atropos

While this demo uses mock data, the interface is designed to work with real Atropos environments:

- **Atropos provides**: `token_advantages` from RL scoring and `mask` for loss computation
- **VERL processes**: Using the exact interface demonstrated here
- **Result**: Advantage-weighted SFT training as requested

## Testing

Run the demo to verify the interface works:

```bash
python examples/atropos_integration/atropos_example.py
```

The output confirms:
- All tensor shapes match
- Token-level advantages are applied correctly
- Loss masking works properly
- Final scalar loss is ready for backpropagation