# FlowRL Implementation Verification Guide

This guide shows you how to verify each step of the FlowRL implementation using `pdb` breakpoints.

## Prerequisites

Add `import pdb; pdb.set_trace()` at key locations as shown in the [FLOWRL_SIMPLE_GUIDE.md](../../../FlowRL/FLOWRL_SIMPLE_GUIDE.md).

## Breakpoint Locations

### Step 1: ProjZ Module Creation
**File**: `recipe/flowrl/flowrl_fsdp_worker.py:232`
**Location**: After `actor_module.add_module("proj_z", ...)`

### Step 2: FlowRLActor Initialization
**File**: `recipe/flowrl/flowrl_fsdp_worker.py:425`
**Location**: Inside `init_model()`, before replacing actor

### Step 3: Forward Pass with log_z
**File**: `recipe/flowrl/flowrl_actor.py:263`
**Location**: Inside `_forward_micro_batch()`, when `return_log_z=True`

### Step 4: Update Policy Call
**File**: `recipe/flowrl/flowrl_actor.py:366`
**Location**: In `update_policy()`, before forward pass

### Step 5: FlowRL Loss Computation
**File**: `recipe/flowrl/flowrl_actor.py:445`
**Location**: In `compute_flowrl_objective()`

---

## Step 1: Verify ProjZ Module Creation

**Breakpoint**: `recipe/flowrl/flowrl_fsdp_worker.py:232`

### What to Check
The ProjZ module should be successfully added to the actor model BEFORE FSDP wrapping.

### PDB Commands

#### 1. Check if ProjZ exists
```python
(Pdb) hasattr(actor_module, 'proj_z')
# Expected: True
```

#### 2. Verify ProjZ type
```python
(Pdb) type(actor_module.proj_z)
# Expected: <class 'recipe.flowrl.flowrl_actor.ProjZModule'>
```

#### 3. Check dimensions
```python
(Pdb) n_dim
# Expected: 1536 (for Qwen2.5-1.5B)

(Pdb) proj_layers
# Expected: 3 (default)

(Pdb) actor_module.config.hidden_size
# Expected: 1536
```

#### 4. Inspect ProjZ architecture
```python
(Pdb) print(actor_module.proj_z)
# Expected: Shows Sequential network with:
# - (num_layers-1) blocks of [Linear, GELU, LayerNorm, Dropout]
# - Final Linear(hidden_size, 1)

(Pdb) list(actor_module.proj_z.net.children())
# Expected: List of layers [Linear, GELU, LayerNorm, Dropout, ..., Linear]
```

#### 5. Count parameters
```python
(Pdb) sum(p.numel() for p in actor_module.proj_z.parameters())
# Expected: ~7M parameters for 3-layer network with hidden_size=1536

(Pdb) [(name, p.shape) for name, p in actor_module.proj_z.named_parameters()]
# Expected: List of (name, shape) for all weights and biases
```

#### 6. Verify it's in the module tree
```python
(Pdb) 'proj_z' in [name.split('.')[0] for name, _ in actor_module.named_modules()]
# Expected: True

(Pdb) actor_module._modules.keys()
# Expected: dict_keys containing 'proj_z'
```

### Quick One-Liner Verification
```python
(Pdb) print(f"ProjZ: exists={hasattr(actor_module, 'proj_z')}, type={type(actor_module.proj_z).__name__ if hasattr(actor_module, 'proj_z') else 'N/A'}, params={sum(p.numel() for p in actor_module.proj_z.parameters()) if hasattr(actor_module, 'proj_z') else 0:,}")
```

### Continue to Next Step
```python
(Pdb) c
```

---

## Step 2: Verify FlowRLActor Initialization

**Breakpoint**: `recipe/flowrl/flowrl_fsdp_worker.py:425`

### What to Check
After parent's `init_model()` completes, we should replace the standard PPOActor with FlowRLActor.

### PDB Commands

#### 1. Verify we're in actor worker
```python
(Pdb) self._is_actor
# Expected: True
```

#### 2. Check current actor type (before replacement)
```python
(Pdb) type(self.actor)
# Expected: <class 'verl.workers.actor.dp_actor.DataParallelPPOActor'>
```

#### 3. Check that FSDP model has proj_z
```python
(Pdb) hasattr(self.actor_module_fsdp, 'proj_z')
# Expected: True (might be wrapped, check below)

(Pdb) hasattr(self.actor_module_fsdp._fsdp_wrapped_module, 'proj_z')
# Expected: True
```

#### 4. Step through actor replacement
```python
(Pdb) n  # Execute lines to create FlowRLActor
# ... step through the FlowRLActor creation
```

#### 5. Verify FlowRLActor was created
```python
(Pdb) type(self.actor)
# Expected: <class 'recipe.flowrl.flowrl_actor.FlowRLActor'>

(Pdb) isinstance(self.actor, FlowRLActor)
# Expected: True
```

#### 6. Check FlowRL beta coefficient
```python
(Pdb) self.actor.flowrl_beta_coef
# Expected: 15.0 (default)
```

#### 7. Verify actor has access to proj_z
```python
(Pdb) hasattr(self.actor.actor_module, 'proj_z')
# Expected: True
```

### Quick One-Liner Verification
```python
(Pdb) print(f"Actor: type={type(self.actor).__name__}, is_FlowRL={isinstance(self.actor, FlowRLActor)}, beta={getattr(self.actor, 'flowrl_beta_coef', 'N/A')}, has_proj_z={hasattr(self.actor.actor_module, 'proj_z') or hasattr(getattr(self.actor.actor_module, '_fsdp_wrapped_module', None), 'proj_z')}")
```

### Continue to Next Step
```python
(Pdb) c
```

---

## Step 3: Verify log_z Computation in Forward Pass

**Breakpoint**: `recipe/flowrl/flowrl_actor.py:263`

### What to Check
During forward pass, when `return_log_z=True`, we should extract hidden states and compute log Z.

### PDB Commands

#### 1. Verify return_log_z flag
```python
(Pdb) return_log_z
# Expected: True
```

#### 2. Check model output has hidden states
```python
(Pdb) hasattr(output, 'hidden_states')
# Expected: True

(Pdb) len(output.hidden_states)
# Expected: num_layers + 1 (e.g., 29 for 28-layer model)

(Pdb) output.hidden_states[-1].shape
# Expected: (1, total_nnz, hidden_size) or (total_nnz, hidden_size)
```

#### 3. Inspect hidden states extraction
```python
(Pdb) last_hidden = output.hidden_states[-1].squeeze(0)
(Pdb) last_hidden.shape
# Expected: (total_nnz, 1536) for remove_padding=True
# Expected: (batch_size, seqlen, 1536) for remove_padding=False
```

#### 4. Check response_length
```python
(Pdb) response_length
# Expected: Positive integer (e.g., 512, 1024, etc.)
```

#### 5. Verify prompt hidden states extraction
```python
(Pdb) n  # Step through to prompts_last_hidden computation
(Pdb) prompts_last_hidden.shape
# Expected: (batch_size, prompt_length, hidden_size)

(Pdb) prompt_attention_mask.shape
# Expected: (batch_size, prompt_length)
```

#### 6. Check averaged hidden state
```python
(Pdb) avg_hidden.shape
# Expected: (batch_size, hidden_size)
```

#### 7. Verify proj_z computation
```python
(Pdb) hasattr(self.actor_module, 'proj_z')
# Expected: True

(Pdb) log_z = self.actor_module.proj_z(avg_hidden)
(Pdb) log_z.shape
# Expected: (batch_size, 1)

(Pdb) log_z
# Expected: Tensor with reasonable values (not NaN, not too large)
```

#### 8. Check return values
```python
(Pdb) entropy.shape if entropy is not None else None
# Expected: (batch_size, response_length) or None

(Pdb) log_probs.shape
# Expected: (batch_size, response_length)

(Pdb) log_z.shape
# Expected: (batch_size, 1)
```

### Quick One-Liner Verification
```python
(Pdb) print(f"Forward: return_log_z={return_log_z}, hidden_layers={len(output.hidden_states) if hasattr(output, 'hidden_states') else 'N/A'}, log_z_shape={log_z.shape if 'log_z' in locals() else 'not computed'}, log_z_mean={log_z.mean().item() if 'log_z' in locals() else 'N/A':.4f}")
```

### Continue to Next Step
```python
(Pdb) c
```

---

## Step 4: Verify Update Policy Forward Call

**Breakpoint**: `recipe/flowrl/flowrl_actor.py:366`

### What to Check
Verify that forward pass is called with `return_log_z=True` during policy update.

### PDB Commands

#### 1. Check input data
```python
(Pdb) model_inputs.keys()
# Expected: dict_keys(['responses', 'response_mask', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages', 'ref_log_prob'])

(Pdb) response_mask.shape
# Expected: (micro_batch_size, response_length)

(Pdb) advantages.shape
# Expected: (micro_batch_size, response_length)

(Pdb) ref_log_prob.shape
# Expected: (micro_batch_size, response_length)
```

#### 2. Check temperature
```python
(Pdb) temperature
# Expected: 1.0 (default)
```

#### 3. Step through forward pass
```python
(Pdb) n  # Execute the forward pass
```

#### 4. Verify outputs
```python
(Pdb) entropy
# Expected: None (since calculate_entropy=False)

(Pdb) log_prob.shape
# Expected: (micro_batch_size, response_length)

(Pdb) log_z.shape
# Expected: (micro_batch_size, 1)
```

#### 5. Check for NaN or Inf
```python
(Pdb) torch.isnan(log_prob).any()
# Expected: False

(Pdb) torch.isnan(log_z).any()
# Expected: False

(Pdb) torch.isinf(log_z).any()
# Expected: False
```

#### 6. Check value ranges
```python
(Pdb) log_prob.min().item(), log_prob.max().item()
# Expected: Negative values (log probabilities)

(Pdb) log_z.min().item(), log_z.max().item()
# Expected: Reasonable range (not extreme values)
```

### Quick One-Liner Verification
```python
(Pdb) print(f"Update: batch_size={log_prob.shape[0]}, log_prob_range=[{log_prob.min().item():.4f}, {log_prob.max().item():.4f}], log_z_range=[{log_z.min().item():.4f}, {log_z.max().item():.4f}], has_nan={torch.isnan(log_z).any()}")
```

### Continue to Next Step
```python
(Pdb) c
```

---

## Step 5: Verify FlowRL Loss Computation

**Breakpoint**: `recipe/flowrl/flowrl_actor.py:445`

### What to Check
Verify the trajectory balance loss computation with all required components.

### PDB Commands

#### 1. Check all inputs exist
```python
(Pdb) logpf is not None and logf_ref is not None and log_z is not None and reward is not None
# Expected: True

(Pdb) logpf.shape, logf_ref.shape, logpf_old.shape, log_z.shape, reward.shape, response_mask.shape
# Expected: All (batch_size, response_length) except log_z which is (batch_size, 1)
```

#### 2. Check clip_ratio
```python
(Pdb) clip_ratio
# Expected: 0.2 or similar (from config)
```

#### 3. Step through loss computation
```python
(Pdb) n  # Squeeze log_z
(Pdb) log_z.shape
# Expected: (batch_size,)

(Pdb) n  # Compute avg_logpf
(Pdb) avg_logpf.shape
# Expected: (batch_size,)

(Pdb) n  # Compute avg_logp_ref
(Pdb) avg_logp_ref.shape
# Expected: (batch_size,)

(Pdb) n  # Compute seq_log_reward
(Pdb) seq_log_reward.shape
# Expected: (batch_size,)
```

#### 4. Check delta computation (TB residual)
```python
(Pdb) delta.shape
# Expected: (batch_size,)

(Pdb) delta
# Expected: Tensor values (can be positive or negative)

(Pdb) self.flowrl_beta_coef
# Expected: 15.0 (default)
```

#### 5. Verify importance weights
```python
(Pdb) log_w.shape
# Expected: (batch_size,)

(Pdb) importance_weight.shape
# Expected: (batch_size,)

(Pdb) importance_weight.min().item(), importance_weight.max().item()
# Expected: Should be around 1.0 initially, may vary in later epochs

(Pdb) clip_importance_weight.shape
# Expected: (batch_size,)
```

#### 6. Check final loss
```python
(Pdb) weighted_losses.shape
# Expected: (batch_size,)

(Pdb) avg_loss.shape
# Expected: torch.Size([]) (scalar)

(Pdb) avg_loss.item()
# Expected: Positive value (MSE loss)
```

#### 7. Check loss metrics
```python
(Pdb) loss_term_dict
# Expected: Dictionary with keys:
# 'actor/logpf', 'actor/logp_ref', 'actor/log_z',
# 'actor/log_reward', 'actor/tb_loss', 'actor/importance_weight'

(Pdb) loss_term_dict['actor/tb_loss']
# Expected: Same as avg_loss.item()

(Pdb) loss_term_dict['actor/log_z']
# Expected: Mean log_z value
```

#### 8. Verify no NaN/Inf
```python
(Pdb) torch.isnan(avg_loss).any()
# Expected: False

(Pdb) torch.isinf(avg_loss).any()
# Expected: False
```

### Quick One-Liner Verification
```python
(Pdb) print(f"Loss: tb_loss={avg_loss.item():.6f}, log_z_mean={log_z.mean().item():.4f}, delta_mean={delta.mean().item():.4f}, delta_std={delta.std().item():.4f}, imp_weight_range=[{importance_weight.min().item():.4f}, {importance_weight.max().item():.4f}]")
```

### Full Loss Formula Check
```python
(Pdb) # Verify TB loss formula: L = E[(log_z + log_pf - β*log_R - log_p_ref)²]
(Pdb) manual_delta = log_z + avg_logpf - self.flowrl_beta_coef * seq_log_reward - avg_logp_ref
(Pdb) torch.allclose(delta, manual_delta)
# Expected: True

(Pdb) manual_loss = torch.mean(clip_importance_weight * (delta ** 2))
(Pdb) torch.allclose(avg_loss, manual_loss)
# Expected: True
```

### Continue Execution
```python
(Pdb) c
```

---

## Common Issues and Troubleshooting

### Issue 1: `proj_z` not found
**Symptom**: `AttributeError: 'XXX' object has no attribute 'proj_z'`

**Check**:
```python
(Pdb) self.actor_module
# If FSDP wrapped, try:
(Pdb) self.actor_module._fsdp_wrapped_module.proj_z
```

### Issue 2: Hidden states not in output
**Symptom**: `AttributeError: 'XXX' object has no attribute 'hidden_states'`

**Check**:
```python
(Pdb) actor_module.config.output_hidden_states
# Expected: True

# If False, you need to enable it in model config
```

### Issue 3: NaN in loss
**Symptom**: Loss becomes NaN during training

**Check**:
```python
(Pdb) torch.isnan(log_z).any()
(Pdb) torch.isnan(log_prob).any()
(Pdb) torch.isnan(reward).any()
(Pdb) torch.isinf(delta).any()

# Check gradient flow
(Pdb) [p.grad is not None and torch.isnan(p.grad).any() for p in self.actor_module.proj_z.parameters()]
```

### Issue 4: Shapes mismatch
**Symptom**: RuntimeError about tensor shapes

**Check all shapes**:
```python
(Pdb) print(f"""
Shapes:
  logpf: {logpf.shape}
  logf_ref: {logf_ref.shape}
  log_z: {log_z.shape}
  reward: {reward.shape}
  response_mask: {response_mask.shape}
""")
```

---

## Complete Verification Checklist

- [ ] **Step 1**: ProjZ module exists and has correct architecture
- [ ] **Step 2**: FlowRLActor successfully replaces DataParallelPPOActor
- [ ] **Step 3**: Forward pass returns `log_z` with correct shape
- [ ] **Step 4**: Update policy receives and uses `log_z`
- [ ] **Step 5**: FlowRL trajectory balance loss is computed correctly
- [ ] **Step 6**: Loss has no NaN/Inf and is backpropagating
- [ ] **Step 7**: Training metrics are being logged properly

---

## PDB Quick Reference

| Command | Description |
|---------|-------------|
| `n` | Execute next line (step over) |
| `s` | Step into function |
| `c` | Continue execution until next breakpoint |
| `l` | List source code around current line |
| `p <expr>` | Print expression |
| `pp <expr>` | Pretty-print expression |
| `w` | Show stack trace |
| `u` | Move up stack frame |
| `d` | Move down stack frame |
| `b <line>` | Set breakpoint at line |
| `cl <n>` | Clear breakpoint number n |
| `q` | Quit debugger |

---

## Tips

1. **Use `pp` for complex objects**: `pp dict(actor_module.named_modules())`
2. **Check shapes frequently**: Tensor shape mismatches are common
3. **Verify no NaN early**: NaN propagates quickly and is hard to debug later
4. **Compare with reference**: Check that `logpf` and `logf_ref` are similar initially
5. **Monitor log_z range**: Should stabilize during training
6. **Watch delta values**: Large delta → high loss → need more training
7. **Save breakpoint state**: Use `!import pickle; pickle.dump(locals(), open('debug.pkl', 'wb'))`

---

## Next Steps After Verification

Once all steps are verified:
1. Remove the `pdb.set_trace()` breakpoints
2. Run full training and monitor:
   - `actor/tb_loss` should decrease
   - `actor/log_z` should stabilize
   - `actor/importance_weight` should stay near 1.0
3. Compare with baseline PPO performance
4. Tune hyperparameters (`flowrl_beta_coef`, learning rate, etc.)

Good luck debugging! 🚀
