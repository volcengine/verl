# FlowRL Implementation

## 4 Simple Steps to Add FlowRL

### Step 1: Add Partition Function Z

**File**: `verl/workers/fsdp_workers.py`

[Add this class at line 100](https://github.com/Xuekai-Zhu/FlowRL/blob/4b0b3bee0e85258b7be46481f9a46ffe9e6b5508/verl_FlowRL/verl/workers/fsdp_workers.py#L100):

```python
class ProjZModule(torch.nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []

        for i in range(num_layers - 1):
            layers.extend([
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.GELU(),
                torch.nn.LayerNorm(hidden_size),
                torch.nn.Dropout(dropout)
            ])
        
        layers.append(torch.nn.Linear(hidden_size, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

[Add this to model building at line 267](https://github.com/Xuekai-Zhu/FlowRL/blob/4b0b3bee0e85258b7be46481f9a46ffe9e6b5508/verl_FlowRL/verl/workers/fsdp_workers.py#L265):

```python
n_dim = actor_module.config.hidden_size  
actor_module.proj_z = ProjZModule(n_dim, num_layers=self.config.actor.porj_layer)
```

### Step 2: Modify Forward Pass

**File**: `verl/workers/actor/dp_actor.py`

[Change method signature at line 75](https://github.com/Xuekai-Zhu/FlowRL/blob/4b0b3bee0e85258b7be46481f9a46ffe9e6b5508/verl_FlowRL/verl/workers/actor/dp_actor.py#L75):

```python
def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False,return_log_z=False) -> Tuple[torch.Tensor, torch.Tensor]:
```

[Add before return at line 232](https://github.com/Xuekai-Zhu/FlowRL/blob/4b0b3bee0e85258b7be46481f9a46ffe9e6b5508/verl_FlowRL/verl/workers/actor/dp_actor.py#L232):

```python
if return_log_z:
    last_hidden = output.hidden_states[-1].squeeze(0) # (total_nnz, hidden size)
    if self.use_ulysses_sp:
            last_hidden = gather_outputs_and_unpad(
                last_hidden,
                gather_dim=0,
                unpad_dim=0,
                padding_size=pad_size, 
            )
    full_last_hidden = pad_input(hidden_states=last_hidden,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen)
    # extract pormpt hiddenstate for log z
    prompts_last_hidden = full_last_hidden[:, : -response_length - 1]
    prompt_attention_mask = attention_mask[:, : -response_length - 1]
    avg_hidden = verl_F.masked_mean(prompts_last_hidden, prompt_attention_mask.unsqueeze(-1), axis=1)

    # avg_hidden = avg_hidden.detach()  # use detach() to stop gradient of proj_z to policy
    log_z = self.actor_module.proj_z(avg_hidden) 

    return entropy, log_probs, log_z
    
else:
    return entropy, log_probs
```

### Step 3: Replace PPO Loss with FlowRL Loss

**File**: `verl/workers/actor/dp_actor.py`

[Replace PPO loss computation around line 412](https://github.com/Xuekai-Zhu/FlowRL/blob/4b0b3bee0e85258b7be46481f9a46ffe9e6b5508/verl_FlowRL/verl/workers/actor/dp_actor.py#L412):

```python
# OLD PPO CODE - REMOVE:
# entropy, log_prob = self._forward_micro_batch(...)
# pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(...)

# NEW FLOWRL CODE:
entropy, log_prob, log_z = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy, return_log_z=True)

policy_loss, data = self.compute_flowrl_objective(logpf=log_prob, 
                                logf_ref=data['ref_log_prob'],
                                logpf_old=old_log_prob,
                                log_z=log_z,
                                reward=advantages,
                                response_mask=response_mask,
                                clip_ratio=self.config.clip_ratio)
```

[Add FlowRL objective function at line 555](https://github.com/Xuekai-Zhu/FlowRL/blob/4b0b3bee0e85258b7be46481f9a46ffe9e6b5508/verl_FlowRL/verl/workers/actor/dp_actor.py#L555):

```python
def compute_flowrl_objective(self, logpf=None, logf_ref=None,  logpf_old=None, log_z=None, reward=None, response_mask=None, clip_ratio=None):
        # we set 𝛽 and 𝛾 to 0.1 and 1.0,
        # squeeze log_z to (B,)
        log_z = log_z.squeeze(-1)
        B = log_z.shape[0]

        # mean of log p_f / log p_ref over valid tokens
        avg_logpf = verl_F.masked_mean(logpf, response_mask, axis=1)
        avg_logp_ref = verl_F.masked_mean(logf_ref, response_mask, axis=1)

        # mean of token-level reward → log
        # we set R = exp(advantage); then log_reward = advantage
        seq_log_reward = verl_F.masked_mean(reward, response_mask, axis=1) 
        
        # TB loss residual
        delta = log_z + avg_logpf - 15 * seq_log_reward - avg_logp_ref

        # important sampling
        log_w = verl_F.masked_sum(logpf - logpf_old, response_mask, axis=1)  # sum over valid tokens per trajectory
        importance_weight = torch.exp(log_w).detach() 
        clip_importance_weight = torch.clamp(importance_weight, 1 - clip_ratio, 1 + clip_ratio)

        weighted_losses = importance_weight * (delta ** 2)
        avg_loss = torch.mean(weighted_losses)
        
        # Loss statistics
        negative_approx_kl = logpf - logf_ref
        ratio = torch.exp(negative_approx_kl)
        loss_term_dict = {
                            "actor/logpf": verl_F.masked_mean(logpf, response_mask).detach().item(),
                            "actor/logp_ref": verl_F.masked_mean(logf_ref, response_mask).detach().item(),
                            "actor/log_z": log_z.mean().detach().item(),
                            "actor/log_reward": verl_F.masked_mean(reward, response_mask).detach().item(),
                            "actor/tb_loss": avg_loss.detach().item(),
                        }
                        
        return avg_loss, loss_term_dict
```

### Step 4: Fix Model Loading

**File**: `verl/workers/sharding_manager/fsdp_vllm.py`

[Change line 290-293](https://github.com/Xuekai-Zhu/FlowRL/blob/4b0b3bee0e85258b7be46481f9a46ffe9e6b5508/verl_FlowRL/verl/workers/sharding_manager/fsdp_vllm.py#L290):

```python
# Skip proj_z parameters when loading to vLLM
loaded_params = model.load_weights(((name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                                            for name, param in updated_params.items()
                                            if not name.startswith("proj_z"))
                                            )
```
