# Atropos-VERL Integration Recipe

> **Technical Implementation**: Advantage-weighted SFT with Automatic Policy Weight Synchronization

[Atropos Repository](https://github.com/NousResearch/atropos)

This recipe provides a working integration for using Atropos RL environments with VERL, implementing automatic policy weight synchronization during RL training.

## Implementation Overview

This integration implements the following components:

1. **Rollout**: Generate sequences using current policy weights
2. **Training**: Update policy weights using advantages from environment  
3. **Weight Sync**: Update inference engine with new weights automatically
4. **Repeat**: Next rollout uses updated policy weights

The weight synchronization is handled automatically through VERL's Sharding Manager system.

## Usage

### Run the Integration Demo

```bash
cd verl  # Repository root
python recipe/atropos/main_atropos.py
```

Output: Complete RL training with policy updates, automatic weight synchronization, and Atropos API integration.

### Run End-to-End Test

```bash
cd verl
python recipe/atropos/test_full_e2e_training.py
```

Output: 5 training steps with API integration testing and comprehensive error handling.

### Advanced Orchestration System

```bash
cd verl
python recipe/atropos/launch_atropos_verl.py
```

Note: Requires full environment setup with Atropos server running.

## Technical Implementation

### Automatic Policy Weight Synchronization

VERL's Sharding Manager pattern handles weight synchronization:

```python
# Automatic weight synchronization via context manager
with self.sharding_manager:  
    # Inference engine receives latest policy weights
    responses = self.inference_engine.generate(prompts)
    # Training proceeds with current policy
```

### Atropos API Integration

Integration with Atropos API endpoints:

```python
# Test API connectivity and raise error if unreachable
self._test_api_connectivity(atropos_url)

# Register trainer with Atropos
response = requests.post(f"{atropos_url}/register", json=registration_data)
trainer_uuid = response.json()['uuid']

# Submit scored data  
requests.post(f"{atropos_url}/scored_data", json=scored_data)

# Retrieve processed batches with retry logic
batch = requests.get(f"{atropos_url}/batch").json()['batch']
```

### Error Handling for API Connectivity

```python
class AtroposAPIError(Exception):
    """Raised when Atropos API is unreachable or returns an error"""
    pass

def _test_api_connectivity(self, atropos_url: str, timeout: int = 10) -> None:
    """Test API connectivity and raise error if unreachable"""
    try:
        response = requests.get(f"{atropos_url}/status", timeout=timeout)
        if response.status_code != 200:
            raise AtroposAPIError(f"Atropos API returned status {response.status_code}")
    except requests.exceptions.ConnectTimeout:
        raise AtroposAPIError(f"Connection timeout to Atropos API at {atropos_url}")
    except requests.exceptions.ConnectionError:
        raise AtroposAPIError(f"Cannot connect to Atropos API at {atropos_url} - ensure server is running")
```

### Advantage-weighted SFT Loss

Interface for advantage-weighted training:

```python
loss = trainer.compute_advantage_weighted_sft_loss(
    input_ids=input_ids,      # Batch of tokens  
    advantages=advantages,    # Token-level advantages from Atropos
    loss_mask=loss_mask,     # Mask prompt vs response tokens
)
```

### Complete RL Training Loop

Full integration implementation:

```python
def rl_training_step(self, prompts: torch.Tensor) -> Dict[str, Any]:
    """
    Complete RL training step with Atropos API integration.
    
    1. Rollout with automatic weight synchronization
    2. Compute advantages using Atropos API (/register, /batch endpoints)
    3. Train with advantage-weighted loss
    4. Next rollout uses updated weights automatically
    """
    print(f"RL TRAINING STEP {self.step}")
    
    # Phase 1: Rollout (inference engine gets updated weights automatically)
    rollout_data = self.rollout_phase(prompts)
    
    # Phase 2: Compute advantages (using Atropos API)
    advantages = self.compute_advantages_from_atropos(rollout_data)
    
    # Phase 3: Training (updates the training model weights)
    training_loss = self.training_phase(rollout_data, advantages)
    
    # Phase 4: Next rollout will automatically use updated weights
    # via the sharding manager context!
    
    return {"loss": training_loss, "advantages": advantages}
```

## Architecture

![Atropos-VERL Integration Architecture](diagram.png)

**Component Interactions:**

1. **RL Trainer** - Manages training loop, updates inference weights, queries Atropos API
2. **Atropos Trajectory API** - Central coordination point handling tokens, masks, scores, and groups
3. **Inference Engine** - Handles rollout generation using current policy weights (vLLM/SGLang)
4. **Environment Servers** - Multiple specialized environments providing task-specific evaluation

## Configuration

The integration uses Python configuration dictionaries:

### Basic Configuration

```python
config = {
    'atropos': {
        'api_url': 'http://localhost:9001',  # Atropos API URL
        'timeout': 30
    },
    'use_advantage_weighting': True,
    'advantage_normalization': 'batch',     # "none", "batch", "global"
    'advantage_clipping': [-3.0, 3.0],     # Prevent extreme values
    'max_response_length': 32,
    'batch_retry_attempts': 8,              # Retry logic
    'batch_retry_delay': 0.3,
    'batch_max_wait_time': 12.0
}
```

### Remote Atropos Configuration

```python
config = {
    'atropos': {
        'api_url': 'https://atropos.example.com/api',  # Remote server
        'timeout': 60                                  # Longer timeout
    },
    # ... other settings
}
```

### Production Configuration

```python
import os

config = {
    'atropos': {
        'api_url': os.getenv('ATROPOS_API_URL', 'http://localhost:9001'),
        'timeout': int(os.getenv('ATROPOS_TIMEOUT', '30'))
    },
    'batch_retry_attempts': 10,             # More retries for production
    'batch_max_wait_time': 30.0,           # Longer wait time
    'advantage_clipping': [-5.0, 5.0],     # Conservative clipping
}
```

## Implementation Details

### Automatic Weight Synchronization

```python
class AtroposShardingManager:
    def __enter__(self):
        # Sync latest training weights → inference engine
        state_dict = self.training_model.state_dict()
        self.inference_engine.update_weights_from_tensor(state_dict)
        self.inference_engine.resume_memory_occupation()
        return self
        
    def __exit__(self, *args):
        # Release inference memory for training phase
        self.inference_engine.release_memory_occupation()
```

### Atropos API Integration

```python
def _register_with_atropos_api(self, atropos_url):
    registration_data = {
        "wandb_group": "verl_atropos_integration",
        "batch_size": 4,
        "max_token_len": 512,
        "checkpoint_dir": "/tmp/verl_checkpoints"
    }
    response = requests.post(f"{atropos_url}/register", json=registration_data)
    self.trainer_uuid = response.json()['uuid']
    return True

def _retrieve_batch_with_retry(self, atropos_url):
    # Retry logic with exponential backoff
    for attempt in range(self.batch_retry_attempts):
        response = requests.get(f"{atropos_url}/batch")
        batch = response.json().get('batch')
        if batch and len(batch) > 0:
            return batch
        time.sleep(self.batch_retry_delay * (1.5 ** attempt))
    return None
```

### Advantage-weighted Loss Implementation

```python
def compute_advantage_weighted_sft_loss(self, input_ids, advantages, loss_mask):
    # Normalize advantages for stable training
    if self.advantage_normalization == "batch":
        valid_advantages = advantages[loss_mask.bool()]
        mean_adv = valid_advantages.mean()
        std_adv = valid_advantages.std() + 1e-8
        advantages = (advantages - mean_adv) / std_adv
    
    # Clip advantages to prevent extreme values
    if self.advantage_clipping:
        min_val, max_val = self.advantage_clipping
        advantages = torch.clamp(advantages, min=min_val, max=max_val)
    
    # Compute cross-entropy loss
    outputs = self.model(input_ids=input_ids)
    logits = outputs.logits
    ce_loss = CrossEntropyLoss(reduction='none')(logits.view(-1, logits.size(-1)), 
                                               input_ids.view(-1))
    
    # Apply advantage weighting and masking
    weighted_loss = ce_loss * advantages.view(-1) * loss_mask.view(-1)
    
    # Reduce to scalar
    return weighted_loss.sum() / (loss_mask.sum() + 1e-8)
```

## Testing

### Complete Test Suite

```bash
# End-to-end test
python recipe/atropos/test_full_e2e_training.py
```

Test Coverage:
- API connectivity and registration (with proper error handling)
- Data submission and batch retrieval
- Advantage computation and normalization  
- Weight synchronization via sharding managers
- Complete RL training loop (5 steps)
- Error handling and retry logic
- Memory management
- Performance metrics

### Individual Components

```bash
# Core integration demo
python recipe/atropos/main_atropos.py

# Advanced orchestration (requires full setup)
python recipe/atropos/launch_atropos_verl.py
```

## Error Scenarios

### API Unreachable

When the Atropos API is not available, the integration will:

1. Test connectivity on initialization
2. Raise `AtroposAPIError` with clear error message
3. Stop training with informative guidance
4. Suggest troubleshooting steps

Example error output:
```
ATROPOS API ERROR: Cannot connect to Atropos API at http://localhost:9001 - ensure server is running

Ensure that:
1. Atropos server is running on http://localhost:9001
2. The API endpoints are accessible
3. Network connectivity is available
```

### Network Issues

The integration handles various network scenarios:
- Connection timeouts
- Connection refused (server not running)
- HTTP errors (4xx, 5xx responses)
- Network unreachable

## Deployment Configurations

### Local Development

```python
config = {
    'atropos': {'api_url': 'http://localhost:9001'}
}
```

### Remote/Production Atropos

```python
config = {
    'atropos': {
        'api_url': 'https://your-atropos-server.com/api',
        'timeout': 60
    }
}
```

## Technical Details

**Weight Synchronization**: VERL's sharding managers use context managers to automatically sync training model weights to inference engines before each rollout.

**Model Compatibility**: Tested with models from 0.5B to larger sizes. Memory management scales automatically.

**API Integration**: Works with any Atropos API deployment. Update the `api_url` in configuration to point to your Atropos instance.

**Inference Engines**: Replace the mock inference engine with VERL's vLLM or SGLang sharding managers for production deployment.

**Error Handling**: Includes comprehensive fallback mechanisms, retry logic with exponential backoff, and proper error propagation when the API is unavailable.

## Running the Integration

```bash
python recipe/atropos/main_atropos.py
```

This demonstrates the complete working integration with proper error handling. 