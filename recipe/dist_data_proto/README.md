# [WIP] DistDataProto: An Easy-to-Use Transparent Distributed Data Layer

A work-in-progress implementation for [volcengine/verl#2847](https://github.com/volcengine/verl/issues/2847) and long sequence training related challenges.

## Overview
We propose **DistDataProto** - a user-friendly, transparent distributed data storage format. This implementation:
- Maintains API compatibility with DataProto
- Replaces storage backend with distributed node-wise storage
- Transmits only metadata and data references during dispatch/collect phases
- Materializes data via P2P communication only when needed
- Implements prefetching to overlap computation and communication delays during mini-batch iterations

## Key Features
✅ **Transparent Distribution**  
Seamless transition from single-node to distributed environment with identical API surface

🔗 **Lazy Materialization**  
Physical data transmission occurs on-demand using P2P communication patterns

⏱️ **Communication-Computation Overlap**  
Prefetch mechanism hides communication latency behind computation tasks

## Communication Optimizations
Completed enhancements:
1. Transitioned from one-to-all scatter/all-to-one gather to all-to-all communication patterns  
   → **Reduced overall communication volume**
2. Prefetch-enabled operations:  
   - `compute_log_prob`
   - `update_actor`  
   → **Effective latency masking for sequential operations**

## Current Limitations
### Generate Sequences
- Current implementation transmits all data to generation engine for internal scheduling, because of potential bubbles in generation engine if we implement mini-batch prefetching here
- **Discussion Needed**: Optimal batching strategy for generation pipeline

### Advantage Calculation
- UID grouping requirement for algorithms like GRPO
- Current architecture requires modifications to enable prefetching at this stage
