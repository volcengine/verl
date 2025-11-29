# Multi-turn Evaluation

Last updated: 11/27/2025.

## Overview

The multi-turn evaluation functionality enables comprehensive evaluation of language models that support tool usage during multi-turn conversations. This feature is particularly useful for assessing models on tasks requiring multiple interactions with external tools, such as mathematical reasoning tasks where models need to use calculators or code interpreters across multiple dialogue turns.

## Key Features

### 1. Tool-Enabled Multi-turn Conversations
- **Interactive Tool Usage**: Models can interact with external tools (calculators, code interpreters, etc.) through multiple conversation turns
- **Custom Tool Configuration**: Support for custom tool definitions via YAML configuration files
- **Conversation State Management**: Proper tracking of conversation context across multiple turns

### 2. Flexible Evaluation Framework
- **Batch Processing**: Efficient batch evaluation of multiple samples
- **Multiple Backends**: Supports both synchronous and asynchronous rollout modes
- **Resource Management**: Intelligent GPU resource allocation and management

### 3. Comprehensive Metrics and Output
- **Tool Performance Tracking**: Detailed metrics on tool usage and effectiveness
- **Turn-by-turn Scoring**: Individual scoring for each conversation turn
- **Agent Metrics**: Performance metrics for the agent loop (when using async mode)
- **Flexible Output Formats**: Support for both Parquet and JSON output formats

## New Files and Components

### 1. `examples/sglang_multiturn/run_qwen3-4b_gsm8k_multiturn_eval.sh`
A comprehensive shell script for running multi-turn evaluation on GSM8K dataset with Qwen3-4B model.

**Key Configuration Features:**
- **Multi-turn Configuration**: Enables multi-turn mode with tool configuration path
- **Data Processing**: Handles GSM8K test dataset with proper batching and filtering

### 2. `verl/trainer/config/multiturn_eval.yaml`
Hydra configuration file that provides a complete parameter structure for multi-turn evaluation.

**Important Configuration Sections:**
- **Multi-turn Settings**: `actor_rollout_ref.rollout.multi_turn.enable: true`
- **Tool Configuration**: Support for `tool_config_path` and `interaction_config_path`
- **Checkpoint Loading**: Optional support for loading trained model checkpoints via `checkpoint_dir`
- **Evaluation Controls**: Configurable `max_batches` and `max_samples` for controlled evaluation

### 3. `verl/trainer/main_multiturn_eval.py`
The main evaluation module that orchestrates the entire multi-turn evaluation pipeline.

**Core Functionality:**
- **Ray Cluster Integration**: Proper initialization and management of Ray clusters
- **Worker Group Management**: Handles both synchronous and asynchronous worker setups
- **Checkpoint Loading**: Robust checkpoint loading with fallback mechanisms
- **Multi-step Evaluation Pipeline**: Complete flow from data loading to results saving

## Advanced Features

### Checkpoint Loading System
The system supports evaluating fine-tuned models from checkpoints:
- **Flexible Path Resolution**: Supports both specific checkpoint paths and automatic latest checkpoint detection
- **Tokenizer Independence**: Checkpoint loading is independent of tokenizer/processor initialization
- **Checkpoint Structure Compatibility**: Follows the same structure as PPO training checkpoints

### Dual Mode Operation
- **Synchronous Mode**: Direct rollout for standard generation without complex tool interactions
- **Asynchronous Mode**: Uses AgentLoopManager for complex multi-turn conversations with tools

### Comprehensive Metrics Collection
- **Tool Rewards**: Scoring for each tool usage interaction
- **Interaction Rewards**: Turn-by-turn interaction quality assessment
- **Generation Performance**: Timing and efficiency metrics
- **Custom Rewards**: Support for user-defined reward functions

## Usage Example

```bash
# Run multi-turn evaluation on GSM8K with Qwen3-4B
bash examples/sglang_multiturn/run_qwen3-4b_gsm8k_multiturn_eval.sh

# With custom configuration overrides
bash examples/sglang_multiturn/run_qwen3-4b_gsm8k_multiturn_eval.sh \
  evaluation.max_samples=100 \
  output.path=./custom_eval_results \
  checkpoint_dir=/path/to/checkpoint/dir
```

## Configuration Details

### Tool Configuration
Tools are defined through YAML configuration files specifying:
- Tool names and descriptions
- Input/output schemas
- Execution parameters
- Reward assignment logic

### Interaction Configuration
Optional configuration for:
- Turn-based interaction patterns
- Conversation flow control
- State management across turns

## Output Artifacts

### 1. Detailed Results (`evaluation_scores.parquet` or `.json`)
Contains per-sample evaluation results including:
- Sample prompts and responses
- Turn scores and interaction rewards
- Tool usage metrics
- Generation timing information
- Agent performance metrics (for async mode)

### 2. Evaluation Trace (`evaluation_trace.json`)
Detailed trace including:
- Complete configuration snapshot
- Sample evaluation records (first 100 samples)
- Debug and troubleshooting information

### 3. Summary Statistics (`evaluation_summary.json`)
Aggregated metrics including:
- Total samples evaluated
- Mean and standard deviation of turn scores
- Average generation and reward computation times
- Overall performance summary

## Integration with Training Pipeline

The multi-turn evaluation system is designed to integrate seamlessly with VERL's training infrastructure:
- **Compatible Configuration Structure**: Uses the same Hydra-based configuration system as PPO training
- **Shared Components**: Reuses data processing, tokenizer loading, and worker management components
- **Checkpoint Compatibility**: Direct evaluation of models trained with VERL's PPO trainer

## Technical Architecture

### AgentLoopManager Integration
When operating in async mode, the system uses AgentLoopManager for:
- Managing conversation state across multiple turns
- Coordinating tool execution and model generation
- Handling complex interaction patterns

### Resource Management
- **Dynamic Resource Allocation**: Intelligent GPU resource assignment based on model size and batch requirements
- **Multi-node Support**: Scalable evaluation across multiple nodes
- **Memory Optimization**: Configurable GPU memory utilization parameters

### Error Handling and Robustness
- **Graceful Degradation**: Fallback mechanisms for reward computation failures
- **Detailed Logging**: Comprehensive logging for debugging and monitoring
- **Validation**: Input validation and error checking throughout the pipeline

## Important Notes

1. **Model Path vs Checkpoint**: `model.path` is always required for tokenizer/processor loading, while `checkpoint_dir` is optional and only needed for loading fine-tuned weights
2. **Tool Configuration**: Proper tool configuration is essential for multi-turn evaluation - ensure tool definitions match your evaluation task requirements
3. **Resource Planning**: Multi-turn evaluation with tools can be resource-intensive - plan GPU allocation accordingly
4. **Batch Size Considerations**: Tool interactions may increase memory usage - adjust batch sizes based on available resources

## Future Enhancements

Potential areas for future development:
- **Additional Tool Integrations**: Support for more specialized tool types
- **Parallel Tool Execution**: Concurrent execution of independent tool calls
- **Advanced Conversation Patterns**: Support for more complex interaction patterns
- **Performance Optimizations**: Further optimizations for large-scale evaluations