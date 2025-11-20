# Copyright 2025 Amazon.com Inc and/or its affiliates
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


import json
import os
import time
import uuid

import boto3
import numpy as np
import pytest
import ray
from dotenv import dotenv_values
from omegaconf import DictConfig

from verl.experimental.agent_loop.agentcore_loop import AgentCoreLoopManager, RolloutBuffer
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.rollout import RolloutConfig

RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        "VLLM_LOGGING_LEVEL": "INFO",
        "VLLM_USE_V1": "1",
    }
}


@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True",
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes",
            ],
        )

    model_path = os.path.expanduser("~/models/Qwen/Qwen3-4B-Instruct-2507")
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = os.environ.get("ROLLOUT_NAME", "vllm")
    config.actor_rollout_ref.rollout.mode = "agentcore"
    config.actor_rollout_ref.rollout.enforce_eager = True
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.n = 4
    config.actor_rollout_ref.rollout.agent.num_workers = 2
    config.actor_rollout_ref.rollout.skip_tokenizer_init = True

    # Separate agentcore specific env vars to a separate file for better
    # modularity & easier organization.
    # Relevant configs can be passed in via command line args too. Using an env file here
    # to avoid hardcoded values.
    agentcore_envs = dotenv_values("agentcore.env")

    # Try setting agentcore related configs below to ensure they have been set up properly.
    # The following won't work without relevant fields defined in
    # `verl/trainer/config/rollout/rollout.yaml`
    config.actor_rollout_ref.rollout.agentcore.agent_name = agentcore_envs.get("AGENT_NAME", "")
    config.actor_rollout_ref.rollout.agentcore.subnets = [
        s for s in agentcore_envs.get("SUBNETS", "").split(",") if s.strip()
    ]
    config.actor_rollout_ref.rollout.agentcore.security_groups = [
        sg for sg in agentcore_envs.get("SECURITY_GROUPS", "").split(",") if sg.strip()
    ]
    config.actor_rollout_ref.rollout.agentcore.container_uri = agentcore_envs.get("CONTAINER_URI", "")
    config.actor_rollout_ref.rollout.agentcore.role_arn = agentcore_envs.get("ROLE_ARN", "")
    config.actor_rollout_ref.rollout.agentcore.s3_bucket = agentcore_envs.get("S3_BUCKET", "")
    config.actor_rollout_ref.rollout.agentcore.reqs_per_sec = 25
    config.actor_rollout_ref.rollout.agentcore.max_rollout_time = 180

    return config


def test_agentcore_config(init_config):
    # First verify AgentCore configuration loads correctly.
    assert init_config is not None
    assert init_config.actor_rollout_ref.rollout.mode == "agentcore"

    agentcore_config = init_config.actor_rollout_ref.rollout.agentcore

    # Required string fields must be non-empty
    required_fields = ["agent_name", "container_uri", "role_arn", "s3_bucket"]

    for field in required_fields:
        value = agentcore_config[field]  # use dict-style access
        assert value != "", f"AgentCore field '{field}' must be set (currently empty)"

    # List fields must have at least one item
    assert len(agentcore_config.subnets) > 0, "subnets must contain at least one subnet"
    assert len(agentcore_config.security_groups) > 0, "security_groups must contain at least one security group"

    # Also verify the dataclass after the hydra config is converted to dataclass
    # Any field missing from verl/workers/config/rollout.py will result in error.
    rollout_config = omega_conf_to_dataclass(init_config.actor_rollout_ref.rollout, dataclass_type=RolloutConfig)
    assert isinstance(rollout_config, RolloutConfig), "Config should convert to RolloutConfig dataclass"
    assert rollout_config.mode == "agentcore", "Rollout mode should be agentcore"

    # Verify agentcore fields exist in dataclass
    assert hasattr(rollout_config, "agentcore"), "RolloutConfig should have agentcore attribute"
    agentcore_dc = rollout_config.agentcore

    # Test the same required fields in the dataclass
    for field in required_fields:
        assert hasattr(agentcore_dc, field), f"AgentCore dataclass missing field: {field}"
        value = getattr(agentcore_dc, field)
        assert value != "", f"AgentCore dataclass field '{field}' must be set (currently empty)"

    # Test list fields in dataclass
    assert hasattr(agentcore_dc, "subnets"), "AgentCore dataclass missing subnets field"
    assert hasattr(agentcore_dc, "security_groups"), "AgentCore dataclass missing security_groups field"
    assert len(agentcore_dc.subnets) > 0, "subnets must contain at least one subnet in dataclass"
    assert len(agentcore_dc.security_groups) > 0, (
        "security_groups must contain at least one security group in dataclass"
    )

    print("AgentCore config initialization and dataclass conversion verified!")


def test_agentcore_rollout_buffer(init_config):
    exp_id = f"test_{time.strftime('%Y%m%d%H%M%S')}"
    rollout_buffer = None
    sqs_client = boto3.client("sqs")
    s3_client = boto3.client("s3")
    s3_bucket = init_config.actor_rollout_ref.rollout.agentcore.s3_bucket
    s3_keys_created = []

    try:
        # Create the rollout buffer
        rollout_buffer = RolloutBuffer(config=init_config, exp_id=exp_id)

        # Check that queue_url exists and is valid
        queue_url = rollout_buffer.get_queue_url()
        assert queue_url is not None
        assert queue_url != ""
        assert isinstance(queue_url, str)
        assert "sqs" in queue_url.lower()
        assert "amazonaws.com" in queue_url.lower()

        # Verify the queue actually exists in AWS by calling get_queue_attributes
        response = sqs_client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["QueueArn"])

        # Verify we got the expected attributes
        assert "Attributes" in response
        attributes = response["Attributes"]
        assert "QueueArn" in attributes

        print(f"Queue verified: {queue_url}")
        print(f"Queue ARN: {attributes['QueueArn']}")

        # Test S3 to SQS integration: Create test rollout messages in S3
        n_rollouts = init_config.actor_rollout_ref.rollout.n
        print(f"Creating {n_rollouts} test rollout messages in S3...")

        for i in range(n_rollouts):
            # Create sample rollout data
            rollout_data = {
                "rollout_data": [
                    {
                        "turn_id": 0,
                        "formatted_request": {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a helpful math assistant. Use the calculator tool.",
                                },
                                {
                                    "role": "user",
                                    "content": [{"text": f"Test problem {i + 1}: What is 2+3?", "type": "text"}],
                                },
                                {
                                    "role": "assistant",
                                    "content": [{"text": "I'll solve 2+3 using the calculator.", "type": "text"}],
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "arguments": '{"expression": "2+3", "mode": "evaluate"}',
                                                "name": "calculator",
                                            },
                                            "id": f"chatcmpl-tool-{i + 1}",
                                            "type": "function",
                                        }
                                    ],
                                },
                            ],
                            "tools": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "description": "Calculator for basic arithmetic operations.",
                                        "parameters": {
                                            "properties": {
                                                "expression": {
                                                    "description": "The mathematical expression to evaluate",
                                                    "type": "string",
                                                },
                                                "mode": {"description": "The calculation mode", "type": "string"},
                                            },
                                            "required": ["expression"],
                                            "type": "object",
                                        },
                                    },
                                }
                            ],
                            "temperature": 1.0,
                        },
                    },
                    {
                        "turn_id": 1,
                        "formatted_request": {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a helpful math assistant. Use the calculator tool.",
                                },
                                {
                                    "role": "user",
                                    "content": [{"text": f"Test problem {i + 1}: What is 2 + 3?", "type": "text"}],
                                },
                                {
                                    "role": "assistant",
                                    "content": [{"text": "I'll solve 2 + 3 using the calculator.", "type": "text"}],
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "arguments": '{"expression": "2 + 3", "mode": "evaluate"}',
                                                "name": "calculator",
                                            },
                                            "id": f"chatcmpl-tool-{i + 1}",
                                            "type": "function",
                                        }
                                    ],
                                },
                                {
                                    "role": "tool",
                                    "tool_call_id": f"chatcmpl-tool-{i + 1}",
                                    "content": [{"text": "Result: 5", "type": "text"}],
                                },
                                {"role": "assistant", "content": [{"text": "The answer is 5.", "type": "text"}]},
                            ],
                            "tools": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "description": "Calculator for basic arithmetic operations.",
                                        "parameters": {
                                            "properties": {
                                                "expression": {
                                                    "description": "The mathematical expression to evaluate",
                                                    "type": "string",
                                                },
                                                "mode": {"description": "The calculation mode", "type": "string"},
                                            },
                                            "required": ["expression"],
                                            "type": "object",
                                        },
                                    },
                                }
                            ],
                            "temperature": 1.0,
                        },
                    },
                ],
                "rewards": [0.0, 0.5 + i * 0.1],  # Rewards for each turn
                "status_code": 200,
                "stop_reason": "end_turn",
                "input_id": str(uuid.uuid4()),
            }

            # Save to S3
            s3_key = f"{exp_id}/rollout_{i + 1}.json"
            s3_keys_created.append(s3_key)

            s3_client.put_object(
                Bucket=s3_bucket, Key=s3_key, Body=json.dumps(rollout_data), ContentType="application/json"
            )
            print(f"Created S3 object: s3://{s3_bucket}/{s3_key}")

            # Send SQS message
            sqs_message = {
                "Records": [
                    {
                        "eventSource": "rollout:collector",
                        "eventName": "ObjectCreated:Put",
                        "eventTime": time.strftime("%Y%m%d%H%M%S"),
                        "s3": {"bucket": {"name": s3_bucket}, "object": {"key": s3_key}},
                    }
                ]
            }

            sqs_client.send_message(QueueUrl=queue_url, MessageBody=json.dumps(sqs_message))

        print("Testing rollout buffer collect_rollout_data...")
        collected_data = rollout_buffer.collect_rollout_data(
            target_size=n_rollouts,
            poll_interval=5,  # Poll every 5 seconds for faster testing
        )

        # Verify we received the expected number of messages
        print(f"Collected {len(collected_data)} rollout messages")
        assert len(collected_data) == n_rollouts, f"Expected {n_rollouts} messages, got {len(collected_data)}"

        # Verify the structure of collected data
        for i, data in enumerate(collected_data):
            assert "rollout" in data, f"Message {i} missing 'rollout' key"
            assert "reward" in data, f"Message {i} missing 'reward' key"
            assert "success" in data, f"Message {i} missing 'success' key"
            assert "uid" in data, f"Message {i} missing 'uid' key"

            # Verify rollout structure
            rollout = data["rollout"]
            assert isinstance(rollout, list), f"Message {i} rollout should be a list"
            assert len(rollout) > 0, f"Message {i} rollout should not be empty"
            assert "formatted_request" in rollout[0], f"Message {i} rollout missing formatted_request"

            # Verify formatted_request structure
            formatted_request = rollout[0]["formatted_request"]
            assert "messages" in formatted_request, f"Message {i} missing messages"
            assert "tools" in formatted_request, f"Message {i} missing tools"

            # Verify messages structure
            messages = formatted_request["messages"]
            assert isinstance(messages, list), f"Message {i} messages should be a list"
            assert len(messages) >= 2, f"Message {i} should have at least 2 messages (user + assistant)"
            assert messages[0]["role"] == "system", f"Message {i} first message should be system role"
            assert messages[1]["role"] == "user", f"Message {i} first message should be user role"
            assert messages[2]["role"] == "assistant", f"Message {i} second message should be assistant role"
            assert "content" in messages[0], f"Message {i} system message missing content"
            assert "content" in messages[1], f"Message {i} user message missing content"
            assert "content" in messages[2], f"Message {i} assistant message missing content"

            # Verify reward structure
            reward = data["reward"]
            assert isinstance(reward, list), f"Message {i} reward should be a list"
            assert len(reward) > 0, f"Message {i} reward should not be empty"
            assert isinstance(reward[-1], int | float), f"Message {i} reward should be numeric"
            expected_max_reward = 0.5 + (n_rollouts - 1) * 0.1
            assert 0.5 <= reward[-1] < expected_max_reward + 0.01, (
                f"Message {i} reward should be in expected range [0.5, {expected_max_reward:.1f}], got {reward[0]}"
            )

        # Verify we got unique rewards (they should be different values)
        rewards = [data["reward"][-1] for data in collected_data]
        unique_rewards = set(rewards)
        assert len(unique_rewards) == n_rollouts, (
            f"Expected at least {n_rollouts} unique rewards, got {len(unique_rewards)}: {rewards}"
        )

        print("S3 to SQS integration test passed!")
        print(f"Successfully processed {len(collected_data)} rollout messages through S3 → SQS → RolloutBuffer")
        print(f"Rewards collected: {sorted(rewards)}")

        # This will also be invoked by PPOTrainer after the last iteration
        rollout_buffer.shutdown()
        time.sleep(2)

        # Expecting the SQS queue to not exist anymore
        actual_queues = sqs_client.list_queues(QueueNamePrefix=rollout_buffer.queue_url.split("/")[-1]).get(
            "QueueUrls", []
        )
        assert len(actual_queues) == 0

        print("SQS queue was deleted as expected")

    finally:
        # Clean up S3 objects
        for s3_key in s3_keys_created:
            try:
                s3_client.delete_object(Bucket=s3_bucket, Key=s3_key)
                print(f"Cleaned up S3 object: {s3_key}")
            except Exception as cleanup_error:
                print(f"Warning: Failed to cleanup S3 object {s3_key}: {cleanup_error}")


def test_agentcore_rollout_buffer_logging(init_config, caplog):
    """Test the logging logic for different status_code and stop_reason combinations."""
    import logging

    exp_id = f"test_success_{time.strftime('%Y%m%d%H%M%S')}"
    rollout_buffer = None
    sqs_client = boto3.client("sqs")
    s3_client = boto3.client("s3")
    s3_bucket = init_config.actor_rollout_ref.rollout.agentcore.s3_bucket
    s3_keys_created = []

    # Test scenarios: (status_code, stop_reason, expected_success, expected_warning_count)
    test_scenarios = [
        (None, None, True, 2),  # Missing both - successful with 2 warnings
        (200, "end_turn", True, 0),  # Perfect case - successful, no warnings
        (500, "end_turn", False, 1),  # Bad status - failed with 1 warning
        (200, "error", False, 1),  # Bad stop reason - failed with 1 warning
        (500, "timeout", False, 2),  # Both bad - failed with 2 warnings
        (None, "end_turn", True, 1),  # Missing status - successful with 1 warning
        (200, None, True, 1),  # Missing stop reason - successful with 1 warning
        (404, "connection_error", False, 2),  # Both bad (different values) - failed with 2 warnings
    ]

    try:
        # Create the rollout buffer
        rollout_buffer = RolloutBuffer(config=init_config, exp_id=exp_id)
        queue_url = rollout_buffer.get_queue_url()

        print(f"Testing {len(test_scenarios)} success flag scenarios...")

        for scenario_idx, (status_code, stop_reason, expected_success, _) in enumerate(test_scenarios):
            # Clear previous log messages for this scenario
            caplog.clear()

            # Create rollout data with specific status_code and stop_reason
            rollout_data_dict = {
                "formatted_request": {
                    "messages": [
                        {"role": "user", "content": f"Test scenario {scenario_idx}: What is 2+2?"},
                        {"role": "assistant", "content": f"Test response {scenario_idx}: The answer is 4."},
                    ],
                    "tools": None,
                }
            }

            rollout_s3_data = {"rollout_data": rollout_data_dict, "rewards": [0.8], "input_id": str(uuid.uuid4())}

            # Add status_code and stop_reason if they are not None
            if status_code is not None:
                rollout_s3_data["status_code"] = status_code
            if stop_reason is not None:
                rollout_s3_data["stop_reason"] = stop_reason

            # Save to S3 with proper prefix to trigger SQS notification
            s3_key = f"{exp_id}/scenario_{scenario_idx}_rollout.json"
            s3_keys_created.append(s3_key)

            s3_client.put_object(
                Bucket=s3_bucket, Key=s3_key, Body=json.dumps(rollout_s3_data), ContentType="application/json"
            )
            print(f"Created S3 object for scenario {scenario_idx}: s3://{s3_bucket}/{s3_key}")

            # Send SQS message
            sqs_message = {
                "Records": [
                    {
                        "eventSource": "rollout:collector",
                        "eventName": "ObjectCreated:Put",
                        "eventTime": time.strftime("%Y%m%d%H%M%S"),
                        "s3": {"bucket": {"name": s3_bucket}, "object": {"key": s3_key}},
                    }
                ]
            }

            sqs_client.send_message(QueueUrl=queue_url, MessageBody=json.dumps(sqs_message))

        # Collect all rollout data
        print("Testing rollout buffer collect_rollout_data with success flags...")
        with caplog.at_level(logging.WARNING):
            collected_data = rollout_buffer.collect_rollout_data(
                target_size=len(test_scenarios),
                poll_interval=5,
            )

        # Verify we received the expected number of messages
        print(f"Collected {len(collected_data)} rollout messages")
        assert len(collected_data) == len(test_scenarios), (
            f"Expected {len(test_scenarios)} messages, got {len(collected_data)}"
        )

        # Group collected data by scenario (we need to match by message content)
        collected_by_scenario = {}
        for data in collected_data:
            # Extract scenario number from the message content
            user_message = data["rollout"]["formatted_request"]["messages"][0]["content"]
            scenario_match = user_message.split("Test scenario ")[1].split(":")[0]
            scenario_idx = int(scenario_match)
            collected_by_scenario[scenario_idx] = data

        # Verify each scenario
        warning_logs = [record.message for record in caplog.records if record.levelname == "WARNING"]
        print(f"Total warning messages logged: {len(warning_logs)}")

        for scenario_idx, (status_code, stop_reason, expected_success, _) in enumerate(test_scenarios):
            print(f"\nVerifying scenario {scenario_idx}: status_code={status_code}, stop_reason={stop_reason}")

            # Verify the collected data has the correct structure
            assert scenario_idx in collected_by_scenario, f"Scenario {scenario_idx} data not found in collected results"
            data = collected_by_scenario[scenario_idx]

            # Verify basic structure
            assert "rollout" in data, f"Scenario {scenario_idx}: missing 'rollout' key"
            assert "reward" in data, f"Scenario {scenario_idx}: missing 'reward' key"
            assert "success" in data, f"Scenario {scenario_idx}: missing 'success' key"

            # Verify success flag
            actual_success = data["success"]
            assert actual_success == expected_success, (
                f"Scenario {scenario_idx}: expected success={expected_success}, got {actual_success}"
            )

            print(f"  ✓ Success flag: {actual_success} (expected: {expected_success})")

        # Verify warning message patterns (we can't easily match specific warnings to scenarios
        # since they're logged during batch processing, but we can verify the general patterns)
        status_code_warnings = [log for log in warning_logs if "status code" in log.lower()]
        stop_reason_warnings = [log for log in warning_logs if "stop reason" in log.lower()]

        print(f"  Status code warnings: {len(status_code_warnings)}")
        print(f"  Stop reason warnings: {len(stop_reason_warnings)}")

        # We should have warnings for scenarios with issues
        assert len(status_code_warnings) >= 2, (
            f"Expected at least 2 status code warnings, got {len(status_code_warnings)}"
        )
        assert len(stop_reason_warnings) >= 2, (
            f"Expected at least 2 stop reason warnings, got {len(stop_reason_warnings)}"
        )

        print("✓ All success flag scenarios passed!")
        print("✓ Warning messages were logged appropriately!")

        # This will also be invoked by PPOTrainer after the last iteration
        rollout_buffer.shutdown()
        time.sleep(2)

        # Expecting the SQS queue to not exist anymore
        actual_queues = sqs_client.list_queues(QueueNamePrefix=rollout_buffer.queue_url.split("/")[-1]).get(
            "QueueUrls", []
        )
        assert len(actual_queues) == 0

        print("SQS queue was deleted as expected")

    finally:
        # Clean up S3 objects
        for s3_key in s3_keys_created:
            try:
                s3_client.delete_object(Bucket=s3_bucket, Key=s3_key)
                print(f"Cleaned up S3 object: {s3_key}")
            except Exception as cleanup_error:
                print(f"Warning: Failed to cleanup S3 object {s3_key}: {cleanup_error}")


def test_agentcore_init(init_config):
    ray.init(
        runtime_env=RAY_RUNTIME_ENV,
        ignore_reinit_error=True,
    )
    agentcore_loop_manager = AgentCoreLoopManager(init_config)

    assert agentcore_loop_manager.agent_id is not None
    assert agentcore_loop_manager.agent_arn is not None

    agentcore_loop_manager.shutdown()
    ray.shutdown()

    print("AgentCoreLoopManager initialization successful!")


def test_agentcore_standalone_rollout(init_config):
    ray.init(
        runtime_env=RAY_RUNTIME_ENV,
        ignore_reinit_error=True,
    )
    agentcore_loop_manager = AgentCoreLoopManager(init_config)

    payloads = [
        {
            "prompt": (
                "Natalia sold clips to 48 of her friends in April, "
                "and then she sold half as many clips in May. "
                "How many clips did Natalia sell altogether in April and May?"
            ),
            "answer": "72",
        },
        {
            "prompt": (
                "Toula went to the bakery and bought various types of pastries. "
                "She bought 3 dozen donuts which cost $68 per dozen, "
                "2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini "
                "cheesecakes for $55 per dozen. How much was the total cost?"
            ),
            "answer": "694",
        },
    ]
    # Follow the behavior of collate_fn at verl/utils/dataset/rl_dataset.py to create input numpy array
    batch = DataProto(
        non_tensor_batch={
            "prompt": np.fromiter([p["prompt"] for p in payloads], dtype=object, count=len(payloads)),
            "answer": np.fromiter([p["answer"] for p in payloads], dtype=object, count=len(payloads)),
            "agent_name": np.fromiter(["math_agent"] * len(payloads), dtype=object, count=len(payloads)),
            "data_source": np.fromiter(["openai/gsm8k"] * len(payloads), dtype=object, count=len(payloads)),
            "uid": np.fromiter([str(uuid.uuid4()) for _ in range(len(payloads))], dtype=object, count=len(payloads)),
        },
    )
    n = init_config.actor_rollout_ref.rollout.n * 4  # test bsz 32
    batch = batch.repeat(n)

    rollout_batch_output = agentcore_loop_manager.generate_sequences(batch)
    assert len(rollout_batch_output) == len(batch)
    assert sorted(batch.non_tensor_batch["uid"].tolist()) == sorted(
        rollout_batch_output.non_tensor_batch["uid"].tolist()
    )
    # TODO: add more checks, particularly, the number of active sessions
    # at this point should be zero. However, the current BedrockAgentCoreClient
    # does not expose the list sessions api yet.
    agentcore_loop_manager.shutdown()
    ray.shutdown()
    print("AgentCoreLoopManager standalone rollout test successful!")


def test_agentcore_post_process_rollout_data(init_config):
    ray.init(
        runtime_env=RAY_RUNTIME_ENV,
        ignore_reinit_error=True,
    )
    agentcore_loop_manager = AgentCoreLoopManager(init_config)

    rollout_data_batch = []

    uid1 = str(uuid.uuid4())
    message = [
        {"role": "user", "content": "Hello, who are you?"},
        {"role": "assistant", "content": "I am your assistant."},
    ]
    rollout = [{"formatted_request": {"messages": message, "tools": None}}]
    rollout_data_batch.append({"rollout": rollout, "reward": [1.0], "uid": uid1})

    message = [
        {"role": "user", "content": "Hello, who are you?"},
        {"role": "assistant", "content": "I am your AI assistant."},
    ]
    rollout = [{"formatted_request": {"messages": message, "tools": None}}]
    rollout_data_batch.append({"rollout": rollout, "reward": [1.0], "uid": uid1})

    uid2 = str(uuid.uuid4())
    message = [
        {"role": "user", "content": "Hello, how can you help me?"},
        {"role": "assistant", "content": "Ask me any question."},
    ]
    rollout = [{"formatted_request": {"messages": message, "tools": None}}]
    rollout_data_batch.append({"rollout": rollout, "reward": [1.0], "uid": uid2})

    rollout_batch_output = agentcore_loop_manager._post_process_rollout_data(rollout_data_batch, 3)
    agentcore_loop_manager.shutdown()
    ray.shutdown()

    for tensor_key in [
        "attention_mask",
        "input_ids",
        "position_ids",
        "prompts",
        "response_mask",
        "responses",
        "reward_tensor",
    ]:
        assert tensor_key in rollout_batch_output.batch
    assert "timing" in rollout_batch_output.meta_info
    assert "uid" in rollout_batch_output.non_tensor_batch

    assert rollout_batch_output.batch.batch_size[0] == 3

    for tensor_key in ["attention_mask", "input_ids", "position_ids"]:
        assert rollout_batch_output.batch[tensor_key].shape[1] == 8192

    for tensor_key in ["prompts", "response_mask", "responses", "reward_tensor"]:
        assert rollout_batch_output.batch[tensor_key].shape[1] == 4096

    assert rollout_batch_output.non_tensor_batch["uid"].tolist() == [uid1, uid1, uid2]
