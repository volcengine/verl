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
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import ray
import requests
import torch
from bedrock_agentcore_starter_toolkit.services.runtime import BedrockAgentCoreClient, generate_session_id
from botocore.config import Config as BotocoreConfig
from dotenv import dotenv_values
from omegaconf import DictConfig
from tensordict import TensorDict
from vllm.entrypoints.chat_utils import _detect_content_format

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, compute_position_id_with_mask
from verl.protocol import DataProto, pad_dataproto_to_divisor
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.workers.rollout.replica import get_rollout_replica_class

# seconds - AgentCore new session cold start time under 25 TPS for container deployment (2025-11)
SESSION_START_TIME = 10

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_aws_region():
    """Auto-detect AWS region, raise error if none found."""
    try:
        session = boto3.Session()
        region = session.region_name
        if region:
            return region
    except Exception as e:
        logger.warning(f"Failed to get region from boto3 session: {e}")

    # Check environment variables
    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
    if region:
        return region

    # If still no region, provide a helpful error
    raise ValueError(
        "No AWS region found. Please either:\n"
        "1. Set AWS_DEFAULT_REGION environment variable\n"
        "2. Configure AWS CLI with 'aws configure'\n"
        "3. Set up ~/.aws/config with a default region"
    )


class RolloutBuffer:
    def __init__(self, config: DictConfig, exp_id: str):
        self.config = config.actor_rollout_ref.rollout.agentcore
        self.exp_id = exp_id

        self.storage = boto3.client("s3")
        self.queue = boto3.client("sqs")

        self.queue_url = self._setup_sqs()

        # Track processed message IDs to handle SQS at-least-once delivery
        self.processed_message_ids = set()

    def _setup_sqs(self):
        queue_name = f"{self.exp_id}_rollout-queue"

        # Create a separate queue for each experiment
        queue_response = self.queue.create_queue(
            QueueName=queue_name,
            Attributes={
                "VisibilityTimeout": "300",  # 5 minutes
                "MessageRetentionPeriod": "1209600",  # 14 days
                "ReceiveMessageWaitTimeSeconds": "20",  # Long polling
            },
        )
        queue_url = queue_response["QueueUrl"]

        return queue_url

    def get_queue_url(self):
        return self.queue_url

    def collect_rollout_data(self, target_size: int, poll_interval: int = 20):
        """Collect rollout data from SQS/S3 until target size or timeout.

        Args:
            target_size (int): Target number of rollouts to collect.
            poll_interval (int): Polling interval in seconds.

        Returns:
            List[dict]: List of rollout data dictionaries.
        """
        start_time = time.time()
        num_rollouts = 0
        rollout_data_batch = []

        # TODO: deal with case where the number of rollouts collected is less than the target
        # size within the allowed time frame.
        while time.time() - start_time < self.config.max_rollout_time and num_rollouts < target_size:
            # Poll SQS for messages
            poll_start = time.time()
            response = self.queue.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=10,  # set to max num messages allowed by sqs
                WaitTimeSeconds=poll_interval,
            )
            poll_duration = time.time() - poll_start

            rollout_data = self._process_sqs_messages(response)
            num_rollouts += len(rollout_data)
            rollout_data_batch.extend(rollout_data)

            elapsed_time = time.time() - start_time

            logger.debug(
                f"{elapsed_time:.1f}s elapsed. {len(rollout_data_batch)} out of {target_size} rollouts collected. "
                f"Last poll took {poll_duration:.1f}s, received {len(rollout_data)} messages."
            )

        logger.info(f"{elapsed_time:.1f}s elapsed. {len(rollout_data_batch)} out of {target_size} rollouts collected.")

        # Filter out empty rollouts due to errors; we counted them previously so that we don't
        # wait on rollouts that already errored out.
        rollout_data_batch = [r for r in rollout_data_batch if r]

        return rollout_data_batch

    def _process_sqs_messages(self, response):
        """Transform SQS message response into rollout data."""
        rollout_data = []

        if "Messages" not in response:
            return rollout_data

        for message in response["Messages"]:
            message_id = message["MessageId"]

            # Check if we've already processed this message (deduplication)
            if message_id in self.processed_message_ids:
                logger.warning(f"Duplicate message detected and skipped: {message_id}")
                # Still delete the duplicate message from the queue
                try:
                    self.queue.delete_message(
                        QueueUrl=self.queue_url,
                        ReceiptHandle=message["ReceiptHandle"],
                    )
                except Exception as e:
                    logger.warning(f"Failed to delete duplicate message {message_id}: {e}")
                continue

            # Mark message as processed
            self.processed_message_ids.add(message_id)

            # Process message and always clean up
            rollouts = self._process_single_sqs_message(message)
            rollout_data.extend(rollouts)

            # Clean up happens once, at the top level
            try:
                self.queue.delete_message(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=message["ReceiptHandle"],
                )
            except Exception as e:
                logger.error(f"Failed to delete processed message {message_id}: {e}")

        return rollout_data

    def _process_single_sqs_message(self, message):
        """Process a single SQS message and return rollout data."""
        message_body = json.loads(message["Body"])

        if "Records" not in message_body:
            logger.warning(f"SQS message missing 'Records' field. Message body: {json.dumps(message_body, indent=2)}\n")
            return []

        rollout_data = []
        for record in message_body["Records"]:
            s3_response = self.storage.get_object(
                Bucket=record["s3"]["bucket"]["name"],
                Key=record["s3"]["object"]["key"],
            )
            s3_data = json.loads(s3_response["Body"].read().decode("utf-8"))

            success = True
            status_code = s3_data.get("status_code")
            stop_reason = s3_data.get("stop_reason")

            if status_code != 200:
                if not status_code:
                    logger.warning(
                        "Please consider returning status code for rollouts for observability and data filtering. "
                        "Defaulting rollout outcome as successful."
                    )
                else:
                    success = False
                    logger.warning(
                        f"Rollout status code: {status_code} != 200. Marking rollout outcome as unsuccessful."
                    )

            if stop_reason != "end_turn":
                if not stop_reason:
                    logger.warning("Please consider returning stop reason for rollouts for observability.")
                else:
                    success = False
                    logger.warning(
                        f"Rollout stop reason: {stop_reason} != end_turn. Marking rollout outcome as unsuccessful."
                    )

            if "rollout_data" in s3_data and "rewards" in s3_data:
                rollout_data.append(
                    {
                        "rollout": s3_data["rollout_data"],
                        "reward": s3_data["rewards"],
                        "uid": s3_data["input_id"],
                        "success": success,
                    }
                )
            else:
                rollout_data.append({})  # signal that we already heard from the request, tho it errored out.
                logger.warning(
                    f"Attention! Skipping rollout missing 'rollout_data' and 'rewards' keys. Existing data: {s3_data}"
                )

        return rollout_data

    def clear_buffer(self):
        """Clear all messages from the rollout buffer queue."""
        try:
            # Clear the processed message IDs when clearing the buffer
            self.processed_message_ids.clear()
            self.queue.purge_queue(QueueUrl=self.queue_url)
            logger.info("Queue purge initiated and processed message IDs cleared")
        except Exception as e:
            logger.info(f"Error purging queue: {e}")

    def shutdown(self):
        """Delete the queue and all messages in it."""
        try:
            self.queue.delete_queue(QueueUrl=self.queue_url)
            logger.info("Queue deleted during shutdown")
        except Exception as e:
            logger.error("Failed to delete queue", exc_info=e)


class RequestDispatcher:
    def __init__(self, max_tps=25):
        max_inflight_requests = max_tps * SESSION_START_TIME  # provides peak throughput
        config = BotocoreConfig(
            read_timeout=900,
            connect_timeout=60,
            retries={"max_attempts": 3},
            max_pool_connections=max_inflight_requests,
        )
        self.client = boto3.client("bedrock-agentcore", region_name=get_aws_region(), config=config)
        self.executor = ThreadPoolExecutor(max_workers=max_inflight_requests)
        self.max_tps = max_tps
        self.request_count = 0

    def submit_request(self, agent_arn: str, session_id: str, payload: dict):
        """Submit a request to AgentCore Runtime with rate limiting.

        Ensures no more than `max_tps` requests are submitted per second.
        Requests are submitted asynchronously (fire and forget) using the executor.

        Args:
            agent_arn: The ARN of the agent runtime
            session_id: The session ID for the request
            payload: The request payload dictionary
        """
        # Submit request asynchronously (fire and forget)
        # Capture request count before submitting to avoid race condition
        cur_req_count = self.request_count

        def _invoke():
            start = time.time()
            try:
                self.client.invoke_agent_runtime(
                    agentRuntimeArn=agent_arn, runtimeSessionId=session_id, payload=json.dumps(payload)
                )
                logger.info(f"{cur_req_count + 1}th request submitted successfully, taking {time.time() - start:.2f}s.")
            except Exception as e:
                logger.error(f"{cur_req_count + 1}th request failed to submit, taking {time.time() - start:.2f}s: {e}")

        self.executor.submit(_invoke)
        self.request_count += 1

        # Sleep after every max_tps requests
        if self.request_count % self.max_tps == 0:
            logger.info(f"Rate limit: submitted {self.max_tps} requests, sleeping for 1 second")
            time.sleep(1.0)

    def shutdown(self):
        """Shutdown the thread pool executor, canceling all pending tasks."""
        self.executor.shutdown(wait=False, cancel_futures=True)
        logger.info("RequestDispatcher shutdown complete")


class AgentCoreLoopManager(AgentLoopManager):
    """Manager that manages a group of agent loops deployed at AWS Bedrock AgentCore Runtime."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup = None):
        """Initialize AgentCore loop manager.

        Args:
            config (DictConfig): global verl config.
            worker_group (RayWorkerGroup): AsyncActorRolloutRef worker group.
        """
        self.config = config
        self.worker_group = worker_group

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        # Get lengths of some special token id sequences of the actor model
        # for producing response_mask in _post_process_rollout_data later.
        self.internal_system_seq_len, self.assistant_start_seq_len = self._get_special_seq_len()
        # Not every model is compatiable with the latest OpenAI message format {"role": <str>, "content": <list[dict]>}
        # Following vLLM, we first detect if the model's chat template only allows using string in content field.
        # If so, we will automatically transform list[dict] to string (concatenate each dict's "text" by "\n")
        # following vLLM in data post processing.
        self.content_format = _detect_content_format(self.tokenizer.chat_template, default="string")

        self.agentcore_client = BedrockAgentCoreClient(region=get_aws_region())
        self.llm_router = SGLangRouterActor.remote(self.config)
        self.req_dispatcher = RequestDispatcher(max_tps=self.config.actor_rollout_ref.rollout.agentcore.reqs_per_sec)
        self.exp_id = self._generate_exp_id()

        self.rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)

        self._initialize_llm_servers_and_router()
        self._initialize_rollout_buffer()
        self._initialize_agentcore_runtime()

        # Initially all servers are in sleep mode.
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

    def _generate_exp_id(self):
        container_uri = self.config.actor_rollout_ref.rollout.agentcore.container_uri
        agent_id = "-".join(container_uri.split("/")[-1].split(":"))
        timestamp = time.strftime("%Y%m%d%H%M%S")
        return f"{agent_id}_{timestamp}"

    def _get_special_seq_len(self):
        empty_messages = [{}]
        internal_system_seq_len = len(
            self.tokenizer.apply_chat_template(empty_messages, tokenizer=True, add_generation_prompt=False)
        )
        assistant_start_seq_len = (
            len(self.tokenizer.apply_chat_template(empty_messages, tokenizer=True, add_generation_prompt=True))
            - internal_system_seq_len
        )

        return internal_system_seq_len, assistant_start_seq_len

    def _initialize_llm_servers_and_router(self):
        """Initialize async llm servers and router."""
        self._initialize_llm_servers()

        ray.get(self.llm_router.wait_healthy.remote())

        # Add new workers
        for server_address in self.server_addresses:
            ray.get(self.llm_router.add_worker.remote(server_address))

    def _initialize_rollout_buffer(self):
        """Initialize rollout buffer."""
        self.rollout_buffer = RolloutBuffer(config=self.config, exp_id=self.exp_id)

    def _initialize_agentcore_runtime(self):
        """Initialize AgentCore Runtime client."""
        env_vars = dotenv_values("agentcore.env")

        # give router ip as base url to agentcore runtime
        env_vars["BASE_URL"] = f"{ray.get(self.llm_router.get_endpoint.remote())}/v1"
        env_vars["MODEL_ID"] = self.config.actor_rollout_ref.model.path

        network_config = {
            "networkMode": "VPC",
            "networkModeConfig": {
                "subnets": list(self.config.actor_rollout_ref.rollout.agentcore.subnets),
                "securityGroups": list(self.config.actor_rollout_ref.rollout.agentcore.security_groups),
            },
        }
        lifecycle_config = {
            "idleRuntimeSessionTimeout": 60,  # 1 minute is the minimum
            "maxLifetime": self.config.actor_rollout_ref.rollout.agentcore.max_rollout_time,
        }

        response = self.agentcore_client.create_agent(
            agent_name=self.config.actor_rollout_ref.rollout.agentcore.agent_name,
            deployment_type="container",
            image_uri=self.config.actor_rollout_ref.rollout.agentcore.container_uri,
            execution_role_arn=self.config.actor_rollout_ref.rollout.agentcore.role_arn,
            network_config=network_config,
            env_vars=env_vars,
            lifecycle_config=lifecycle_config,
            auto_update_on_conflict=True,
        )

        self.agent_id = response["id"]
        self.agent_arn = response["arn"]

        # Wait for deployment to be ready
        endpoint_response = self.agentcore_client.wait_for_agent_endpoint_ready(agent_id=self.agent_id, max_wait=120)
        # When timed out, the response is an error string instead of the actual endpoint arn
        if self.agent_arn not in endpoint_response:
            raise TimeoutError(endpoint_response)

        logger.info(f"Agent endpoint ARN: {endpoint_response}")

    def _agentcore_loop(self, rollout_batch_input: DataProto):
        """Run agent loop in AgentCore using RequestDispatcher with rate limiting."""
        # non_tensor_batch should contain all required fields in the right format
        # expected by agentcore runtime endpoint

        start_time = time.time()
        reqs = []  # needed to perform filtering later.

        for i in range(len(rollout_batch_input)):
            non_tensor_item = rollout_batch_input[i].non_tensor_batch
            session_id = generate_session_id()
            payload = {k: v for k, v in non_tensor_item.items()}

            if "uid" not in payload:
                raise ValueError(
                    "Missing 'uid' in input data. UIDs are required to group outputs by input prompts "
                    + "during advantage computation."
                )

            if "_training" in payload:
                logger.warning("_training key already exists in non_tensor_item, overwriting it.")

            payload["_training"] = {
                "exp_id": self.exp_id,
                "session_id": session_id,
                "s3_bucket": self.config.actor_rollout_ref.rollout.agentcore.s3_bucket,
                "sqs_url": self.rollout_buffer.get_queue_url(),
                "input_id": payload.pop("uid"),
            }

            reqs.append({session_id: payload})

            # Submit request asynchronously with rate limiting (`self.req_dispatcher.max_tps` requests/second)
            self.req_dispatcher.submit_request(agent_arn=self.agent_arn, session_id=session_id, payload=payload)

        logger.info(
            f"Submitted {len(reqs)} reqs to AgentCore, taking {time.time() - start_time:.2f}s "
            f"with rate limiting of {self.req_dispatcher.max_tps} requests/second."
        )
        return reqs

    def _cleanup_sessions(self, reqs):
        """Clean up AgentCore sessions by terminating them."""
        if not reqs:
            logger.info("No sessions to clean up.")
            return

        logger.info(f"Starting cleanup of {len(reqs)} AgentCore sessions...")
        cleanup_start_time = time.time()
        successful_cleanups = 0
        failed_cleanups = 0

        for req in reqs:
            # Extract session_id from the request dict {session_id: payload}
            session_id = next(iter(req.keys()))
            try:
                self.agentcore_client.stop_runtime_session(agent_arn=self.agent_arn, session_id=session_id)
                successful_cleanups += 1
                logger.debug(f"Successfully terminated session: {session_id}")
            except Exception as e:
                failed_cleanups += 1
                # Log as warning since sessions might already be terminated or expired
                logger.warning(f"Failed to terminate session {session_id}: {e}")

        cleanup_duration = time.time() - cleanup_start_time
        logger.info(
            f"Session cleanup completed in {cleanup_duration:.2f}s. "
            f"Successful: {successful_cleanups}, Failed: {failed_cleanups}"
        )

    def _get_rollout_data(self, target_size: int):
        """Get rollout data for RL training."""
        rollout_data_batch = self.rollout_buffer.collect_rollout_data(target_size)

        logger.info(f"{target_size} sessions were launched and {len(rollout_data_batch)} rollout traces were obtained.")

        if len(rollout_data_batch) == 0:
            raise RuntimeError(
                "No rollout data received from SQS. "
                "Check previous logs for agent failures or verify S3/SQS connectivity."
            )

        if len(rollout_data_batch) < target_size:
            logger.warning(
                f"{target_size - len(rollout_data_batch)} rollout traces are missing. "
                "Check previous logs to find agent failures."
            )

        return self._post_process_rollout_data(rollout_data_batch, target_size)

    def _clear_rollout_buffer(self):
        """Clear all messages from the rollout buffer queue."""
        self.rollout_buffer.clear_buffer()

    def _post_process_rollout_data(self, rollout_data_batch, target_size: int):
        """Post-process raw rollout data batch to DataProto format in verl.

        Args:
            rollout_data_batch (list[dict]): list of raw rollout data dict,
                each dict has multi-turn text data ("rollout") and reward ("reward").

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.
            - reward_tensor: [bsz, response_length], reward scores.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        # Get tokenized and padded tensor data per rollout and concatenate them to a batch
        tensor_data_dicts = [
            self._tokenize_and_pad_multi_turn_rollout(rollout_data["rollout"]) for rollout_data in rollout_data_batch
        ]

        tensor_data_batch = torch.cat(tensor_data_dicts, dim=0)

        # Get outcome reward score tensor.
        rewards = [rollout_data["reward"][-1] for rollout_data in rollout_data_batch]
        prompt_length = tensor_data_batch["prompts"].size(1)
        response_end_idx = tensor_data_batch["attention_mask"][:, prompt_length:].sum(dim=1) - 1
        reward_tensor = torch.zeros_like(tensor_data_batch["response_mask"], dtype=torch.float32)
        reward_tensor[torch.arange(tensor_data_batch.batch_size[0]), response_end_idx] = torch.tensor(
            rewards, dtype=torch.float32
        )
        tensor_data_batch["reward_tensor"] = reward_tensor

        uids = [rollout_data["uid"] for rollout_data in rollout_data_batch]
        batch = DataProto(
            batch=tensor_data_batch, non_tensor_batch={"uid": np.array(uids, dtype=object)}, meta_info={"timing": {}}
        )

        if target_size > len(tensor_data_dicts):
            logger.warning(
                f"Attention! Because {target_size - len(tensor_data_dicts)} agents' rollout traces are missing,"
                " we pad existing data examples to avoid cuda errors."
            )
            batch, _ = pad_dataproto_to_divisor(batch, target_size)

        return batch

    def _tokenize_and_pad_multi_turn_rollout(self, rollout_list):
        """Tokenize the raw multi-turn rollout text to tensors and do padding.

        Args:
            rollout_list (list): the list with every turn's OpenAI format message.

        Returns:
            TensorDict: batch size 1 tensor dict.
            - prompts: [1, prompt_length], prompt token ids from dataset.
            - responses: [1, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [1, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [1, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [1, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [1, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        # Get tool information and the complete message histroy from the last turn.
        tools = rollout_list[-1]["formatted_request"]["tools"]
        full_messages = rollout_list[-1]["formatted_request"]["messages"]

        # We need to check from which turn, actor model starts to make the first response. All
        # messages before it is considered as "prompt" part (it could also have "assistant" role
        # message), those after it (including) is considered as "response" part.
        # This separation point is at the final "assistant" role message of the first turn messages
        # as this is the place where actor model makes the first response.
        first_turn_messages = rollout_list[0]["formatted_request"]["messages"]
        response_start = len(first_turn_messages)
        for turn in range(len(first_turn_messages) - 1, -1, -1):
            if first_turn_messages[turn]["role"] == "assistant":
                response_start = turn
                break

        # Following vLLM, if chat template only allows using string in content field, we need to convert list
        # to string.
        for cur_turn_message in full_messages:
            content = cur_turn_message["content"]
            if self.content_format == "string" and isinstance(content, list) and len(content) > 0:
                content_texts = [content_item["text"] for content_item in content if "text" in content_item]
                cur_turn_message["content"] = "\n".join(content_texts)

        # Tokenize and pad prompt at left side.
        # TODO: in padding, we pad sequences to the same length, but it is possible the sequence is
        # longer than the pad length. Now we have to do hard cutoff, we need to consider how to handle
        # them in the future.
        prompt_messages = full_messages[:response_start]
        prompt_token_ids = self.tokenizer.apply_chat_template(prompt_messages, tokenize=True, tools=tools)
        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": prompt_token_ids[: self.config.actor_rollout_ref.rollout.prompt_length]},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        # Tokenize response, get response mask where 1 indicates actor generated tokens.
        response_token_ids, response_mask = [], []
        for turn in range(response_start, len(full_messages)):
            cur_turn_message = full_messages[turn]
            cur_turn_token_ids = self.tokenizer.apply_chat_template([cur_turn_message])[self.internal_system_seq_len :]
            response_token_ids.extend(cur_turn_token_ids)

            mask = [0] * len(cur_turn_token_ids)
            if cur_turn_message["role"] == "assistant":
                mask = [0] * self.assistant_start_seq_len + [1] * (
                    len(cur_turn_token_ids) - self.assistant_start_seq_len
                )
            response_mask.extend(mask)

        # Pad response and response mask at right side.
        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": response_token_ids[: self.config.actor_rollout_ref.rollout.response_length]},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_output = self.tokenizer.pad(
            {"input_ids": response_mask[: self.config.actor_rollout_ref.rollout.response_length]},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        # Get final output.
        # In attention mask, 1 indicates non-padding tokens.
        # In response mask, 1 indicates actor generated tokens.
        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        output_dict = TensorDict(
            {
                "prompts": prompt_output["input_ids"],
                "responses": response_output["input_ids"],
                "response_mask": response_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=1,
        )
        return output_dict

    def generate_sequences(self, rollout_batch_input: DataProto) -> DataProto:
        """The rollout function called by main training loop.

        Args:
            rollout_batch_input (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        # Wake up all servers for handling inference requests.
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()

        self._clear_rollout_buffer()
        reqs = self._agentcore_loop(rollout_batch_input)
        rollout_batch_output = self._get_rollout_data(target_size=len(rollout_batch_input))

        # Clean up AgentCore sessions after rollout data collection.
        # We might want to make this & self._clear_rollout_buffer() above optional
        # if we want off-policy learning.
        self._cleanup_sessions(reqs)

        # Set all servers to sleep mode after rollout process terminates.
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        return rollout_batch_output

    def shutdown(self):
        """Clean up any AWS resources after the training has finished"""
        self.rollout_buffer.shutdown()
        self.req_dispatcher.shutdown()


@ray.remote
class SGLangRouterActor:
    def __init__(self, config: DictConfig):
        self.host = config.actor_rollout_ref.rollout.agentcore.get("router_host") or ray.util.get_node_ip_address()
        self.port = config.actor_rollout_ref.rollout.agentcore.get("router_port") or 42_000
        self.timeout = config.actor_rollout_ref.rollout.agentcore.get("router_timeout") or 120
        self.endpoint = f"http://{self.host}:{self.port}"

        # By default, when the actor stops due to driver termination or system fault,
        # Ray will terminate all direct children of the worker process,
        # See: https://docs.ray.io/en/latest/ray-core/user-spawn-processes.html
        self.process = subprocess.Popen(
            [sys.executable, "-m", "sglang_router.launch_router", "--host", self.host, "--port", str(self.port)]
        )

    def get_endpoint(self):
        return self.endpoint

    def wait_healthy(self):
        # State for retry and timeout
        done = False
        start = time.time()

        while not done:
            try:
                done = requests.get(f"{self.endpoint}/health", timeout=self.timeout).status_code == 200
            except Exception:
                # Ignoring all exceptions because the router won't be healthy immediately
                pass

            if not done and time.time() - start > self.timeout:
                raise Exception("Reached timeout before the router became healthy")

        logger.info(f"LLM router is healthy at {self.endpoint}")

    def add_worker(self, worker_address: str):
        try:
            url = f"{self.endpoint}/workers"
            payload = {"url": f"http://{worker_address}"}
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            logger.info(f"Successfully registered {worker_address=} with router")
        except Exception as e:
            logger.error(f"Failed to register with router: {e}")
            raise
