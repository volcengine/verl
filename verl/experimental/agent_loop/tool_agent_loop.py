# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import copy
import json
import logging
import os
from enum import Enum
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"Initialized tools: {cls.tools}")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        # Initialize interactions from config file
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex

        # Initialize state machine
        state = AgentState.PENDING

        # Initialize other variables
        prompt_ids = []
        response_mask = []
        turn_scores = []
        tools_kwargs = kwargs.get("tools_kwargs", {})
        user_turns, assistant_turns = 0, 0
        add_messages = []

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)

        # State machine loop
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(
                    messages, image_data, sampling_params, request_id, prompt_ids, response_mask, metrics
                )
            elif state == AgentState.GENERATING:
                state, response_ids, tool_calls = await self._handle_generating_state(
                    messages, image_data, sampling_params, request_id, prompt_ids, response_mask, metrics, add_messages
                )
                # Update turns count after generation
                assistant_turns += 1
            elif state == AgentState.PROCESSING_TOOLS:
                state, responses = await self._handle_processing_tools_state(
                    tool_calls,
                    tools_kwargs,
                    metrics,
                    add_messages,
                    messages,
                    image_data,
                    interaction,
                    prompt_ids,
                    response_mask,
                )
            elif state == AgentState.INTERACTING:
                state, responses, reward = await self._handle_interacting_state(
                    interaction, request_id, messages, interaction_kwargs, add_messages
                )
                if reward is not None:
                    turn_scores.append(reward)
                user_turns += 1
            else:
                # This should never happen
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]
        multi_modal_data = {"image": image_data} if image_data is not None else {}

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
            extra_fields={"turn_scores": turn_scores},
        )
        return output

    async def _handle_pending_state(
        self, messages, image_data, sampling_params, request_id, prompt_ids, response_mask, metrics
    ):
        """Handle the pending state: prepare the prompt and start generation."""
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            prompt_ids[:] = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids[:] = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, messages, image_data, sampling_params, request_id, prompt_ids, response_mask, metrics, add_messages
    ):
        """Handle the generating state: generate model response and check for tool calls."""
        with simple_timer("generate_sequences", metrics):
            response_ids = await self.server_manager.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
            )
        prompt_ids += response_ids
        response_mask += [1] * len(response_ids)

        # Check termination conditions
        if len(response_mask) >= self.response_length:
            return AgentState.TERMINATED, response_ids, []

        # Extract tool calls
        _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(None, lambda id=response_ids: self.tokenizer.decode(id))
            add_messages.append({"role": "assistant", "content": assistant_message})

        # Determine next state
        if tool_calls:
            return AgentState.PROCESSING_TOOLS, response_ids, tool_calls
        elif self.interaction_config_file:
            return AgentState.INTERACTING, response_ids, []
        else:
            return AgentState.TERMINATED, response_ids, []

    async def _handle_processing_tools_state(
        self,
        tool_calls,
        tools_kwargs,
        metrics,
        add_messages,
        messages,
        image_data,
        interaction,
        prompt_ids,
        response_mask,
    ):
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        tasks = []
        for tool_call in tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, tools_kwargs))
        with simple_timer("tool_calls", metrics):
            responses = await asyncio.gather(*tasks)

        # Handle responses for interaction if needed
        if self.interaction_config_file:
            for response in responses:
                if response.text:
                    messages.append({"role": "tool", "content": response.text})
                    print(f"messages: {messages}")

        # Process tool responses and update multi_modal_data
        new_images_this_turn = []
        for tool_response in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                if image_data is None:
                    image_data = []
                elif not isinstance(image_data, list):
                    image_data = [image_data]

                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            image_data.append(img)
                            new_images_this_turn.append(img)
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        image_data.append(tool_response.image)
                        new_images_this_turn.append(tool_response.image)

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            # Update prompt with tool responses
            if self.processor is not None:
                raw_tool_response = await self.loop.run_in_executor(
                    None,
                    lambda messages=add_messages: self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    ),
                )
                # Use only the new images from this turn for processing tool responses
                current_images = new_images_this_turn if new_images_this_turn else None
                model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
                response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
            else:
                response_ids = await self.loop.run_in_executor(
                    None,
                    lambda messages=add_messages: self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True
                    ),
                )
            response_ids = response_ids[len(self.system_prompt) :]

            # Update prompt_ids and response_mask
            prompt_ids += response_ids
            response_mask += [0] * len(response_ids)

        return AgentState.GENERATING, responses

    async def _handle_interacting_state(self, interaction, request_id, messages, interaction_kwargs, add_messages):
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await interaction.generate_response(request_id, messages, **interaction_kwargs)
        responses = [{"role": "user", "content": interaction_responses}]
        add_messages.append(responses)

        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED, responses, reward
        else:
            return AgentState.GENERATING, responses, reward

    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]) -> ToolResponse:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, _, _ = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return ToolResponse(
                text=f"Error when executing tool: {e}",
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs)

    @classmethod
    def _initialize_interactions(cls, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map
