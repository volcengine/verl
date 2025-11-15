import asyncio
import json
import yaml
import os
from typing import Any
from pathlib import Path
from typing_extensions import Self
from threading import Thread

from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig, TemplateConfig
from sweagent.environment.swe_env import SWEEnv, EnvironmentConfig
from sweagent.agent.models import GenericAPIModelConfig
from sweagent.agent.problem_statement import TextProblemStatement
from sweagent.tools.tools import ToolConfig
from sweagent.tools.parsing import Identity
from sweagent.types import AgentRunResult
from sweagent.environment.repo import PreExistingRepoConfig
from sweagent.run.run_single import RunSingle, RunSingleConfig
from swebench.harness.test_spec.test_spec import make_test_spec
from sweagent.run.common import save_predictions

from recipe.swe_agent.eval import run_eval

from recipe.swe_agent.chat_model import ChatModel
from recipe.swe_agent.re_swe_agent import ReSweAgent
from recipe.swe_agent.vefaas_deployment import VefaasDeployment, VefaasDeploymentConfig
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

class ReRunSingle(RunSingle):
    def run(self) -> AgentRunResult:
        self._chooks.on_start()
        self.logger.info("Starting environment")
        self.env.start()
        self.logger.info("Running agent")
        self._chooks.on_instance_start(index=0, env=self.env, problem_statement=self.problem_statement)
        output_dir = self.output_dir / self.problem_statement.id
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.agent.replay_config is not None:  # type: ignore[attr-defined]
            (output_dir / "config.yaml").write_text(yaml.dump(self.agent.replay_config.model_dump_json(), indent=2))  # type: ignore[attr-defined]
        result = self.agent.run(
            problem_statement=self.problem_statement,
            env=self.env,
            output_dir=output_dir,
        )
        self._chooks.on_instance_completed(result=result)
        self.logger.info("Done")
        self._chooks.on_end()
        save_predictions(self.output_dir, self.problem_statement.id, result)
        self.env.close()
        return result


class SWEAgentLoop(AgentLoopBase):
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        tools_kwargs = kwargs["tools_kwargs"]
        dataset_id, instance_id = tools_kwargs["dataset_id"], tools_kwargs["instance_id"]
        metadata, image = tools_kwargs["metadata"], tools_kwargs["image"]

        temperature = sampling_params.get("temperature", 0.0)
        top_p = sampling_params.get("top_p", 0.0)

        model_path = self.config.actor_rollout_ref.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        rollout = self.config.actor_rollout_ref.rollout
        max_model_len = (
            rollout.max_model_len if rollout.max_model_len else rollout.prompt_length + rollout.response_length
        )

        agent_yaml_path = os.path.join(os.path.dirname(__file__), "agent.yaml")
        with open(agent_yaml_path, "r") as f:
            main_config = yaml.safe_load(f)[0]
        print(main_config)
        os.environ['SWE_AGENT_TRAJECTORY_DIR'] = os.path.join(os.path.dirname(__file__), 'trajectory')
        agent_config_data = main_config.get("agent", {})

        template_config_data = agent_config_data.get("templates", {})
        template_config = TemplateConfig(
            system_template=template_config_data.get("system_template", ""),
            instance_template=template_config_data.get("instance_template", ""),
            next_step_template=template_config_data.get("next_step_template", "Observation: {{observation}}"),
            next_step_truncated_observation_template=template_config_data.get("next_step_truncated_observation_template",
                                                                          "Observation: {{observation[:max_observation_length]}}<response clipped><NOTE>Observations should not exceeded {{max_observation_length}} characters. {{elided_chars}} characters were elided. Please try a different command that produces less output or use head/tail/grep/redirect the output to a file. Do not use interactive pagers.</NOTE>"),
            max_observation_length=template_config_data.get("max_observation_length", 100000),
            next_step_no_output_template=template_config_data.get("next_step_no_output_template", ""),
            strategy_template=template_config_data.get("strategy_template", ""),
            demonstration_template=template_config_data.get("demonstration_template", ""),
            demonstrations=template_config_data.get("demonstrations", []),
            put_demos_in_history=template_config_data.get("put_demos_in_history", False),
            disable_image_processing=template_config_data.get("disable_image_processing", False),
            shell_check_error_template=template_config_data.get("shell_check_error_template",
                                                            "Your bash command contained syntax errors and was NOT executed. Please fix the syntax errors and try again. This can be the result of not adhering to the syntax for multi-line commands. Here is the output of `bash -n`:\n{{bash_stdout}}\n{{bash_stderr}}"),
            command_cancelled_timeout_template=template_config_data.get("command_cancelled_timeout_template",                                                           "The command '{{command}}' was cancelled because it took more than {{timeout}} seconds. Please try a different command that completes more quickly. Note: A common source of this error is if the command is interactive or requires user input (it is impossible to receive user input in the current environment, so the command will never complete)."),
        )

        tool_config_data = agent_config_data.get("tools", {})
        tool_config = ToolConfig(
            # filter=tool_config_data.get("filter", []),
            # bundles=tool_config_data.get("bundles", []),
            # propagate_env_variables=tool_config_data.get("propagate_env_variables", True),
            env_variables=tool_config_data.get("env_variables", {}),
            registry_variables=tool_config_data.get("registry_variables", {}),
            # submit_command=tool_config_data.get("submit_command", None),
            parse_function=tool_config_data.get("parse_function", None),
            enable_bash_tool=tool_config_data.get("enable_bash_tool", True),
            format_error_template=tool_config_data.get("format_error_template",
                                                   "Your output was not formatted correctly. Please make sure you follow the response format instructions."),
            # command_docs=tool_config_data.get("command_docs", {}),
            multi_line_command_endings=tool_config_data.get("multi_line_command_endings", {}),
            submit_command_end_name=tool_config_data.get("submit_command_end_name", None),
            reset_commands=tool_config_data.get("reset_commands", []),
            execution_timeout=tool_config_data.get("execution_timeout", 120),
            install_timeout=tool_config_data.get("install_timeout", 300),
            total_execution_timeout=tool_config_data.get("total_execution_timeout", 300),
            max_consecutive_execution_timeouts=tool_config_data.get("max_consecutive_execution_timeouts", 3),
        )

        default_agent_config = DefaultAgentConfig(
            name=agent_config_data.get("name", "main"),
            model=GenericAPIModelConfig(
                name=model_name,
            ),
            templates=template_config,
            history_processors=agent_config_data.get("history_processors", []),
            tools=tool_config,
            max_requeries=agent_config_data.get("max_requeries", 3),
            action_sampler=agent_config_data.get("action_sampler", None),
            type=agent_config_data.get("type", "default"),
        )

        deployment_config = VefaasDeploymentConfig(
            type="vefaas",
            image=image,
            command="curl -fsSL https://pjw-test-empty.tos-cn-beijing.volces.com/bin/tos_swe_rex.sh | bash -s -- {token}",
            timeout=300,
            instance_id=instance_id,
        )
    
        env_config = EnvironmentConfig(
            repo = PreExistingRepoConfig(
                repo_name = "testbed",
                base_commit= metadata.get("base_commit", None),
            )
        )
        env_config.deployment = deployment_config

        problem_statement = TextProblemStatement(
            id=instance_id,
            text=metadata.get("problem_statement", ""),
            extra_fields={
                "working_dir": "/testbed"
            }
        )

        run_output_dir = "/tmp/trajectories/" + model_name
        single_run_config = RunSingleConfig(
            env = env_config,
            agent = default_agent_config,
            problem_statement = problem_statement,
            output_dir=run_output_dir,
        )

        single_run = ReRunSingle.from_config(single_run_config)

        # replace chat model and agent.
        model = ChatModel(
            model=model_name,
            client=self.server_manager,
            tokenizer=self.tokenizer,
            temperature=temperature,
            top_p=top_p,
            max_model_len=max_model_len,
            max_parallel_calls=rollout.multi_turn.max_parallel_calls,
            tool_parser=rollout.multi_turn.format,
        )
        single_run.agent = ReSweAgent.from_config(default_agent_config, model=model)

        try:
            result = await asyncio.to_thread(single_run.run)
            output_dir = single_run.output_dir
            patch_output_dir = Path(output_dir) / instance_id
            patch_output_file = patch_output_dir / f"{instance_id}.pred"

            test_spec = make_test_spec(metadata)
            with open(patch_output_file,"r") as f:
                pred = json.load(f)

            vefaas_config = VefaasDeploymentConfig(
                type="vefaas",
                image=image,
                command="curl -fsSL https://pjw-test-empty.tos-cn-beijing.volces.com/bin/tos_swe_rex.sh | bash -s -- {token}",
                function_id="e666js9h",
                function_route="https://sd49dbp6fog2gt9o99rtg.apigateway-cn-beijing.volceapi.com",
                instance_id=instance_id,
                timeout=300,
            )
            deployment = VefaasDeployment.from_config(vefaas_config)

            # Run evaluation
            print("Starting evaluation...")

            eval_result = await run_eval(
                test_spec=test_spec,
                pred=pred,
                container=deployment,
                run_id=f"vefaas_test_{instance_id}",
                timeout=1800,  # 30 minutes timeout
            )
            print(eval_result)

            return self.convert_to_agent_output(result, eval_result)

        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

            output = AgentLoopOutput(
                prompt_ids=[self.tokenizer.pad_token_id],
                response_ids=[self.tokenizer.pad_token_id],
                response_mask=[0],
                reward_score=0,
                num_turns=0,
                metrics={},
            )
            return output
        finally:
            print("finish")
    
    def convert_to_agent_output(self, result: AgentRunResult, eval_result: dict) -> AgentLoopOutput:
        """Convert SWE-agent result to AgentLoopOutput.
        
        Args:
            result: The result from SWE-agent run
            
        Returns:
            AgentLoopOutput: The converted output
        """
        if result is None:
            output = AgentLoopOutput(
                prompt_ids=[self.tokenizer.pad_token_id],
                response_ids=[self.tokenizer.pad_token_id],
                response_mask=[0],
                reward_score=0,
                num_turns=0,
                metrics={},
            )
            return output
        
        trajectory = result.trajectory
        num_turns = len(trajectory)

        for i in range(len(trajectory) - 1, -1, -1):
            if trajectory[i]["extra_info"].get("raw") is not None:
                break
        last_trajectory = trajectory[i]
        assert "raw" in last_trajectory["extra_info"], "Last trajectory extra_info should have 'raw' key"
        raw_response = last_trajectory["extra_info"]["raw"]

        prompt_ids = raw_response.get("prompt_ids", [self.tokenizer.pad_token_id])
        response_mask = raw_response.get("response_mask", [0])

        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        response_length = self.config.actor_rollout_ref.rollout.response_length
        output = AgentLoopOutput(
            prompt_ids=prompt_ids[:prompt_length],
            response_ids=response_ids[:response_length],
            response_mask=response_mask[:response_length],
            reward_score=int(eval_result.get("resolved", False)),
            num_turns=num_turns,
            metrics={},
        )
        return output