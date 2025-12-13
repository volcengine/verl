import os
import asyncio
import json
import yaml
from pathlib import Path
from typing import Any, Optional

from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig, TemplateConfig
from sweagent.environment.swe_env import SWEEnv, EnvironmentConfig
from sweagent.agent.models import GenericAPIModelConfig
from sweagent.agent.problem_statement import TextProblemStatement
from sweagent.tools.tools import ToolConfig
from sweagent.types import AgentInfo, AgentRunResult
from recipe.swe_agent.vefaas_deployment import VefaasDeployment, VefaasDeploymentConfig
from recipe.swe_agent.eval import run_eval
from swebench.harness.test_spec.test_spec import TestSpec
from sweagent.run.run_single import RunSingle, RunSingleConfig
from sweagent.environment.repo import PreExistingRepoConfig
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    LATEST,
    MAP_REPO_TO_EXT,
    MAP_REPO_VERSION_TO_SPECS,
    SWEbenchInstance,
)
from swebench.harness.test_spec.python import (
    make_repo_script_list_py,
    make_env_script_list_py,
    make_eval_script_list_py,
)


def make_test_spec(
        instance: SWEbenchInstance,
        namespace: Optional[str] = None,
        base_image_tag: str = LATEST,
        env_image_tag: str = LATEST,
        instance_image_tag: str = LATEST,
        arch: str = "x86_64",
) -> TestSpec:
    if isinstance(instance, TestSpec):
        return instance
    assert base_image_tag is not None, "base_image_tag cannot be None"
    assert env_image_tag is not None, "env_image_tag cannot be None"
    assert instance_image_tag is not None, "instance_image_tag cannot be None"
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance["repo"]
    version = instance.get("version")
    base_commit = instance["base_commit"]
    test_patch = instance["test_patch"]

    def _from_json_or_obj(key: str) -> Any:
        """If key points to string, load with json"""
        if key not in instance:
            # If P2P, F2P keys not found, it's a validation instance
            return []
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]

    pass_to_pass = _from_json_or_obj("PASS_TO_PASS")
    fail_to_pass = _from_json_or_obj("FAIL_TO_PASS")

    env_name = "testbed"
    repo_directory = f"/{env_name}"
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    docker_specs = specs.get("docker_specs", {})

    repo_script_list = make_repo_script_list_py(
        specs, repo, repo_directory, base_commit, env_name
    )
    env_script_list = make_env_script_list_py(instance, specs, env_name)
    eval_script_list = make_eval_script_list_py(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    )
    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language=MAP_REPO_TO_EXT[repo],
        docker_specs=docker_specs,
        namespace=namespace,
        base_image_tag=base_image_tag,
        env_image_tag=env_image_tag,
        instance_image_tag=instance_image_tag,
    )


def load_instance(instance_id: str, subnet: str = "verified"):
    def get_swe_bench_info(instance_id: str, subnet: str = "verified") -> dict:
        with open("swe-bench-image.json","r") as f:
            output_data = json.load(f)
        if subnet == "swe-bench":
            return output_data["swe-bench"].get(instance_id, None)
        elif subnet == "verified":
            return output_data["swe-bench-verified"].get(instance_id, None)
        else:
            raise ValueError(f"Unknown subnet: {subnet}")
    swe_bench_info = get_swe_bench_info(instance_id, subnet)
    if swe_bench_info:
        metadata = swe_bench_info['metadata']
        image = swe_bench_info['image']
        return metadata,image
    else:
        raise Exception(f"Instance {instance_id} not found in {subnet} dataset.")


def save_predictions(traj_dir: Path, instance_id: str, result: AgentRunResult):
    """Save predictions in a file readable by SWE-bench"""
    output_file = traj_dir / instance_id / (instance_id + ".pred")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    datum = {
        "model_name_or_path": traj_dir.name,
        "instance_id": instance_id,
        "model_patch": result.info.get("submission"),
    }
    output_file.write_text(json.dumps(datum))

def create_prediction_dict(instance_id, model_name, patch):
    """Create prediction dictionary in the format expected by run_eval."""
    return {
        "model_name_or_path": model_name,
        "instance_id": instance_id,
        "model_patch": patch,
        "prediction": patch  # Some versions use 'prediction' instead of 'model_patch'
    }

def get_model_name_from_config(agent_config_data):
    """Extract model name from agent config."""
    model_config = agent_config_data.get("model", {})
    return model_config.get("name", "unknown_model")

def main():
    instance_id = "django__django-15375"
    subnet="verified"
    metadata,image = load_instance(instance_id, subnet)



    with open("test.yaml","r") as f:
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
        command_cancelled_timeout_template=template_config_data.get("command_cancelled_timeout_template",
                                                                    "The command '{{command}}' was cancelled because it took more than {{timeout}} seconds. Please try a different command that completes more quickly. Note: A common source of this error is if the command is interactive or requires user input (it is impossible to receive user input in the current environment, so the command will never complete)."),
    )

    # Create tool config
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
    agent_model_config_data = agent_config_data.get("model", {})
    agent_model_config = GenericAPIModelConfig(
        name=agent_model_config_data.get("name", "Qwen/Qwen3-30B-A3B-Instruct"),
        per_instance_cost_limit=agent_model_config_data.get("per_instance_cost_limit", 3.0),
        total_cost_limit=agent_model_config_data.get("total_cost_limit", 0.0),
        per_instance_call_limit=agent_model_config_data.get("per_instance_call_limit", 0),
        temperature=agent_model_config_data.get("temperature", 0.0),
        top_p=agent_model_config_data.get("top_p", 1.0),
        api_base=agent_model_config_data.get("api_base", "http://localhost:8000"),
        api_version=agent_model_config_data.get("api_version", None),
        api_key=agent_model_config_data.get("api_key", None),
        stop=agent_model_config_data.get("stop", []),
        completion_kwargs=agent_model_config_data.get("completion_kwargs", {}),
        convert_system_to_user=agent_model_config_data.get("convert_system_to_user", False),
        delay=agent_model_config_data.get("delay", 0.0),
        fallbacks=agent_model_config_data.get("fallbacks", []),
        choose_api_key_by_thread=agent_model_config_data.get("choose_api_key_by_thread", True),
        max_input_tokens=agent_model_config_data.get("max_input_tokens", None),
        max_output_tokens=agent_model_config_data.get("max_output_tokens", None),
        litellm_model_registry=agent_model_config_data.get("litellm_model_registry", None),
        custom_tokenizer=agent_model_config_data.get("custom_tokenizer", None),
    )
    agent_config = DefaultAgentConfig(
        name=agent_config_data.get("name", "main"),
        templates=template_config,
        tools=tool_config,
        history_processors=agent_config_data.get("history_processors", []),
        model=agent_model_config,
        max_requeries=agent_config_data.get("max_requeries", 3),
        action_sampler=agent_config_data.get("action_sampler", None),
        type=agent_config_data.get("type", "default"),
    )

    vefaas_config = VefaasDeploymentConfig(
        type="vefaas",
        image=image,
        command="curl -fsSL https://pjw-test-empty.tos-cn-beijing.volces.com/bin/tos_swe_rex.sh | bash -s -- {token}",
    )

    # Create environment configuration
    env_config = EnvironmentConfig(
        repo = PreExistingRepoConfig(
            repo_name = "testbed",
            base_commit= metadata.get("base_commit", None),
        )
    )
    env_config.deployment = vefaas_config


    # Create SWE environment with Vefaas deployment
    problem_statement = TextProblemStatement(
        id = instance_id,
        text=metadata.get("problem_statement", ""),
        extra_fields={
            "working_dir": "/testbed"
        }
    )

    single_run_config = RunSingleConfig(
        env = env_config,
        agent = agent_config,
        problem_statement = problem_statement,
    )

    single_run = RunSingle.from_config(single_run_config)

    try:
        single_run.run()
        print("coding finish")
        output_dir = single_run.output_dir
        patch_output_dir = Path(output_dir) / instance_id
        patch_output_file = patch_output_dir / f"{instance_id}.pred"

        test_spec = make_test_spec(metadata)
        with open(patch_output_file,"r") as f:
            pred = json.load(f)
        #

        vefaas_config = VefaasDeploymentConfig(
            type="vefaas",
            image=image,
            command="curl -fsSL https://pjw-test-empty.tos-cn-beijing.volces.com/bin/tos_swe_rex.sh | bash -s -- {token}",
        )
        deployment = VefaasDeployment.from_config(vefaas_config)

        # # Run evaluation
        print("Starting evaluation...")

        eval_result = asyncio.run(run_eval(
            test_spec=test_spec,
            pred=pred,
            container=deployment,
            run_id=f"vefaas_test_{instance_id}",
            timeout=1800  # 30 minutes timeout
        ))

        print(eval_result)

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("finish")

if __name__ == "__main__":
    main()
    # evaluate()
