import json
import os

from swerex.deployment.abstract import AbstractDeployment
from pathlib import Path, PurePosixPath
import traceback
from swerex.runtime.abstract import (
    AbstractRuntime,
    Command,
    UploadRequest,
)
from swebench.harness.docker_build import (
    close_logger,
    setup_logger,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_PATCH,
    DOCKER_WORKDIR,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_INSTANCE,
    LOG_TEST_OUTPUT,
)

LATEST = "latest"

GIT_APPLY_CMDS = [
    ["git", "apply", "--verbose"],
    ["git", "apply", "--verbose", "--reject"],
    ["patch", "--batch", "--fuzz=5", "-p1", "-i"],
]

from swebench.harness.utils import (
    EvaluationError,
)
from swebench.harness.test_spec.test_spec import TestSpec
import swerex.exceptions


async def copy_to_container(container: AbstractRuntime, src: Path, dst: Path):
    """
    Copy a file from local to a docker container

    Args:
        container (Container): Docker container to copy to
        src (Path): Source file path
        dst (Path): Destination file path in the container
    """
    # Make directory if necessary
    await container.execute(Command(command=["mkdir", "-p", str(dst.parent)]))

    # Upload file to container
    resp = await container.upload(UploadRequest(source_path=str(src), target_path=str(dst)))
    print(resp)




async def run_eval(
    test_spec: TestSpec,
    pred: dict,
    container: AbstractDeployment,
    run_id: str,
    timeout: int | None = None,
    rewrite_reports: bool = False
):

    EVALUATION_DIR = os.getenv("EVALUATION_DIR", os.path.join(os.path.dirname(__file__), 'sweb/eval'))
    instance_id = test_spec.instance_id
    model_name_or_path = pred.get(KEY_MODEL, "None").replace("/", "__")
    log_dir = Path(EVALUATION_DIR) / run_id / model_name_or_path / instance_id

    # Set up report file
    report_path = log_dir / LOG_REPORT
    if rewrite_reports:
        test_output_path = log_dir / LOG_TEST_OUTPUT
        if not test_output_path.exists():
            raise ValueError(f"Test output file {test_output_path} does not exist")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return {
            "completed": True,
            "resolved": report[instance_id]["resolved"],
        }
    if report_path.exists():
        report = json.loads(report_path.read_text())
        return {
            "completed": True,
            "resolved": report[instance_id]["resolved"],
        }

    # Set up logger
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_INSTANCE
    logger = setup_logger(instance_id, log_file)
    print(f"eval log file: {log_file}")

    # Run the instance
    eval_completed = False
    report = {}

    try:
        await container.start()
        # Copy model prediction as patch file to container
        patch_file = Path(log_dir / "patch.diff")
        patch_file.write_text(pred[KEY_PREDICTION] or "")
        print(f"pred content: {pred}")
        print(f"patch file content: {patch_file.read_text()}")
        logger.info(
            f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container..."
        )
        await copy_to_container(container.runtime, patch_file, PurePosixPath(DOCKER_PATCH))
        # Attempt to apply patch to container (TODO: FIX THIS)

        print(f"copy_to_container patch.diff successfully, instance id: {instance_id}")

        applied_patch = False
        for git_apply_cmd in GIT_APPLY_CMDS:
            val = await container.runtime.execute(
                Command(
                    command=git_apply_cmd + [str(DOCKER_PATCH)],
                    cwd=DOCKER_WORKDIR,
            ))
            print(f"git apply cmd: {git_apply_cmd}, return val: {val}, instance id: {instance_id}")
            if val.exit_code == 0:
                logger.info(f"{APPLY_PATCH_PASS}:\n{val.stdout}")
                applied_patch = True
                break
            else:
                logger.info(f"Failed to apply patch to container: {git_apply_cmd}")
        if not applied_patch:
            logger.info(f"{APPLY_PATCH_FAIL}:\n{val.stdout}")
            raise EvaluationError(
                instance_id,
                f"{APPLY_PATCH_FAIL}:\n{val.stdout}",
                logger,
            )

        # Get git diff before running eval script
        val = await container.runtime.execute(
            Command(command=["git", "-c", "core.fileMode=false", "diff"], cwd=DOCKER_WORKDIR),
        )

        git_diff_output_before = val.stdout.strip()
        logger.info(f"Git diff before:\n{git_diff_output_before}")
        print(f"git diff before: {git_diff_output_before}")

        print(f"test spec: {test_spec}")
        eval_file = Path(log_dir / "eval.sh")
        eval_file.write_text(test_spec.eval_script)
        logger.info(
            f"Eval script for {instance_id} written to {eval_file}; copying to container..."
        )
        await copy_to_container(container.runtime, eval_file, PurePosixPath("/eval.sh"))
        print(f"copy_to_container eval.sh successfully, instance id: {instance_id}")

        # Run eval script, write output to logs
        test_output_path = log_dir / LOG_TEST_OUTPUT
        with open(test_output_path, "w") as f:
            try:
                val = await container.runtime.execute(
                    Command(command=["/bin/bash", "/eval.sh"],
                            timeout=timeout),

                )
                print(f"eval.sh run return val: {val}, instance id: {instance_id}")
            except swerex.exceptions.CommandTimeoutError:
                logger.info(f"Test timed out after {timeout} seconds.")
                raise EvaluationError(
                    instance_id,
                    f"Test timed out after {timeout} seconds.",
                    logger,
                )
            finally:
                f.write(val.stdout)

        # Get git diff after running eval script (ignore permission changes)
        val = await  container.runtime.execute(
            Command(command=["git", "-c", "core.fileMode=false", "diff"], cwd=DOCKER_WORKDIR),
        )
        git_diff_output_after = val.stdout.strip()
        print(f"git diff after: {git_diff_output_after}")


        # Check if git diff changed after running eval script
        logger.info(f"Git diff after:\n{git_diff_output_after}")
        if git_diff_output_after != git_diff_output_before:
            logger.info("Git diff changed after running eval script")

        # Get report from test output
        logger.info(f"Grading answer for {instance_id}...")
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        logger.info(
            f"report: {report}\n"
            f"Result for {instance_id}: resolved: {report[instance_id]['resolved']}"
        )

        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        eval_completed = True
    except Exception as e:
        error_msg = (
            f"Error in evaluating model for {instance_id}: {e}\n"
            f"{traceback.format_exc()}\n"
            f"Check ({logger.log_file}) for more information."
        )
        logger.error(error_msg)
    finally:
        await container.runtime.close()
        close_logger(logger)
        return {
            "completed": eval_completed,
            "resolved": report.get(instance_id, {}).get("resolved", False),
        }



