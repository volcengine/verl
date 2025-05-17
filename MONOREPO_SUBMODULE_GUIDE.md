# Verl Monorepo - Submodule Setup Guide

This document describes the structure of the Verl Monorepo, which uses Git Submodules to manage its core dependencies (Megatron-LM, SGLang, Pai-Megatron-Patch), and how to set it up for development.

## 1. Goal

*   Manage core dependencies (SGLang, Megatron-LM, Pai-Megatron-Patch) as Git Submodules within the `verl` main repository.
*   Allow `verl` to use specific versions of these dependencies, pinned to particular commits.
*   Provide a clear structure for offline code access (once submodules are initialized) and for understanding inter-dependencies, especially for adapting `verl` to support DeepSeekV3 using `Pai-Megatron-Patch`.

## 2. Monorepo Structure Overview (with Submodules)

-   `verl/` (Main Git Repository):
    -   `.gitmodules`: Defines all submodules.
    -   `third_party/`:
        -   `sglang_for_verl/` (Submodule): SGLang `v0.4.4` (commit `6aaeb84...`), for current `verl` dependency.
        -   `sglang_latest/` (Submodule): SGLang latest (main branch), for reference or future use.
        -   `Megatron-LM_for_verl/` (Submodule): NVIDIA Megatron-LM `core_v0.12.0rc3` (commit `408eb718...`), as a reference baseline for `verl`'s current Megatron adaptation.
        -   `pai-megatron-patch/` (Submodule): Alibaba's Pai-Megatron-Patch (main branch).
            -   `Megatron-LM-250328/` (Nested Submodule, managed by `pai-megatron-patch`): NVIDIA Megatron-LM (`commit 6ba97dd...`). This is the target Megatron version for DSv3 support.
            -   `examples/`, `toolkits/`, `megatron_patch/` etc.
            -   Its `.gitmodules` file is configured to only manage `Megatron-LM-250328`.
    -   `requirements.txt`: Main project dependencies, points to `-e ./third_party/pai-megatron-patch/Megatron-LM-250328`.
    -   `requirements_sglang.txt`: SGLang dependencies, points to `-e ./third_party/sglang_for_verl`.
    -   `MONOREPO_SUBMODULE_GUIDE.md`: This document.
    -   `AGENT_TASK_CONTEXT.md`: Describes project background and future work (DSv3 migration).

## 3. Setup Instructions

### 3.1 Clone the Main `verl` Repository
```bash
git clone --recurse-submodules <verl_project_git_url> verl
cd verl
```
The `--recurse-submodules` flag will attempt to initialize all defined submodules, including nested ones like `pai-megatron-patch/Megatron-LM-250328`.

If you have already cloned `verl` without this flag, navigate to the `verl` directory and run:
```bash
git submodule update --init --recursive
```

### 3.2 Verify Submodule Status
After cloning/initialization, you can check the status of submodules:
```bash
git submodule status --recursive
```
This should show all submodules checked out to their designated commits.

### 3.3 Python Environment and Dependencies
1.  **Create and activate a Python virtual environment** (recommended). E.g., Python 3.8+.
2.  **Install dependencies**:
    The `requirements.txt` and `requirements_sglang.txt` files are configured to install the Megatron-LM and SGLang versions from the local submodule paths in editable mode.
    ```bash
    pip install -r requirements.txt
    pip install -r requirements_sglang.txt
    # If verl itself is a package:
    # pip install -e .
    ```
    Ensure all other Python package dependencies listed in `MONOREPO_SUBMODULE_GUIDE.md` (Section 4 of previous versions of this guide, or from `AGENT_TASK_CONTEXT.md`) are also installed. For a fully offline setup, these would need to be downloaded as wheels first.

## 4. Working with Submodules

*   **Updating Submodules**:
    *   To update a specific submodule to its latest version (from its own remote):
        ```bash
        cd third_party/<submodule_name>
        git pull origin <branch_name>
        cd ../..
        git add third_party/<submodule_name>
        git commit -m "Update <submodule_name> to latest"
        ```
    *   To update all submodules to the commit registered in the main `verl` project:
        ```bash
        git submodule update --remote --recursive
        ```
*   **Making changes within a submodule**: If you make changes directly within a submodule's directory, you'll need to commit and push those changes within that submodule's own repository first, then commit the submodule's new commit hash in the main `verl` repository.

## 5. Next Steps (Development)
Refer to `AGENT_TASK_CONTEXT.md` for the plan regarding `verl` code adaptation to use the target Megatron-LM version (`pai-megatron-patch/Megatron-LM-250328`) for DeepSeekV3 support.

--- 