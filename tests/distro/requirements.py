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

import ast
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def extract_install_requires(setup_py_path):
    """Extracts the install_requires list from setup.py using AST parsing."""
    with open(setup_py_path, "r") as f:
        tree = ast.parse(f.read())

    # Locate the setup() function call
    setup_call = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup"):
            setup_call = node
            break

    if not setup_call:
        raise ValueError("setup() call not found in setup.py")

    # Extract the install_requires keyword argument
    install_requires = None
    for keyword in setup_call.keywords:
        if keyword.arg == "install_requires":
            install_requires = keyword.value
            break

    if not install_requires:
        raise ValueError("install_requires not specified in setup() call")

    # Handle cases where install_requires is a variable or a direct list
    if isinstance(install_requires, ast.Name):
        var_name = install_requires.id
        requires = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        if isinstance(node.value, ast.List):
                            for element in node.value.elts:
                                if isinstance(element, ast.Constant):
                                    requires.append(element.value)
                            return requires
                        else:
                            raise ValueError(f"install_requires references non-list variable {var_name}")
        raise ValueError(f"Variable {var_name} not found in setup.py")
    elif isinstance(install_requires, ast.List):
        return [element.value for element in install_requires.elts if isinstance(element, ast.Constant)]
    else:
        raise ValueError("install_requires must be a list or variable referencing a list")


def test_dependencies_consistent():
    # Paths to the project root (adjust if test script is in a subdirectory)
    project_root = Path(__file__).parent.parent.parent

    # Extract dependencies from setup.py
    setup_deps = extract_install_requires(project_root / "setup.py")

    # Extract dependencies from pyproject.toml
    pyproject_path = project_root / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    toml_deps = pyproject_data["project"]["dependencies"]

    # Assert equality to ensure consistency
    assert setup_deps == toml_deps, ("Please make sure dependencies in setup.py and pyproject.toml matches.\n"
                                     f"setup.py: {setup_deps}\n"
                                     f"pyproject.toml: {toml_deps}")
