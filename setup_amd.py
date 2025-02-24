# # Copyright 2024 Bytedance Ltd. and/or its affiliates
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# # setup.py is the fallback installation script when pyproject.toml does not work
# from setuptools import setup, find_packages
# from setuptools.command.install import install 
# import os
# import sys
# from subprocess import check_call
# import tempfile
# import shutil

# version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

# with open(os.path.join(version_folder, 'verl/version/version')) as f:
#     __version__ = f.read().strip()


# # with open('requirements.txt') as f:
# with open('requirements_amd.txt') as f:
#     required = f.read().splitlines()
#     install_requires = [item.strip() for item in required if item.strip()[0] != '#']


# class CustomInstall(install):
#     def run(self):
#         print("Checking vllm installation...")
#         try:
#             import vllm
#             if hasattr(vllm, '__version__') and vllm.__version__ == '0.6.3+rocm624':
#                 print("Correct vllm version is already installed")
#                 return
#             else:
#                 print(f"Found incompatible vllm version: {getattr(vllm, '__version__', 'unknown')}")
#                 print("Removing existing vllm installation...")
#                 check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'vllm'])
#         except ImportError:
#             print("vllm not found, proceeding with installation")
        
#         # Install vllm
#         try:
#             print("Installing vllm from source...")
#             # Create temporary directory for cloning and building vllm
#             with tempfile.TemporaryDirectory() as temp_dir:
#                 # Clone vllm repository
#                 check_call(['git', 'clone', 'https://github.com/vllm-project/vllm.git', temp_dir])
                
#                 # Checkout specific version
#                 check_call(['git', 'checkout', 'v0.6.3'], cwd=temp_dir)
                
#                 # Set ROCm architecture environment variable
#                 os.environ['PYTORCH_ROCM_ARCH'] = 'gfx90a;gfx942'
                
#                 # Install vllm in development mode
#                 check_call([sys.executable, 'setup.py', 'develop'], cwd=temp_dir)
                
#             print("vllm installation completed")
#         except Exception as e:
#             print(f"Warning: Failed to install vllm: {str(e)}")

#         # Install packages that require --no-deps
#         try:
#             with open('requirements_amd_no_deps.txt') as f:
#                 no_deps_packages = [
#                     item.strip() 
#                     for item in f.read().splitlines() 
#                     if item.strip() and not item.strip().startswith('#')
#                 ]
            
#             for package in no_deps_packages:
#                 check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', package])
#         except FileNotFoundError:
#             print("Warning: requirements_amd_no_deps.txt not found, skipping no-deps installations")
        
#         # Run original install process
#         super().run()


# extras_require = {
#     'test': ['pytest', 'yapf']
# }

# from pathlib import Path
# this_directory = Path(__file__).parent
# long_description = (this_directory / "README.md").read_text()

# setup(
#     name='verl',
#     version=__version__,
#     package_dir={'': '.'},
#     packages=find_packages(where='.'),
#     url='https://github.com/volcengine/verl',
#     license='Apache 2.0',
#     author='Bytedance - Seed - MLSys',
#     author_email='zhangchi.usc1992@bytedance.com, gmsheng@connect.hku.hk',
#     description='veRL: Volcano Engine Reinforcement Learning for LLM',
#     install_requires=install_requires,
#     extras_require=extras_require,
#     package_data={'': ['version/*'],
#                   'verl': ['trainer/config/*.yaml'],},
#     include_package_data=True,
#     long_description=long_description,
#     long_description_content_type='text/markdown',
#     cmdclass={
#         'install': CustomInstall,  # self-define install pkg not no-dep
#     }
# )











# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from setuptools import setup, find_packages
from setuptools.command.install import install 
import os
import sys
from subprocess import check_call
import tempfile
import shutil
from pathlib import Path

# Get version
version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))
with open(os.path.join(version_folder, 'verl/version/version')) as f:
    __version__ = f.read().strip()

# # Read requirements
# with open('requirements_amd.txt') as f:
#     required = f.read().splitlines()
#     install_requires = [item.strip() for item in required if item.strip() and not item.strip().startswith('#')]

class CustomInstall(install):
    def run(self):
        1. Install required dependencies first
        print("Installing dependencies from requirements_amd.txt...")
        try:
            with open('requirements_amd.txt') as f:
                required = [
                    item.strip() 
                    for item in f.read().splitlines() 
                    if item.strip() and not item.strip().startswith('#')
                ]
            for package in required:
                try:
                    check_call([sys.executable, '-m', 'pip', 'install', package])
                except Exception as e:
                    print(f"Warning: Failed to install {package}: {str(e)}")
        except FileNotFoundError:
            print("Warning: requirements_amd.txt not found")

        # 2. Install packages with --no-deps flag
        print("Installing no-deps packages from requirements_amd_no_deps.txt...")
        try:
            with open('requirements_amd_no_deps.txt') as f:
                no_deps_packages = [
                    item.strip() 
                    for item in f.read().splitlines() 
                    if item.strip() and not item.strip().startswith('#')
                ]
            for package in no_deps_packages:
                try:
                    check_call([sys.executable, '-m', 'pip', 'install', package, '--no-deps'])
                except Exception as e:
                    print(f"Warning: Failed to install {package}: {str(e)}")
        except FileNotFoundError:
            print("Warning: requirements_amd_no_deps.txt not found")

        
        # 3. Check and install vllm
        print("Checking vllm installation...")
        try:
            import vllm
            print(f"Found existing vllm version: {getattr(vllm, '__version__', 'unknown')}")
            print("Removing existing vllm installation...")
            check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', 'vllm'])
        except ImportError:
            print("vllm not found, proceeding with installation")
        
        # Install vllm from source
        print("Installing vllm from source...")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Clone vllm repository
                check_call(['git', 'clone', 'https://github.com/vllm-project/vllm.git', temp_dir])
                
                # Checkout specific version
                check_call(['git', 'checkout', 'v0.6.3'], cwd=temp_dir)
                
                # Set ROCm architecture environment variable
                os.environ['PYTORCH_ROCM_ARCH'] = 'gfx90a;gfx942'
                
                # Install vllm in development mode
                check_call([sys.executable, 'setup.py', 'develop'], cwd=temp_dir)
                
            print("vllm installation completed successfully")
        except Exception as e:
            print(f"Warning: Failed to install vllm: {str(e)}")
            print("Please try to install vllm manually following the instructions at:")
            print("https://docs.vllm.ai/en/v0.6.3/getting_started/amd-installation.html")

        4. Run original install process
        super().run()

# Extra requirements for testing
# extras_require = {
#     'test': ['pytest', 'yapf']
# }
TEST_REQUIRES = ['pytest', 'yapf', 'py-spy']
PRIME_REQUIRES = ['pyext']
# GPU_REQUIRES = ['liger-kernel', 'flash-attn']
GPU_REQUIRES = ['liger-kernel']

extras_require = {
  'test': TEST_REQUIRES,
  'prime': PRIME_REQUIRES,
  'gpu': GPU_REQUIRES,
}

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Setup configuration
setup(
    name='verl',
    version=__version__,
    package_dir={'': '.'},
    packages=find_packages(where='.'),
    url='https://github.com/volcengine/verl',
    license='Apache 2.0',
    author='Bytedance - Seed - MLSys',
    author_email='zhangchi.usc1992@bytedance.com, gmsheng@connect.hku.hk',
    description='veRL: Volcano Engine Reinforcement Learning for LLM',
    # install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        '': ['version/*'],
        'verl': ['trainer/config/*.yaml'],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown',
    cmdclass={
        'install': CustomInstall,
    }
)