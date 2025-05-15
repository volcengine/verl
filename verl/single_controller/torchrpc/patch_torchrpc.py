import torch.distributed.rpc as rpc
import os

module_path = os.path.dirname(rpc.__file__)
file_to_patch = os.path.join(module_path, "internal.py")
print("Patching", file_to_patch)
with open(file_to_patch, 'r', encoding='utf-8') as f:
    lines = f.readlines()
modified = False
new_lines = []
for line in lines:
    stripped_line = line.strip()
    if not modified and stripped_line == 'import pickle':
        new_lines.append('import dill as pickle\n')
        modified = True
    else:
        new_lines.append(line)
if modified:
    with open(file_to_patch, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    exit(0)
else:
    raise Exception("Patch Point not Found!")
