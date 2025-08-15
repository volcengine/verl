# FlashAttention-3 on `verlai/verl:base-verl0.5-preview-cu128-cudnn9.8-torch2.7.1-fa2.8.0`

This guide shows two reliable ways to enable **FlashAttention-3 (FA3)** inside the provided CUDA 12.8 + PyTorch 2.7.x container:

- **Option A:** Install a **prebuilt wheel** (fastest).
- **Option B:** **Build from source** inside the container (for custom commits/arches).

It also includes quick checks and troubleshooting tips.

---

## 0) Pull & start the container

```bash
# Pull the image
docker pull verlai/verl:base-verl0.5-preview-cu128-cudnn9.8-torch2.7.1-fa2.8.0

# Run it with GPU access
docker run --rm -it --gpus all   --name verl-fa3   verlai/verl:base-verl0.5-preview-cu128-cudnn9.8-torch2.7.1-fa2.8.0 bash

# Inside the container, verify CUDA + GPU:
nvidia-smi
python3 - <<'PY'
import torch
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY
```

> The image already includes **PyTorch 2.7.1 (cu128)** and **FA2.8.0**. We’ll add **FA3** on top.

---

## 1) Option A — Install the **prebuilt FA3 wheel** (recommended)

**Why:** Zero toolchain hassle; consistent with CUDA 12.8 & Torch 2.7.x in this image.

```bash
# Inside the container
python3 -m pip install -U pip

# Install the wheel you published
python3 -m pip install   "https://github.com/CedricHwong/Prebuilt-Python-wheels-storage/releases/download/flash_attn_3-3.0.0b1%2Bcu128/flash_attn_3-3.0.0b1+cu128-cp39-abi3-linux_x86_64.whl"

# Sanity check
python3 - <<'PY'
import importlib, torch
m = importlib.import_module("flash_attn_3")
print("flash_attn_3 OK; torch", torch.__version__, "cuda", torch.version.cuda)
PY
```

If that prints without errors, FA3 is ready.

---

## 2) Option B — Build FA3 from source (for custom refs/arches)

Use the following script **inside** the container. It will:
- Ensure build deps are present
- Confirm Torch is **cu128**
- Clone **Dao-AILab/flash-attention**
- Build **FA3** wheel for **sm_90 (H100/H200)**

> If you need a specific tag/commit, set `FLASH_ATTENTION_REF` accordingly (e.g. `v3.0.0b1`).

```bash
#!/usr/bin/env bash
set -euo pipefail

FLASH_ATTENTION_REPO="${FLASH_ATTENTION_REPO:-https://github.com/Dao-AILab/flash-attention.git}"
FLASH_ATTENTION_REF="${FLASH_ATTENTION_REF:-main}"   # tag/commit acceptable
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.7.*}"
TORCH_CUDA_ARCH_LIST_VAL="${TORCH_CUDA_ARCH_LIST_VAL:-9.0}"  # H100/H200 = sm_90

# Limit parallel build threads to <= 8
if command -v nproc >/dev/null 2>&1; then CORES=$(nproc); else CORES=8; fi
if [ "${CORES}" -gt 8 ]; then MAX_JOBS_VAL=8; else MAX_JOBS_VAL="${CORES}"; fi

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

info(){ echo -e "\033[1;34m[INFO]\033[0m $*"; }
warn(){ echo -e "\033[1;33m[WARN]\033[0m $*"; }

# ---- Deps
if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
  info "Installing build deps via apt-get..."
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends     git build-essential ninja-build ca-certificates
  rm -rf /var/lib/apt/lists/*
fi
info "Installing python build deps via pip..."
python3 -m pip install -U pip setuptools wheel packaging ninja pytest

# ---- Torch cu128
TORCH_CUDA_VER="$(python3 -c 'import torch,sys; print(torch.version.cuda or "cpu")' 2>/dev/null || echo none)"
if [ "$TORCH_CUDA_VER" != "12.8" ]; then
  info "Installing PyTorch (${TORCH_VERSION}) from ${TORCH_INDEX_URL} ..."
  python3 -m pip install --index-url "$TORCH_INDEX_URL" "torch==${TORCH_VERSION}"
else
  info "Detected existing PyTorch (CUDA $TORCH_CUDA_VER): $(python3 -c 'import torch; print(torch.__version__)')"
fi

# ---- nvcc
if command -v nvcc >/dev/null 2>&1; then
  info "nvcc: $(nvcc --version | tail -n1)"
else
  warn "nvcc not found; ensure CUDA 12.8 toolchain exists."
fi

# ---- Build
WORKDIR="${PWD}"
SRC_DIR="${WORKDIR}/flash-attention"
if [ -d "$SRC_DIR/.git" ]; then
  info "Repo exists, fetching..."
  git -C "$SRC_DIR" fetch --all
else
  info "Cloning flash-attention..."
  git clone --recursive "$FLASH_ATTENTION_REPO" "$SRC_DIR"
fi
info "Checkout ${FLASH_ATTENTION_REF} & submodules..."
git -C "$SRC_DIR" checkout "$FLASH_ATTENTION_REF"
git -C "$SRC_DIR" submodule update --init --recursive

cd "$SRC_DIR/hopper"
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST_VAL"  # e.g., 9.0 for H100/H200, 8.0 for A100
export MAX_JOBS="$MAX_JOBS_VAL"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE
export FLASH_ATTN_LOCAL_VERSION=cu128
export FLASH_ATTN_CUDA_ARCHS="90"  # sm_90; change to 80 if targeting A100

info "Building wheel (ARCH=${TORCH_CUDA_ARCH_LIST}, MAX_JOBS=${MAX_JOBS})..."
python3 setup.py bdist_wheel

OUT_DIR="${WORKDIR}/dist_fa3"
mkdir -p "$OUT_DIR"
cp -v dist/*.whl "$OUT_DIR"/
info "Done. Wheels are at: $OUT_DIR"
ls -lh "$OUT_DIR"

# Install the freshly built wheel (optional)
pip install "$OUT_DIR"/flash_attn_3-*.whl

# Final check
python3 - <<'PY'
import importlib
m = importlib.import_module("flash_attn_3")
print("flash_attn_3 import OK from source build")
PY
```

### Building for other GPUs
- **H100/H200:** `TORCH_CUDA_ARCH_LIST_VAL=9.0` and `FLASH_ATTN_CUDA_ARCHS="90"`
- **A100:** `TORCH_CUDA_ARCH_LIST_VAL=8.0` and `FLASH_ATTN_CUDA_ARCHS="80"`

> You can pass these as env vars before running the script:
>
> ```bash
> TORCH_CUDA_ARCH_LIST_VAL=8.0 FLASH_ATTN_CUDA_ARCHS=80 ./build_fa3.sh
> ```

---

## 3) Minimal runtime check

```bash
python3 - <<'PY'
import torch, importlib
fa3 = importlib.import_module("flash_attn_3")
print("FA3 imported. GPU:", torch.cuda.get_device_name(0))
# Optional: allocate a small CUDA tensor so we confirm runtime is alive
x = torch.randn(1, 1, 128, 128, device="cuda", dtype=torch.float16)
print("CUDA OK:", x.shape, x.dtype, x.device)
PY
```

If you use FA3 functions in your training loop, verify a short forward pass on a single batch to confirm kernels load without errors.

---

## 4) Tips & Notes

- **Keep Torch & CUDA aligned:** This image is **Torch 2.7.1 + cu128**. Make sure any wheel you install matches **CUDA 12.8** and **your GPU arch**.
- **ABI/C++ standard:** The script forces **CXX11 ABI** to match common PyTorch builds.
- **Reproducibility:** Pin a concrete FA commit/tag via `FLASH_ATTENTION_REF` (e.g., `v3.0.0b1`) for deterministic builds across machines.
- **FA2 coexistence:** The base image has **FA2.8.0**; installing FA3 adds a separate package (`flash_attn_3`). Your code should import the correct package explicitly.

---

## 5) Troubleshooting

**`ImportError: ... undefined symbol ...` (ABI/CUDA mismatch)**  
- Ensure `torch.version.cuda == "12.8"` and the FA3 wheel is **+cu128**.  
- Rebuild FA3 with the same compiler/CUDA as the runtime.

**`no kernel image is available`**  
- Arch mismatch. Rebuild with correct `TORCH_CUDA_ARCH_LIST_VAL` and `FLASH_ATTN_CUDA_ARCHS` (e.g., `9.0` for H100/H200, `8.0` for A100).

**`nvcc: command not found` during source build**  
- Install CUDA 12.8 toolchain inside the container or use **Option A** (prebuilt wheel).

**Unit tests**  
- After building from source, you can run lightweight checks:
  ```bash
  cd /path/to/flash-attention/hopper
  pytest -q -k "flash_attn"  # adjust patterns as needed
  ```

---

## 6) Quick “copy-paste” summary

```bash
# A) Prebuilt wheel (fastest)
docker run --rm -it --gpus all verlai/verl:base-verl0.5-preview-cu128-cudnn9.8-torch2.7.1-fa2.8.0 bash
python3 -m pip install -U pip
python3 -m pip install "https://github.com/CedricHwong/Prebuilt-Python-wheels-storage/releases/download/flash_attn_3-3.0.0b1%2Bcu128/flash_attn_3-3.0.0b1+cu128-cp39-abi3-linux_x86_64.whl"
python3 -c "import flash_attn_3, torch; print('OK', torch.__version__, torch.version.cuda)"

# B) Build from source (custom)
# (inside container)
bash build_fa3.sh   # Use the script above
```
