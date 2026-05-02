#!/usr/bin/env bash
set -euo pipefail

# If this setup.bash is submitted alone, set REPO_URL to your public GitHub repo.
# Example: REPO_URL="https://github.com/your-user/gnr-638-project.git"
REPO_URL="${REPO_URL:-https://github.com/Rrp10157/GNR-638-Project-2.git}"
ENV_NAME="${ENV_NAME:-gnr_project_env}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL_DIR="${MODEL_DIR:-models/qwen-vl}"

ROOT_DIR="$(pwd)"

if [ ! -f "inference.py" ]; then
  if [ -z "$REPO_URL" ]; then
    echo "ERROR: inference.py not found and REPO_URL is empty."
    echo "Set REPO_URL inside setup.bash before submitting the single-file zip."
    exit 1
  fi
  echo "Cloning project repository from $REPO_URL"
  tmp_clone_dir="_repo_clone"
  rm -rf "$tmp_clone_dir"
  git clone "$REPO_URL" "$tmp_clone_dir"
  shopt -s dotglob
  cp -R "$tmp_clone_dir"/* "$ROOT_DIR"/
  shopt -u dotglob
  rm -rf "$tmp_clone_dir"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found on PATH."
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating existing conda environment: $ENV_NAME"
else
  echo "Creating conda environment: $ENV_NAME"
  conda create -y -n "$ENV_NAME" python=3.11 pip
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
python -m pip install -r requirements.txt

mkdir -p "$MODEL_DIR"
echo "Downloading model weights: $MODEL_ID -> $MODEL_DIR"
python - <<PYEOF
from huggingface_hub import snapshot_download
import os, glob, sys

snapshot_download(
    repo_id="$MODEL_ID",
    local_dir="$MODEL_DIR",
    ignore_patterns=["*.md", ".gitattributes", "onnx/*", "gguf/*"],
)

# Verify weight files are present
weights = glob.glob(os.path.join("$MODEL_DIR", "*.safetensors"))
if not weights:
    print("ERROR: Model download incomplete — no .safetensors files found.")
    sys.exit(1)
print(f"Model download verified: {len(weights)} weight file(s) found.")
PYEOF

echo "$MODEL_ID" > "$MODEL_DIR/model_id.txt"

python - <<'PY'
import torch
print("Python environment ready")
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Setup complete. Activate with: conda activate gnr_project_env"
