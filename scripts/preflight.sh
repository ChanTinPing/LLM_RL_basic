#!/usr/bin/env bash
set -euo pipefail

echo "[0/6] activate py_env & network"
source .py_env/bin/activate
source /etc/network_turbo || echo "no /etc/network_turbo (skip)"

echo "[1/6] python & pip"
python -V
python -c "import sys; print('pip:', __import__('pip').__version__)"

echo "[2/6] torch & cuda"
python - <<'PY'
import torch, json
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("cuda:", torch.version.cuda)
PY

echo "[3/6] nvidia-smi"
command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi not found"

echo "[4/6] disk space"
df -h .

echo "[5/6] huggingface connectivity (dns only)"
getent hosts huggingface.co >/dev/null && echo "hf dns ok" || echo "hf dns fail (not fatal)"

echo "[6/6] model smoke test (HF load)"
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "hyper-accel/tiny-random-gpt2"

print(f"loading tokenizer: {MODEL_ID}")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print("loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
print("loaded:", MODEL_ID)
PY

echo "preflight OK"
