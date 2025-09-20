#!/usr/bin/env bash
set -euo pipefail

echo "[0/7] activate py_env & network"
source .py_c128/bin/activate
source /etc/network_turbo

echo "[1/7] python & pip"
python -V
python -c "import sys; print('pip:', __import__('pip').__version__)"

echo "[2/7] torch & cuda & common libs"
python - <<'PY'
import os, json, platform
ok = True

def pfx(name): print(f"[check] {name}:")

# --- torch & cuda ---
try:
    import torch
    pfx("torch")
    print("  torch.__version__        :", torch.__version__)
    print("  python                   :", platform.python_version())
    print("  cuda_available           :", torch.cuda.is_available())
    print("  torch.version.cuda       :", getattr(torch.version, "cuda", None))
    if torch.cuda.is_available():
        print("  gpu_name                 :", torch.cuda.get_device_name(0))
        print("  capability               :", torch.cuda.get_device_capability(0))
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        y = x @ x.t()
        print("  matmul(smoke)            :", tuple(y.shape), str(y.dtype))
except Exception as e:
    ok = False
    print("  [FAIL] torch check ->", e)

# --- triton ---
try:
    import triton
    pfx("triton")
    print("  triton.__version__       :", getattr(triton, "__version__", "unknown"))
except Exception as e:
    print("  [WARN] triton not OK ->", e)

# --- flash-attn ---
try:
    import flash_attn
    pfx("flash-attn")
    fav = getattr(flash_attn, "__version__", None)
    print("  flash_attn.__version__   :", fav or "unknown")
    try:
        from flash_attn.layers.rotary import apply_rotary_emb
        print("  submodule(rotary)        : OK")
    except Exception as e:
        print("  submodule(rotary)        :", e)
except Exception as e:
    ok = False
    print("  [FAIL] flash-attn import ->", e)

# --- bitsandbytes ---
try:
    import bitsandbytes as bnb
    pfx("bitsandbytes")
    print("  bitsandbytes.__version__ :", getattr(bnb, "__version__", "unknown"))
    print("  env.BNB_CUDA_VERSION     :", os.getenv("BNB_CUDA_VERSION"))
except Exception as e:
    print("  [WARN] bitsandbytes not OK ->", e)

# --- verl ---
try:
    import verl
    pfx("verl")
    print("  verl.__version__         :", getattr(verl, "__version__", "installed"))
except Exception as e:
    ok = False
    print("  [FAIL] verl import ->", e)

# --- transformers / trl / peft ---
try:
    import transformers
    pfx("transformers")
    print("  transformers.__version__ :", getattr(transformers, "__version__", "unknown"))
except Exception as e:
    print("  [WARN] transformers not OK ->", e)

try:
    import trl
    pfx("trl")
    print("  trl.__version__          :", getattr(trl, "__version__", "unknown"))
except Exception as e:
    print("  [WARN] trl not OK ->", e)

try:
    import peft
    pfx("peft")
    print("  peft.__version__         :", getattr(peft, "__version__", "unknown"))
except Exception as e:
    print("  [WARN] peft not OK ->", e)

print("[result] step2 overall:", "OK" if ok else "HAS_FAILURES")
PY

echo "[3/7] nvidia-smi"
command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi not found"

echo "[4/7] disk space"
df -h .

echo "[5/7] huggingface connectivity (dns only)"
getent hosts huggingface.co >/dev/null && echo "hf dns ok" || echo "hf dns fail (not fatal)"

echo "[6/7] model smoke test (HF load)"
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

echo "[7/7] export pip freeze"
pip freeze > ./requirements.txt
echo "requirements exported to requirements.txt"

echo "preflight OK"
