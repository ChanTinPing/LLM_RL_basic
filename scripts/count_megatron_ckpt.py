# sum_ckpt_sizes.py
from torch.distributed.checkpoint import FileSystemReader

# === 修改这里为你的 dist_ckpt 目录 ===
CKPT_DIR = "weights/grpo-qwen3-1p7b/global_step_1/actor/dist_ckpt"

reader = FileSystemReader(CKPT_DIR)
meta = reader.read_metadata()
sd = meta.state_dict_metadata  # name -> (TensorStorageMetadata | BytesStorageMetadata | ...)

# ---- 工具函数 ----
DTYPE_BYTES = {
    "TORCH.BFLOAT16": 2,
    "TORCH.FLOAT16":  2, "TORCH.HALF": 2,
    "TORCH.FLOAT32":  4, "TORCH.FLOAT": 4,
    "TORCH.UINT8":    1, "TORCH.INT8":  1,
}

def dtype_nbytes(dt_obj):
    s = str(dt_obj).upper()
    for k, v in DTYPE_BYTES.items():
        if k in s:
            return v
    return None

def prod(sz):
    n = 1
    for x in sz:
        n *= int(x)
    return n

def tensor_nbytes(info):
    """返回条目字节数（尽量准确）。优先用 properties/size×dtype，其次用 length/num_bytes。无则返回 0。"""
    # A: TensorStorageMetadata
    if hasattr(info, "properties") and hasattr(info, "size"):
        bpe = dtype_nbytes(info.properties.dtype)
        if bpe is not None:
            return prod(info.size) * bpe
    # B: BytesStorageMetadata（rng 等）
    for fld in ("length", "num_bytes", "total_size"):
        if hasattr(info, fld):
            val = getattr(info, fld)
            if val is not None:
                return int(val)
    # C: 尝试 chunks 的 sizes（没有 dtype 就不估）
    if hasattr(info, "chunks") and info.chunks:
        ch0 = info.chunks[0]
        if hasattr(ch0, "sizes"):
            # 不知道 dtype 时，不做盲估，返回 0 更稳
            return 0
    return 0

# ---- 统计桶 ----
b_exp_avg = 0
b_exp_avg_sq = 0
b_fp32_param = 0
b_decoder = 0   # 仅模型参数，不包含 optimizer
b_embedding = 0        # 仅模型参数，不包含 optimizer
b_total = 0

for name, info in sd.items():
    nb = tensor_nbytes(info)
    if nb <= 0:
        continue
    b_total += nb

    # 优化器三类（按前缀精确分类）
    if name.startswith("optimizer.state.exp_avg."):
        b_exp_avg += nb
        continue
    if name.startswith("optimizer.state.exp_avg_sq."):
        b_exp_avg_sq += nb
        continue
    if name.startswith("optimizer.state.fp32_param."):
        b_fp32_param += nb
        continue

    # 下面是“模型参数”两类：decoder.layers.*, embedding.*（排除 optimizer 已经 continue 的）
    if name.startswith("decoder."):
        b_decoder += nb
    elif name.startswith("embedding."):
        b_embedding += nb

def to_gb(x): return x / 1e9

print("=== Size summary ===")
print(f"optimizer.state.exp_avg     : {to_gb(b_exp_avg):.3f} GB")
print(f"optimizer.state.exp_avg_sq  : {to_gb(b_exp_avg_sq):.3f} GB")
print(f"optimizer.state.fp32_param  : {to_gb(b_fp32_param):.3f} GB")
print(f"decoder     (model only)    : {to_gb(b_decoder):.3f} GB")
print(f"embedding   (model only)    : {to_gb(b_embedding):.3f} GB")
print("--------------------------------------")
print(f"Grand total (all entries)   : {to_gb(b_total):.3f} GB")

b_others = b_total - (b_exp_avg + b_exp_avg_sq + b_fp32_param + b_decoder + b_embedding)
print(f"others (uncategorized)      : {to_gb(b_others):.3f} GB")
