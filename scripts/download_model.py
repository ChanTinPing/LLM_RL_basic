#!/usr/bin/env python
import os, sys, shutil
from pathlib import Path

def check_pkgs():
    # 1) 必要包与版本
    try:
        import huggingface_hub, transformers, safetensors
        print("[ok] huggingface_hub:", huggingface_hub.__version__)
        print("[ok] transformers   :", transformers.__version__)
        print("[ok] safetensors    :", safetensors.__version__)
    except Exception as e:
        print("[err] 依赖缺失：", e)
        print("    pip install -U huggingface_hub transformers safetensors")
        sys.exit(1)

def print_env():
    # 2) 关键环境变量（用于控制联网/加速）
    for k in ["HF_HUB_ENABLE_HF_TRANSFER", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
              "HUGGINGFACE_HUB_CACHE", "HTTPS_PROXY", "HTTP_PROXY"]:
        v = os.environ.get(k)
        if v:
            print(f"[env] {k}={v}")

def expect_files_ok(model_dir: str) -> bool:
    # 3) 关键文件自检（不同模型文件名略有差异，尽量通用）
    must_have_any = [
        "config.json",
        "model.safetensors.index.json",  # 分片索引（若是单文件模型，可能没有）
        "tokenizer.json",                 # 有的模型用 sentencepiece: tokenizer.model
        "tokenizer.model",
        "special_tokens_map.json",
        "generation_config.json",
    ]
    p = Path(model_dir)
    if not p.exists():
        print("[err] 模型目录不存在：", p)
        return False

    missing = []
    for f in must_have_any:
        if not (p / f).exists():
            missing.append(f)
    # 只要 tokenizer.json 或 tokenizer.model 其一存在即可
    if "tokenizer.json" in missing and "tokenizer.model" in missing:
        pass  # 两个都缺就算缺
    else:
        # 只缺其一不算缺
        missing = [m for m in missing if m not in ("tokenizer.json", "tokenizer.model")]

    if missing:
        print("[warn] 有可能缺少（不一定都必须）：", ", ".join(missing))
    # 至少要有 safetensors 分片或单文件
    has_weights = any(f.name.endswith(".safetensors") for f in p.glob("*"))
    if not has_weights:
        print("[err] 未检测到 *.safetensors 权重文件")
        return False
    return True

def human_size(num_bytes: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f}PB"

def dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except FileNotFoundError:
                pass
    return total

def main():
    check_pkgs()
    print_env()

    from huggingface_hub import snapshot_download

    repo_id = "Qwen/Qwen3-1.7B"         # 模型仓库
    local   = "weights/Qwen3-1p7B"      # 本地保存目录

    os.makedirs(local, exist_ok=True)
    print(f"[run] snapshot_download(repo_id='{repo_id}', local_dir='{local}')")
    p = snapshot_download(
        repo_id=repo_id,
        local_dir=local,
        resume_download=True,           # 断点续传
        local_dir_use_symlinks=False,   # 新版会忽略 symlinks，但写上也无妨
        max_workers=8,                  # 并发下载，视带宽/IO 调整
        # force_download=True,          # 如需强制重下可打开
    )
    print("[ok] downloaded_to:", p)

    # 体积与若干文件自检
    size = human_size(dir_size(local))
    print(f"[info] 目录大小: {size}")
    ok = expect_files_ok(local)
    if ok:
        print("[ok] 基本文件齐全，可离线加载")
    else:
        print("[warn] 建议核对上面的提示，确认文件是否完整")

    # 简单列出最重的几个文件，帮助你确认分片
    try:
        from glob import glob
        shards = sorted(glob(os.path.join(local, "*.safetensors")))
        largest = sorted(shards, key=lambda f: os.path.getsize(f), reverse=True)[:5]
        if largest:
            print("[info] Top shards:")
            for f in largest:
                print("   ", os.path.basename(f), human_size(os.path.getsize(f)))
    except Exception:
        pass

if __name__ == "__main__":
    main()
