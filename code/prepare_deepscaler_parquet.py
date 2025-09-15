import os, yaml
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

def _pick_first(*keys):
    for d in keys:
        for k in d:
            if k in d and d[k] is not None:
                return d[k]
    return None

def main(out_path="data/deepscaler.parquet", cfg_path="configs/grpo.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    PROMPT_INSTRUCTION = cfg["prompt_template"]
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset")
    split = "train" if "train" in ds else list(ds.keys())[0]

    rows = []
    for ex in ds[split]:
        # 尝试兼容字段名（以官方字段为准：problem / solution / answer）
        problem = ex.get("problem") or ex.get("question") or ex.get("prompt")
        answer  = ex.get("answer") or ex.get("final_answer") or ex.get("label")
        if problem is None or answer is None:
            continue
        prompt = [{"role": "user", "content": PROMPT_INSTRUCTION.format(problem=problem)}]
        rows.append({
            "prompt": prompt, 
            "reward_model": {"ground_truth": str(answer)},  
            "data_source": "deepscaler",
        })
    Dataset.from_list(rows).to_parquet(out_path)
    print(f"[prepare] wrote {len(rows)} rows -> {out_path}")


def small_parquet(out_path, n):
    in_path  = "data/deepscaler.parquet"        # 原始 parquet 路径
    out_path = out_path  # 输出 parquet 路径

    df = pd.read_parquet(in_path)
    df_small = df.head(n)   # 只取前 5 条

    df_small.to_parquet(out_path, index=False)
    print(f"[done] wrote {len(df_small)} rows -> {out_path}")


def longest_parquet_len(data_path, model_path):
    df = pd.read_parquet(data_path)
    prompts = df["prompt"].astype(str)
    max_len = prompts.str.len().max()
    print("最大 prompt 长度:", max_len)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    token_counts = prompts.apply(lambda x: len(tokenizer.encode(x, add_special_tokens=False)))
    max_tokens = token_counts.max()
    print("最大 token 数:", max_tokens)


if __name__ == "__main__":
    main()
    small_parquet(out_path="data/deepscaler_20.parquet", n=20)
    longest_parquet_len("data/deepscaler_20.parquet", "weights/Qwen3-1p7B")
