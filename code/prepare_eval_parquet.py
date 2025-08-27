# tools/prepare_eval_parquet.py
import os, yaml
import pandas as pd
from datasets import load_dataset, Dataset, get_dataset_config_names

def prepare_eval_parquet(dataset_id: str, out_path: str, cfg_path: str = "configs/grpo.yaml", split: str = None):
    cfg = yaml.safe_load(open(cfg_path))
    PROMPT_INSTRUCTION = cfg["prompt_template"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    # 枚举该 HF 数据集的所有子配置（若没有则返回空列表）
    try:
        subsets = get_dataset_config_names(dataset_id) or []
    except Exception:
        subsets = []

    if len(subsets) > 0:
        # 有 subset：逐个 subset 读取同一个 split，并合并
        for subset in subsets:
            if split is not None:
                ds = load_dataset(dataset_id, subset, split=split)
            else:
                ds_all = load_dataset(dataset_id, subset)
                use_split = "test" if "test" in ds_all else list(ds_all.keys())[0]
                ds = ds_all[use_split]

            for ex in ds:
                # 尝试兼容字段名（以常见字段为准：problem / question / prompt 与 answer 系）
                problem = ex.get("problem") or ex.get("question") or ex.get("prompt") or ex.get("input") or ex.get("query") or ex.get("instruction")
                answer  = ex.get("answer")  or ex.get("final_answer") or ex.get("label") or ex.get("target") or ex.get("output")
                if problem is None or answer is None:
                    continue
                prompt = [{"role": "user", "content": PROMPT_INSTRUCTION.format(problem=str(problem))}]
                rows.append({
                    "prompt": prompt,
                    "reward_model": {"ground_truth": str(answer)},
                    "data_source": f"{dataset_id}",  # f"{dataset_id}:{subset}",
                })
    else:
        # 无 subset：直接按 split 读取
        if split is not None:
            ds = load_dataset(dataset_id, split=split)
        else:
            ds_all = load_dataset(dataset_id)
            use_split = "test" if "test" in ds_all else list(ds_all.keys())[0]
            ds = ds_all[use_split]

        for ex in ds:
            problem = ex.get("problem") or ex.get("question") or ex.get("prompt") or ex.get("input") or ex.get("query") or ex.get("instruction")
            answer  = ex.get("answer")  or ex.get("final_answer") or ex.get("label") or ex.get("target") or ex.get("output")
            if problem is None or answer is None:
                continue
            prompt = [{"role": "user", "content": PROMPT_INSTRUCTION.format(problem=str(problem))}]
            rows.append({
                "prompt": prompt,
                "reward_model": {"ground_truth": str(answer)},
                "data_source": dataset_id,
            })

    Dataset.from_list(rows).to_parquet(out_path)
    print(f"[prepare] wrote {len(rows)} rows -> {out_path}")
    return len(rows)

def small_parquet(in_path: str, out_path: str, n: int = 10):
    df = pd.read_parquet(in_path)
    df_small = df.head(n)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_small.to_parquet(out_path, index=False)
    print(f"[small] wrote {len(df_small)} rows -> {out_path}")
    return len(df_small)

if __name__ == "__main__":
    # 示例：AIME2025（会自动合并 AIME2025-I 与 AIME2025-II）
    prepare_eval_parquet(
        dataset_id="opencompass/AIME2025",
        out_path="data/aime2025-all.parquet",
        cfg_path="configs/grpo.yaml",
        split="test",   # 两个 subset 都用同一个 split 读取
    )
    #small_parquet("data/aime2025-all.parquet", "data/aime2025-all-small.parquet", n=10)
