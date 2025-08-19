import os, yaml
from datasets import load_dataset, Dataset

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
            "prompt": prompt,                 # VERL期望是 messages 列表
            "ground_truth": str(answer),      # 用于reward核对
            "data_source": "deepscaler",
        })
    Dataset.from_list(rows).to_parquet(out_path)
    print(f"[prepare] wrote {len(rows)} rows -> {out_path}")

if __name__ == "__main__":
    main()
