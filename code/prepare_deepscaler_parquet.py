import os, yaml, json, random
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


def small_parquet(out_path, start, end):
    in_path = "data/deepscaler.parquet"   # 原始 parquet 路径
    out_path = out_path                   # 输出 parquet 路径

    df = pd.read_parquet(in_path)
    df_small = df.iloc[start:end]   # 左闭右开，取[start, end)

    df_small.to_parquet(out_path, index=False)
    print(f"[done] wrote {len(df_small)} rows ({start}..{end-1}) -> {out_path}")


def longest_parquet_len(data_path, model_path):
    df = pd.read_parquet(data_path)
    prompts = df["prompt"].astype(str)
    max_len = prompts.str.len().max()
    print("最大 prompt 长度:", max_len)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    token_counts = prompts.apply(lambda x: len(tokenizer.encode(x, add_special_tokens=False)))
    max_tokens = token_counts.max()
    print("最大 token 数:", max_tokens)


def make_synthetic_coinflip_dataset(
    out_path="data/coinflip_boxed.parquet",
    num_rows=200,                  # 生成多少条 prompt（建议>=100，便于统计全对/全错发生率）
    filler_words_min=300,          # 要求输出的“长字串”下限（词数/近似token数）
    seed=1234,
):
    """
    生成用于 DAPO 全对/全错丢弃机制的合成数据：
    - 模型被强制先输出一大段内容，且中间禁止出现任何 `\\boxed{}`。
    - 最后一步再输出且仅输出一次 `\\boxed{1}` 或 `\\boxed{2}`（模型自行等概率选择）。
    - 奖励真值固定为 "1"，用于判别“对/错”。
    """
    random.seed(seed)

    task_en = (
        f"Rules (EN mirror):\n"
        f"- First, write at least {filler_words_min} words of arbitrary text (narration/lists/whatever), "
        "but **do NOT include any `\\boxed{}` anywhere** in that body;\n"
        "- Then, on the **last line**, output **exactly once** either `\\boxed{1}` or `\\boxed{2}`;\n"
        "- Choose `\\boxed{1}` and `\\boxed{2}` with equal probability (50/50);\n"
        "- `\\boxed{}` must appear **exactly once in the whole output** (only the final line);\n"
        "- Do NOT use other math delimiters like `$...$` or `\\[...\\]` anywhere.\n"
        "- Put only the `\\boxed{1}` or `\\boxed{2}` on the last line, with no trailing characters."
    )

    base_problem = (
        "生成一段任意主题的长文本作为“正文”，严格遵守上述规则后在最后给出盒子结果。"
    )

    def build_user_content():
        body = f"{task_en}\n\n{base_problem}"
        return body

    rows = []
    for _ in range(num_rows):
        user_text = build_user_content()
        prompt_msgs = [{"role": "user", "content": user_text}]

        rows.append({
            "prompt": prompt_msgs,
            "reward_model": {"ground_truth": "1"},
            "data_source": "synthetic_coinflip_boxed",
        })

    # 写 parquet
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds = Dataset.from_list(rows)
    ds.to_parquet(out_path)
    print(f"[synthetic] wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    # main()
    # small_parquet(out_path="data/deepscaler_8_15.parquet", start=8, end=15)
    # longest_parquet_len("data/deepscaler_20.parquet", "weights/Qwen3-1p7B")
    make_synthetic_coinflip_dataset("data/random_dapo.parquet", 10)
