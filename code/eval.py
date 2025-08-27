import os, time, json, math, re
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, date
from reward_math_verify import compute_score_wpred
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ===== CONFIG（在这里改）=====
MODEL_PATH   = "/root/autodl-tmp/LLM_RL_basic/weights/Qwen3-1p7B"
DATA_PATH    = "/root/autodl-tmp/LLM_RL_basic/data/aime2025-all-small.parquet"   
DATA_SOURCE  = "AIME2025"   # 没差            

# vLLM & 采样参数
USE_BF16         = True
TENSOR_PARALLEL  = 1
MAX_NEW_TOKENS   = 2048
TEMPERATURE      = 0.6
TOP_P            = 0.95
TOP_K            = 20         # <=0 表示不启用 top-k
STOP_TOKENS      = []         # 例如 ["<|eot_id|>", "</s>"]

# 推理分批参数（外层切片；防 OOM）
BATCH_SIZE       = 1
# 输出
OUT_PATH         = "/root/autodl-tmp/LLM_RL_basic/data/eval/aime2025_eval.json"

def load_messages_like_verl(row: pd.Series) -> List[Dict[str, str]]:
    """尽量按 VERL 风格从一行样本中抽取对话消息."""
    # 1) messages
    if "messages" in row and pd.notna(row["messages"]):
        m = row["messages"]
        if isinstance(m, str):
            try:
                m = json.loads(m)
            except Exception:
                pass
        norm = []
        if isinstance(m, list):
            for seg in m:
                if isinstance(seg, dict) and "role" in seg and "content" in seg:
                    norm.append({"role": seg["role"], "content": str(seg["content"])})
                elif isinstance(seg, str):
                    norm.append({"role": "user", "content": seg})
        if norm:
            return norm

    # 2) system + (user/prompt)
    sys_txt = None
    if "system" in row and pd.notna(row["system"]):
        sys_txt = str(row["system"])

    user_txt = None
    if "user" in row and pd.notna(row["user"]):
        user_txt = str(row["user"])
    elif "prompt" in row and pd.notna(row["prompt"]):
        user_txt = str(row["prompt"])

    if sys_txt or user_txt:
        msgs = []
        if sys_txt:
            msgs.append({"role": "system", "content": sys_txt})
        if user_txt:
            msgs.append({"role": "user", "content": user_txt})
        return msgs

    # 3) 仅 prompt
    if "prompt" in row and pd.notna(row["prompt"]):
        return [{"role": "user", "content": str(row["prompt"])}]

    raise ValueError("无法从该样本构造对话：请提供 messages / (system+user|prompt) / prompt 至少其一。")

def render_prompt_from_messages(tokenizer, messages: List[Dict[str, str]]) -> str:
    """优先用 chat_template；否则退化为简单文本拼接."""
    try:
        tmpl = getattr(tokenizer, "chat_template", None)
        if tmpl:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        pass

    # 退化：简单标注角色
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = str(m.get("content", ""))
        if role == "system":
            parts.append(f"[SYSTEM]\n{content}\n")
        elif role == "assistant":
            parts.append(f"[ASSISTANT]\n{content}\n")
        else:
            parts.append(f"[USER]\n{content}\n")
    parts.append("[ASSISTANT]\n")
    return "\n".join(parts).strip()

def _extract_gold_from_reward_model(val):
    """从 reward_model 列提取 ground_truth；支持 dict 或 JSON 字符串。"""
    if isinstance(val, dict):
        return str(val.get("ground_truth", ""))
    if isinstance(val, str):
        try:
            obj = json.loads(val)
            if isinstance(obj, dict):
                return str(obj.get("ground_truth", ""))
        except Exception:
            pass
        return val
    return str(val)

_BOXED_PAT = r"\\boxed\{"
def _find_boxed_span_end(text: str) -> int | None:
    """
    返回 \boxed{...} 右花括号后的索引（切片上界），若未找到返回 None。
    处理嵌套花括号：从 '\boxed{' 的 '{' 开始做栈匹配。
    """
    m = re.search(_BOXED_PAT, text)
    if not m:
        return None
    i = m.end() - 1  # 指向 '\boxed{' 的 '{'
    depth = 0
    for j in range(i, len(text)):
        ch = text[j]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return j + 1  # 切片上界：包含右括号
    # 没闭合就不裁剪
    return None

def _trim_pred_and_count_tokens_until_boxed(pred: str, tokenizer) -> tuple[str, int | None]:
    """
    若存在完整的 \boxed{...} 则裁剪到此处（含右括号），并返回
    该裁剪片段的 token 数；若不存在，返回原文与 None。
    """
    end = _find_boxed_span_end(pred)
    if end is None:
        return pred, None
    trimmed = pred[:end]
    # 仅对生成片段计数（不含prompt）；使用 HF tokenizer 近似统计
    toks = tokenizer.encode(trimmed, add_special_tokens=False)
    return trimmed, int(len(toks))

def _extract_question_from_prompt_raw(prompt_raw: str) -> str:
    """
    按你给的格式：
    "[{'content': '...Problem:\\n题面...', 'role': 'user'}]"
    直接用正则把 content 抠出来，再保留从 'Problem:' 开始的部分。
    """
    if not isinstance(prompt_raw, str):
        prompt_raw = str(prompt_raw)

    # 抠出 content（非贪婪，直到 ', 'role': 'user'）
    m = re.search(r"content':\s*'(.*?)'\s*,\s*'role':\s*'user'", prompt_raw, flags=re.DOTALL)
    content = m.group(1) if m else prompt_raw

    # 仅保留 Problem: 后面的独立题面
    pos = content.find("Problem:")
    return content[pos:].strip() if pos != -1 else content.strip()

def _to_jsonable(x):
    """把 numpy / pandas 等对象递归转成可 JSON 序列化的纯 Python 类型。"""
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if x is pd.NA:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return [_to_jsonable(i) for i in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {str(_to_jsonable(k)): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (pd.Timestamp, datetime, date)):
        return x.isoformat()
    # 最后兜底成字符串
    try:
        return str(x)
    except Exception:
        return None

def main():
    # 读取数据
    df = pd.read_parquet(DATA_PATH)
    if "ground_truth" in df.columns:
        _golds = df["ground_truth"].astype(str).tolist()
    elif "reward_model" in df.columns:
        _golds = [_extract_gold_from_reward_model(x) for x in df["reward_model"]]
    else:
        raise ValueError("未找到 ground truth：请提供 ground_truth 列，或在 reward_model 中包含 {'ground_truth': ...}")


    # tokenizer 只用于 chat_template 渲染（不会加载模型权重）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)

    # 构造 prompts
    prompts: List[str] = []
    for _, row in df.iterrows():
        msgs = load_messages_like_verl(row)
        prompts.append(render_prompt_from_messages(tokenizer, msgs))

    # vLLM 初始化
    sampling_kwargs: Dict[str, Any] = dict(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_NEW_TOKENS,
        stop=STOP_TOKENS if STOP_TOKENS else None,
    )
    if TOP_K and TOP_K > 0:
        sampling_kwargs["top_k"] = TOP_K

    sampling = SamplingParams(**sampling_kwargs)
    engine = LLM(
        model=MODEL_PATH,
        dtype="bfloat16" if USE_BF16 else "auto",
        tensor_parallel_size=TENSOR_PARALLEL,
    )

    predictions: List[str] = []
    scores: List[float] = []
    tok_counts_all: List[Optional[int]] = []
    pred_anss: List[str] = []

    total = len(prompts)
    for s in range(0, total, BATCH_SIZE):
        batch_prompts = prompts[s:s + BATCH_SIZE]
        outputs = engine.generate(batch_prompts, sampling)

        batch_preds = [o.outputs[0].text if o.outputs else "" for o in outputs]
        golds_batch = _golds[s:s + BATCH_SIZE]

        # 裁剪到 \boxed{...} 并统计 token 数
        trimmed_preds: List[str] = []
        tok_counts_batch: List[Optional[int]] = []
        for p in batch_preds:
            tp, tc = _trim_pred_and_count_tokens_until_boxed(p, tokenizer)  # <- 用你前面加的函数
            trimmed_preds.append(tp)
            tok_counts_batch.append(tc)

        for pred, gold in zip(batch_preds, golds_batch):
            try:
                sc, pred_ans = compute_score_wpred(DATA_SOURCE, pred, gold, extra_info=None)
            except Exception:
                sc, pred_ans = 0.0, None
            predictions.append(pred)
            scores.append(sc)
            pred_anss.append(pred_ans)
            tok_counts_all.append(tc)

        mean_so_far = sum(scores) / len(scores)
        print(f"[{s + len(batch_prompts):>6}/{total}] mean_score={mean_so_far:.4f}")

    # 汇总 & 保存
    mean_score = float(sum(scores) / max(1, len(scores)))
    pass_like = float(sum(1 for x in scores if x >= 1.0) / max(1, len(scores)))  # 若你的评分非 0/1，可忽略或改阈值

    valid_tok = [t for t in tok_counts_all if t is not None]
    avg_tokens_until_boxed = float(sum(valid_tok) / len(valid_tok)) if valid_tok else 0.0

    print("\n=== SUMMARY ===")
    print(json.dumps({
        "num_items": total,
        "mean_score": mean_score,
        "pass@1_like": pass_like,
        "avg_tokens_until_boxed": avg_tokens_until_boxed
    }, ensure_ascii=False, indent=2))

    # 存储
    items = []
    use_df_prompt = "prompt" in df.columns
    for idx, (gold, pred, pred_ans, score, tokc) in enumerate(
        zip(_golds, predictions, pred_anss, scores, tok_counts_all)
    ):
        prompt_raw = df.iloc[idx]["prompt"] if use_df_prompt else prompts[idx]
        question = _extract_question_from_prompt_raw(prompt_raw)

        items.append({
            "prompt": _to_jsonable(prompt_raw),
            "question": _to_jsonable(question),
            "ground_truth": _to_jsonable(gold),
            "pred": _to_jsonable(pred),
            "pred_ans": _to_jsonable(pred_ans),
            "score": float(score),
            "tokens_until_boxed": _to_jsonable(tokc),
        })

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"\n已保存评测结果：{OUT_PATH}")

if __name__ == "__main__":
    main()
