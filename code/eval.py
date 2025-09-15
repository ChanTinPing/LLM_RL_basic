import os, time, json, math, re, torch
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, date
from reward_math_verify import compute_score_wpred
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ===== CONFIG（在这里改）=====
MODEL_PATH     = "/root/autodl-tmp/LLM_RL_basic/weights/Qwen3-1p7B"
DATA_PATH      = "/root/autodl-tmp/LLM_RL_basic/data/aime2025-all.parquet"   
DATA_SOURCE    = "useless"   # 没差  
K_TOTAL        = 8          # 想要的最终候选数
K_PER_ROUND    = 4          # 单轮 k
MAX_NEW_TOKENS = 32768
assert K_TOTAL % K_PER_ROUND == 0

# vLLM & 采样参数
USE_BF16         = True
TENSOR_PARALLEL  = 1
TEMPERATURE      = 0.6
TOP_P            = 0.95
TOP_K            = 20         # <=0 表示不启用 top-k
STOP_TOKENS      = []         # 例如 ["<|eot_id|>", "</s>"]

# 显存相关
BATCH_SIZE              = 1
MAX_NUM_SEQS            = BATCH_SIZE * K_PER_ROUND   # 并发槽位上限；单人推理就设成你的最大 batch
GPU_MEMORY_UTILIZATION  = 0.9          # 预留显存比例
# 输出
OUT_PATH         = "/root/autodl-tmp/LLM_RL_basic/data/eval/aime2025_eval.json"
PARTS_DIR        = OUT_PATH + ".parts"            # ← 新增：分片目录
PROGRESS_PATH    = OUT_PATH + ".progress.json"    # ← 新增：进度文件
SUMMARY_PATH     = OUT_PATH + ".summary.json"     # 在配置区加上这一行


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

def _part_path(idx: int) -> str:
    return os.path.join(PARTS_DIR, f"{idx:06d}.json")

def _load_done_indices() -> set:
    if not os.path.isdir(PARTS_DIR):
        return set()
    s = set()
    for fn in os.listdir(PARTS_DIR):
        if fn.endswith(".json"):
            try:
                s.add(int(fn.split(".")[0]))
            except Exception:
                pass
    return s

def _write_progress(d: dict):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def main():
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)  
    os.makedirs(PARTS_DIR, exist_ok=True)                         
    
    # 读取数据
    df = pd.read_parquet(DATA_PATH)
    try:
        _golds = [_extract_gold_from_reward_model(x) for x in df["reward_model"]]
    except Exception as e:
        raise ValueError("未找到 reward_model")

    # tokenizer 只用于 chat_template 渲染（不会加载模型权重）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=True)

    # 构造 prompts
    prompts: List[str] = []
    for _, row in df.iterrows():
        msgs = load_messages_like_verl(row)
        prompts.append(render_prompt_from_messages(tokenizer, msgs))
    total = len(prompts)  

    # vLLM 初始化
    engine = LLM(
        model=MODEL_PATH,
        dtype="bfloat16" if USE_BF16 else "auto",
        tensor_parallel_size=TENSOR_PARALLEL,
        max_num_seqs=MAX_NUM_SEQS,                     
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION  
    )
    
    # 断点续跑：读取已完成分片
    done_idx = _load_done_indices()             
    pending = [i for i in range(total) if i not in done_idx]
    print(f"已完成 {len(done_idx)}/{total}；将继续剩余 {len(pending)} 个样本")  
    _write_progress({"done": len(done_idx), "total": total, "last_finished": (max(done_idx) if done_idx else None)})

    # ===== 每个 batch 多轮生成到 K_TOTAL =====
    for chunk_start in range(0, len(pending), BATCH_SIZE):      # ← 替换原 for s in ...
        idxs = pending[chunk_start:chunk_start + BATCH_SIZE]    # 全局索引列表
        batch_prompts = [prompts[i] for i in idxs]
        golds_batch   = [_golds[i]  for i in idxs]

        # 为本 batch 每题准备候选累积容器
        cand_texts_batch: List[List[str]] = [[] for _ in range(len(batch_prompts))]

        num_rounds = K_TOTAL // K_PER_ROUND
        for r in range(num_rounds):
            sampling_round = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_NEW_TOKENS,
                stop=STOP_TOKENS if STOP_TOKENS else None,
                n=K_PER_ROUND,
                **({"top_k": TOP_K} if TOP_K and TOP_K > 0 else {})
            )
            outputs_r = engine.generate(batch_prompts, sampling_round)
            for i, out in enumerate(outputs_r):
                cand_texts_batch[i].extend([cand.text for cand in (out.outputs or [])])

        # === 逐题：计 token、打分、聚合 ===
        for local_i, (cand_texts, gold) in enumerate(zip(cand_texts_batch, golds_batch)):
            global_idx = idxs[local_i]  # 映射到全局题号

            # 逐候选计 token
            cand_tokc: List[Optional[int]] = []
            for ptxt in cand_texts:
                toks = tokenizer.encode(ptxt, add_special_tokens=False)
                cand_tokc.append(len(toks))

            # 逐候选打分 + 记录boxed答案
            cand_scores: List[float] = []
            cand_predans: List[Optional[str]] = []
            for ptxt in cand_texts:
                try:
                    sc, pred_ans = compute_score_wpred(DATA_SOURCE, ptxt, gold, extra_info=None)
                except Exception:
                    sc, pred_ans = 0.0, None
                cand_scores.append(sc)
                cand_predans.append(pred_ans)

            # 聚合：avg@K、pass@K、tokens 平均
            avg_k = float(sum(cand_scores) / max(1, len(cand_scores)))
            pass_k = 1.0 if any(sc >= 1.0 for sc in cand_scores) else 0.0
            valid_tok = [t for t in cand_tokc if t is not None]
            avg_tok = float(sum(valid_tok) / len(valid_tok)) if valid_tok else 0.0

            # —— 构造“单题分片”并立即写盘（断点续跑的关键） ——
            use_df_prompt = "prompt" in df.columns
            prompt_raw = df.iloc[global_idx]["prompt"] if use_df_prompt else prompts[global_idx]
            question = _extract_question_from_prompt_raw(prompt_raw)

            item_dict = {
                "prompt": _to_jsonable(prompt_raw),
                "question": _to_jsonable(question),
                "answer": _to_jsonable(_golds[global_idx]),
                "candidates": [
                    {
                        "pred": _to_jsonable(cand_texts[j]),
                        "pred_ans": _to_jsonable(cand_predans[j]),
                        "score": float(cand_scores[j]),
                        "avg_tokens": _to_jsonable(cand_tokc[j]),
                    }
                    for j in range(len(cand_texts))
                ],
                f"avg@{K_TOTAL}": float(avg_k),
                f"pass@{K_TOTAL}": float(pass_k),
                "avg_tokens_item": float(avg_tok),
                "index": int(global_idx),
            }

            with open(_part_path(global_idx), "w", encoding="utf-8") as f:
                json.dump(item_dict, f, ensure_ascii=False, indent=2)

            # —— 更新进度、并打印“第几个 prompt” ——
            done_now = len(_load_done_indices())
            _write_progress({"done": done_now, "total": total, "last_finished": int(global_idx)})
            print(f"[{done_now}/{total}] 完成 index={global_idx}")

    # ===== 汇总 =====
    done_idx = _load_done_indices()
    n = len(done_idx)
    if n > 0:
        avg_at_k = 0.0
        pass_at_k_like = 0.0
        avg_tokens = 0.0

        for idx in done_idx:
            with open(_part_path(idx), "r", encoding="utf-8") as f:
                item = json.load(f)
            avg_at_k += item.get(f"avg@{K_TOTAL}", 0.0)
            pass_at_k_like += item.get(f"pass@{K_TOTAL}", 0.0)
            avg_tokens += item.get("avg_tokens_item", 0.0)

        avg_at_k /= n
        pass_at_k_like /= n
        avg_tokens /= n
        summary = {
            "num_items": n,
            f"avg@{K_TOTAL}": avg_at_k,
            f"pass@{K_TOTAL}": pass_at_k_like,
            "avg_tokens": avg_tokens,
        }

        print("\n=== SUMMARY ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n已保存 summary：{SUMMARY_PATH}")
    else:
        print("\n=== SUMMARY ===")
        print("没有任何分片结果。")

if __name__ == "__main__":
    main()
