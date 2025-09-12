import re
from math_verify import parse, verify
from math_verify.parser import (
    LatexExtractionConfig,
    ExprExtractionConfig,
    NormalizationConfig,
)

# 配置保持你之前的取向：gold 走 ExprExtraction；pred 只依赖 boxed 提取
GOLD_LATEX = LatexExtractionConfig()
PRED_LATEX = LatexExtractionConfig(
    try_extract_without_anchor=False,
    boxed_match_priority=0,
    normalization_config=NormalizationConfig(
        basic_latex=True,
        units=True,
        malformed_operators=False,
        nits=False,
        boxed="all",
        equations=False,
    ),
)
_GOLD_CFG = [GOLD_LATEX, ExprExtractionConfig()]
_PRED_CFG = [PRED_LATEX]

_BOXED_PAT = r"\\boxed\{"
def _find_all_boxed_spans(text: str):
    """返回所有完整 \boxed{...} 的 (start, end)（end 为右花括号后一位）。"""
    spans = []
    start = 0
    while True:
        m = re.search(_BOXED_PAT, text[start:])
        if not m:
            break
        i = start + m.end() - 1  # 指向 '{'
        depth = 0
        end_here = None
        for j in range(i, len(text)):
            ch = text[j]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end_here = j + 1
                    break
        if end_here is None:
            break  # 找到开头但未闭合，停止
        spans.append((start + m.start(), end_here))
        start = end_here
    return spans

def _extract_last_boxed(s: str) -> str | None:
    """在整段文本中返回最后一个完整 \\boxed{...} 的原文子串；不存在返回 None。"""
    spans = _find_all_boxed_spans(s)
    if not spans:
        return None
    st, ed = spans[-1]
    return s[st:ed]

def _normalize_gold(gt: str) -> str:
    """若 gold 没有成对的数学定界符，则自动包一层 $$...$$。"""
    s = gt.strip()
    # 常见的几类成对定界
    if (s.startswith("$$") and s.endswith("$$")):
        return s
    if (s.startswith("$") and s.endswith("$")):
        return s
    if (s.startswith(r"\[") and s.endswith(r"\]")):
        return s
    if (s.startswith(r"\(") and s.endswith(r"\)")):
        return s
    # 其它情况一律包 $$...$$
    return f"$${s}$$"

def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    s = str(solution_str).strip()
    last_boxed = _extract_last_boxed(s)
    if last_boxed is None:
        return 0.0
    try:
        gold_norm = _normalize_gold(str(ground_truth))
        gold = parse(gold_norm, extraction_config=_GOLD_CFG)
        pred = parse(last_boxed, extraction_config=_PRED_CFG)
        return 1.0 if verify(gold, pred) else 0.0
    except Exception:
        return 0.0

def compute_score_wpred(data_source, solution_str, ground_truth, extra_info=None):
    s = str(solution_str).strip()
    last_boxed = _extract_last_boxed(s)
    if last_boxed is None:
        return 0.0, None
    try:
        gold_norm = _normalize_gold(str(ground_truth))
        gold = parse(gold_norm, extraction_config=_GOLD_CFG)
        pred = parse(last_boxed, extraction_config=_PRED_CFG)
        pred_pretty = pred[0] if isinstance(pred, (list, tuple)) and pred else pred
        return (1.0 if verify(gold, pred) else 0.0), pred_pretty
    except Exception:
        return 0.0, None
