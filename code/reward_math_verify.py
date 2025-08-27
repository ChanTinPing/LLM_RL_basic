import re
from math_verify import parse, verify
from math_verify.parser import (
    LatexExtractionConfig,
    ExprExtractionConfig,
    NormalizationConfig,
)

# gold：默认即可（你这版的 LatexExtractionConfig 默认就带合理的 NormalizationConfig）
GOLD_LATEX = LatexExtractionConfig()

# pred：提升 boxed 优先级，其他保持默认（已含 basic_latex/units/boxed="all"/equations=False）
PRED_LATEX = LatexExtractionConfig(
    try_extract_without_anchor=False,
    boxed_match_priority=0,  # 让 \boxed{} 优先抽取
    normalization_config=NormalizationConfig(  # 如需覆盖，按需写；不写就用默认
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

_BOX_ANYWHERE = re.compile(r'\\boxed\s*\{.+?\}')
def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    # 没有boxed或超过一个boxed输出0分
    s = solution_str.strip()
    matches = re.findall(_BOX_ANYWHERE, s)
    if len(matches) != 1:
        return 0.0
    
    try:
        gold = parse(str(ground_truth), extraction_config=_GOLD_CFG)
        pred = parse(str(solution_str), extraction_config=_PRED_CFG)
        score = 1.0 if verify(gold, pred) else 0.0
        return score
    except Exception:
        # 解析/验证失败按 0 分，避免训练中断
        return 0.0

def compute_score_wpred(data_source, solution_str, ground_truth, extra_info=None) -> tuple[float, str]:
    # 没有boxed或超过一个boxed输出0分
    s = solution_str.strip()
    matches = re.findall(_BOX_ANYWHERE, s)
    if len(matches) != 1:
        return 0.0, None
    
    try:
        gold = parse(str(ground_truth), extraction_config=_GOLD_CFG)
        pred = parse(str(solution_str), extraction_config=_PRED_CFG)
        score = 1.0 if verify(gold, pred) else 0.0
        return score, pred[0]
    except Exception:
        # 解析/验证失败按 0 分，避免训练中断
        return 0.0, None
