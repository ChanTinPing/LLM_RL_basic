# 自定义奖励：正确=1.0，错误=0.0；解析失败按0.0处理，避免训练中断
# VERL会以 (data_source, solution_str, ground_truth, extra_info) 调用该函数
try:
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
except Exception as e:
    raise RuntimeError(
        "请先安装 Math-Verify： pip install 'math-verify[antlr4_13_2]'"
    )

# 解析配置：gold较宽松；pred强调boxed
_GOLD_CFG = [LatexExtractionConfig(), ExprExtractionConfig()]
_PRED_CFG = [
    LatexExtractionConfig(basic_latex=True, units=True, equations=False, boxed="all"),
    ExprExtractionConfig()
]

def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    try:
        gold = parse(str(ground_truth), extraction_config=_GOLD_CFG)
        pred = parse(str(solution_str), extraction_config=_PRED_CFG)
        ok = verify(gold, pred)
        return 1.0 if ok else 0.0
    except Exception:
        return 0.0
