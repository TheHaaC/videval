"""videval.stats — 配对比较的统计显著性检验。

为什么不能只数「胜/平/负」
------------------------
把每个用例的分差离散化成胜/平/负，会丢掉分差的大小：同样是「赢」，
+0.05 和 +0.5 完全不是一回事。更关键的是，每段只有 12 个配对用例，
小样本下「6 胜 1 负」很可能只是随机波动。

因此这里对**连续的分差值序列**做两件事：

    1. Wilcoxon 符号秩检验 —— 检验分差的中位数是否显著偏离 0
       （非参数，不假设正态分布，适合小样本配对数据）；
    2. bootstrap 置信区间 —— 重采样估计均值分差的 95% CI，
       看它是否跨过 0。

报告时只认 p 值和置信区间，不再只给「几胜几负」。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "PairedTest",
    "paired_test",
    "bootstrap_ci",
    "segment_significance",
    "verdict",
]

#: 显著性判定阈值
P_SIGNIFICANT = 0.05
P_MARGINAL = 0.10


def verdict(p: float, mean_margin: float) -> str:
    """把 p 值 + 方向转成可读判定。"""
    if p < P_SIGNIFICANT:
        return "显著占优" if mean_margin > 0 else "显著落后"
    if p < P_MARGINAL:
        return "边缘占优" if mean_margin > 0 else "边缘落后"
    return "无显著差异"


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """对均值做 bootstrap 置信区间。

    注意重采样单位是「用例」，而非「成片」——保持配对结构，
    否则会低估方差、把不显著的差异误判成显著。
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


@dataclass
class PairedTest:
    """某个维度（或总分）的显著性检验结果。"""

    dimension: str
    dimension_name: str
    n: int
    mean_margin: float
    median_margin: float
    wilcoxon_p: float
    ci_lo: float
    ci_hi: float

    @property
    def outcome(self) -> str:
        return verdict(self.wilcoxon_p, self.mean_margin)

    @property
    def ci_str(self) -> str:
        return f"[{self.ci_lo:+.2f}, {self.ci_hi:+.2f}]"


def paired_test(
    margins: Sequence[float],
    dimension: str,
    dimension_name: str = "",
    n_boot: int = 10000,
) -> PairedTest:
    """对一组配对分差做 Wilcoxon 检验 + bootstrap CI。"""
    from scipy.stats import wilcoxon

    arr = np.asarray([m for m in margins if m == m], dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return PairedTest(
            dimension=dimension, dimension_name=dimension_name, n=0,
            mean_margin=0.0, median_margin=0.0, wilcoxon_p=1.0,
            ci_lo=0.0, ci_hi=0.0,
        )

    # 去掉精确为 0 的差值（Wilcoxon 符号秩检验要求非零）
    nonzero = arr[arr != 0.0]
    if len(nonzero) == 0:
        p = 1.0
    elif len(nonzero) == 1:
        p = 1.0  # 单样本无法拒绝原假设
    else:
        _, p = wilcoxon(nonzero)

    lo, hi = bootstrap_ci(arr, n_boot=n_boot)
    return PairedTest(
        dimension=dimension,
        dimension_name=dimension_name,
        n=len(arr),
        mean_margin=float(arr.mean()),
        median_margin=float(np.median(arr)),
        wilcoxon_p=float(p),
        ci_lo=lo,
        ci_hi=hi,
    )


def segment_significance(
    comps: Sequence,
    segment: str,
) -> list[PairedTest]:
    """对某分段的所有维度做显著性检验。

    comps 来自 prefs.build_preferences，含 case_id / dimension / margin。
    按维度聚合出「每个用例一个分差」的序列后逐一检验。
    """
    from .design import DIMENSIONS, DIMENSION_BY_KEY

    keys = [d.key for d in DIMENSIONS] + ["total"]
    rows = [c for c in comps if c.segment == segment]

    out: list[PairedTest] = []
    for k in keys:
        margins = [c.margin for c in rows if c.dimension == k]
        if not margins:
            continue
        name = DIMENSION_BY_KEY[k].name if k in DIMENSION_BY_KEY else "加权总分"
        out.append(paired_test(margins, k, name))
    return out
