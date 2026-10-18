"""videval.prefs — 配对对比与偏好数据构造。

这一段是「评测」通向「后训练」的接口。

为什么评测天然产出偏好数据
--------------------------
本次实验是配对设计：同一条提示词、同一套参考资产，只有模型不同。
于是任意一次打分，本质都是一次「同一条件下 A 与 B 谁更好」的判断——
这正是 RLHF / DPO 训练所需的最小单元：

    (prompt, chosen, rejected)

多数评测项目止步于「给每个模型打个总分排个名」，偏好信息被丢弃了。
这里把它显式地留下来，一份评测可以同时产出：
    1. 横评报告（给人看，用于模型选型）；
    2. 偏好对（给训练用，用于 DPO / RLHF 的 reward 建模）。

如何避免把噪声喂进训练
----------------------
分差很小的样本对没有区分度，强行标注只是把标注噪声固化进模型。
因此这里只保留分差超过 tie_margin 的样本对，
并按分差排序，方便按预算截取置信度最高的一批。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from .design import (
    DIMENSIONS,
    DIMENSION_BY_KEY,
    SEGMENTS,
    SUBJECT_MODEL,
    segment_of,
)

__all__ = [
    "Comparison",
    "compare_case",
    "build_preferences",
    "aggregate",
    "to_dpo_records",
]


@dataclass
class Comparison:
    """一次「用例 × 维度」的配对比较结果。"""

    case_id: str
    segment: str
    opponent: str
    dimension: str
    dimension_name: str
    subject_mean: float
    opponent_mean: float
    margin: float
    outcome: str  # win | tie | lose（主语恒为评测对象）
    subject_n: int
    opponent_n: int


def _mean(xs: Sequence[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def compare_case(
    subject_rows: Sequence[dict[str, Any]],
    opponent_rows: Sequence[dict[str, Any]],
    case_id: str,
    segment_key: str,
    opponent: str,
    dimension: str,
    tie_margin: float = 0.25,
) -> Comparison | None:
    """比较同一用例下评测对象与对手在某个维度上的表现。"""
    if not subject_rows or not opponent_rows:
        return None

    s_vals = [r["dims"].get(dimension) for r in subject_rows]
    o_vals = [r["dims"].get(dimension) for r in opponent_rows]
    s_vals = [v for v in s_vals if v is not None]
    o_vals = [v for v in o_vals if v is not None]
    if not s_vals or not o_vals:
        return None

    s_mean, o_mean = _mean(s_vals), _mean(o_vals)
    margin = s_mean - o_mean

    if margin > tie_margin:
        outcome = "win"
    elif margin < -tie_margin:
        outcome = "lose"
    else:
        outcome = "tie"

    return Comparison(
        case_id=case_id,
        segment=segment_key,
        opponent=opponent,
        dimension=dimension,
        dimension_name=DIMENSION_BY_KEY[dimension].name,
        subject_mean=round(s_mean, 3),
        opponent_mean=round(o_mean, 3),
        margin=round(margin, 3),
        outcome=outcome,
        subject_n=len(s_vals),
        opponent_n=len(o_vals),
    )


def build_preferences(
    scored: list[dict[str, Any]],
    case_ids: Iterable[str] | None = None,
    tie_margin: float = 0.25,
    subject: str = SUBJECT_MODEL,
) -> list[Comparison]:
    """遍历各分段各用例，生成全量配对比较记录。

    比较维度除六维外，额外包含加权总分（key 为 "total"）。
    """
    wanted = set(case_ids) if case_ids is not None else None
    comps: list[Comparison] = []

    for seg in SEGMENTS:
        for cid in seg.case_ids:
            if wanted is not None and cid not in wanted:
                continue

            rows = [
                s for s in scored
                if str(s.get("case_id")) == cid
            ]
            s_rows = [r for r in rows if r.get("model") == subject]
            o_rows = [r for r in rows if r.get("model") == seg.opponent]
            if not s_rows or not o_rows:
                continue

            for dim in DIMENSIONS:
                c = compare_case(
                    s_rows, o_rows, cid, seg.key, seg.opponent,
                    dim.key, tie_margin,
                )
                if c:
                    comps.append(c)

            # 加权总分
            s_tot = _mean([r.get("total", 0.0) for r in s_rows])
            o_tot = _mean([r.get("total", 0.0) for r in o_rows])
            comps.append(Comparison(
                case_id=cid,
                segment=seg.key,
                opponent=seg.opponent,
                dimension="total",
                dimension_name="加权总分",
                subject_mean=round(s_tot, 3),
                opponent_mean=round(o_tot, 3),
                margin=round(s_tot - o_tot, 3),
                outcome=(
                    "win" if s_tot - o_tot > tie_margin
                    else "lose" if s_tot - o_tot < -tie_margin
                    else "tie"
                ),
                subject_n=len(s_rows),
                opponent_n=len(o_rows),
            ))

    return comps


@dataclass
class Aggregate:
    """某维度的胜负聚合。"""

    dimension: str
    dimension_name: str
    win: int = 0
    tie: int = 0
    lose: int = 0
    mean_margin: float = 0.0

    @property
    def decisive(self) -> int:
        """去平局后的有效比较数。"""
        return self.win + self.lose

    @property
    def net_win_rate(self) -> float:
        """净胜率：(胜-负)/有效数，区间 [-1, 1]。"""
        if not self.decisive:
            return 0.0
        return round((self.win - self.lose) / self.decisive, 3)

    @property
    def win_rate(self) -> float:
        """胜率：胜/有效数。"""
        if not self.decisive:
            return 0.0
        return round(self.win / self.decisive, 3)

    @property
    def outcome(self) -> str:
        """该维度的整体判定，主语恒为评测对象。

        用胜负场数而非平均分差判定：平均分差容易被个别极端样本
        拉偏，而场数是「赢了多少次」，更贴近模型选型的实际语境。
        """
        if self.win > self.lose:
            return "win"
        if self.lose > self.win:
            return "lose"
        return "tie"


def aggregate(
    comps: Sequence[Comparison],
    segment: str | None = None,
) -> list[Aggregate]:
    """按维度聚合胜负，可按分段过滤。"""
    from .design import DIMENSIONS

    keys = [d.key for d in DIMENSIONS] + ["total"]
    rows = [c for c in comps if segment is None or c.segment == segment]

    out: list[Aggregate] = []
    for k in keys:
        sub = [c for c in rows if c.dimension == k]
        if not sub:
            continue
        name = (
            DIMENSION_BY_KEY[k].name if k in DIMENSION_BY_KEY else "加权总分"
        )
        agg = Aggregate(dimension=k, dimension_name=name)
        for c in sub:
            if c.outcome == "win":
                agg.win += 1
            elif c.outcome == "lose":
                agg.lose += 1
            else:
                agg.tie += 1
        agg.mean_margin = round(_mean([c.margin for c in sub]), 3)
        out.append(agg)
    return out


def to_dpo_records(
    comps: Sequence[Comparison],
    prompt_by_case: dict[str, str],
    video_by: Any,
    tie_margin: float = 0.25,
    subject: str = SUBJECT_MODEL,
) -> list[dict[str, Any]]:
    """把比较结果转成 DPO 训练可直接读取的偏好记录。

    只保留分差超过 tie_margin 的样本对，避免把无区分度的噪声
    固化进 reward 模型。按 |margin| 降序返回，便于按预算截取。

    Args:
        comps: build_preferences 的输出。
        prompt_by_case: 用例 → 提示词原文。
        video_by: 可调用对象 (case_id, model) -> 视频路径列表。
    """
    records: list[dict[str, Any]] = []

    for c in comps:
        if c.outcome == "tie" or abs(c.margin) <= tie_margin:
            continue

        prompt = prompt_by_case.get(c.case_id, "")
        s_paths = video_by(c.case_id, subject)
        o_paths = video_by(c.case_id, c.opponent)

        if c.outcome == "win":
            chosen_model, chosen_paths = subject, s_paths
            rejected_model, rejected_paths = c.opponent, o_paths
        else:
            chosen_model, chosen_paths = c.opponent, o_paths
            rejected_model, rejected_paths = subject, s_paths

        records.append({
            "prompt": prompt,
            "chosen": {
                "model": chosen_model,
                "videos": chosen_paths,
            },
            "rejected": {
                "model": rejected_model,
                "videos": rejected_paths,
            },
            "meta": {
                "case_id": c.case_id,
                "segment": c.segment,
                "dimension": c.dimension,
                "margin": c.margin,
                "source": "videval-paired-comparison",
            },
        })

    records.sort(key=lambda r: abs(r["meta"]["margin"]), reverse=True)
    return records
