"""videval.design — 评测设计常量：分段、维度、权重。

把「评什么、跟谁比、怎么加权」这类设计决策集中在一处，
指标计算与报告生成都从这里取值，避免阈值散落在各模块里漂移。

本次评测的实验设计
------------------
评测对象（subject）：Rally
对手（opponent）：按用例分段而异

    Segment A  用例 01-12   对手 = Seedance 2.0
    Segment B  用例 13-24   对手 = Seedance 2.5

采用配对对比（paired comparison）：同一条提示词、同一套参考资产，
仅模型不同，从而把「提示词难度」这个混淆变量从比较中消除。
每个用例每个模型 2 次生成，可用于估计模型自身的输出方差。
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: 评测对象。所有胜负统计均以它为主语表述。
SUBJECT_MODEL = "Rally"


@dataclass(frozen=True)
class Segment:
    """一个对比分段。"""

    key: str
    label: str
    opponent: str
    case_ids: tuple[str, ...]
    note: str = ""

    @property
    def pair(self) -> tuple[str, str]:
        return (SUBJECT_MODEL, self.opponent)


def _ids(start: int, end: int) -> tuple[str, ...]:
    return tuple(str(i) for i in range(start, end + 1))


SEGMENTS: tuple[Segment, ...] = (
    Segment(
        key="A",
        label="分段 A",
        opponent="Seedance 2.0",
        case_ids=_ids(1, 12),
        note="偏中短镜头，对手为 Seedance 2.0",
    ),
    Segment(
        key="B",
        label="分段 B",
        opponent="Seedance 2.5",
        case_ids=_ids(13, 24),
        note="含多镜头长视频，对手为 Seedance 2.5",
    ),
)

SEGMENT_BY_KEY: dict[str, Segment] = {s.key: s for s in SEGMENTS}

#: 用例 → 所属分段
CASE_TO_SEGMENT: dict[str, Segment] = {
    cid: seg for seg in SEGMENTS for cid in seg.case_ids
}


def segment_of(case_id: str) -> Segment | None:
    return CASE_TO_SEGMENT.get(str(case_id))


def opponent_of(case_id: str) -> str | None:
    seg = segment_of(case_id)
    return seg.opponent if seg else None


# ---------------------------------------------------------------- 六维体系


@dataclass(frozen=True)
class Dimension:
    """一个评测维度。"""

    key: str
    name: str
    desc: str
    #: 参与该维度打分的客观指标（key 需与 metrics 模块输出一致）
    inputs: tuple[str, ...] = ()
    #: 默认权重（各套预设见 WEIGHT_PRESETS）
    default_weight: float = 1.0


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        key="consistency",
        name="人物场景一致性",
        desc="成片与参考图在人物、场景上的吻合程度，跨帧是否稳定",
        inputs=("ref_sim_mean", "ref_sim_std", "subject_drift"),
        default_weight=1.2,
    ),
    Dimension(
        key="motion",
        name="运动合理性",
        desc="运动的幅度是否恰当、是否平滑，有无抖动抽搐或过度凝固",
        inputs=("motion_mean", "motion_cv", "static_ratio", "motion_outlier_ratio"),
        default_weight=1.2,
    ),
    Dimension(
        key="frame_quality",
        name="单帧画面正确性",
        desc="清晰度、曝光、有无形变或崩坏的典型缺陷",
        inputs=("sharpness_mean", "overexp_ratio", "crushed_ratio"),
        default_weight=1.0,
    ),
    Dimension(
        key="continuity",
        name="镜头衔接流畅性",
        desc="镜头切分是否执行、切点是否干净、衔接是否连贯",
        inputs=("shot_cut_count", "cut_intent_delta", "temporal_sim_mean"),
        default_weight=1.0,
    ),
    Dimension(
        key="color",
        name="光影色彩和谐性",
        desc="对比度、饱和度、色彩丰富度是否落在舒适区间",
        inputs=("contrast_mean", "saturation_mean", "colorfulness"),
        default_weight=0.8,
    ),
    Dimension(
        key="usability",
        name="可用性与成本",
        desc="音轨完整性、分辨率与码率等工程可用性因素",
        inputs=("has_audio", "height", "efficiency"),
        default_weight=0.8,
    ),
)

DIMENSION_BY_KEY: dict[str, Dimension] = {d.key: d for d in DIMENSIONS}

#: 权重预设。切换场景时只需换一个 key。
WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "balanced": {d.key: d.default_weight for d in DIMENSIONS},
    "story": {
        "consistency": 1.5,
        "motion": 1.2,
        "frame_quality": 1.0,
        "continuity": 1.3,
        "color": 0.6,
        "usability": 0.4,
    },
    "commercial": {
        "consistency": 1.0,
        "motion": 1.0,
        "frame_quality": 1.5,
        "continuity": 0.8,
        "color": 1.2,
        "usability": 1.3,
    },
}

DEFAULT_PRESET = "balanced"


def weights_of(preset: str = DEFAULT_PRESET) -> dict[str, float]:
    if preset not in WEIGHT_PRESETS:
        raise KeyError(
            f"未知权重预设：{preset}，可选 {list(WEIGHT_PRESETS)}"
        )
    return dict(WEIGHT_PRESETS[preset])


@dataclass
class Band:
    """分段线性打分区间：把原始指标映射到 0-5 分。

    含义：value 落在 [lo, hi] 内时，分数在 [score_lo, score_hi] 间线性插值。
    超出边界取端点值。这样每一分都能说清「因为什么值、落在哪一段」。
    """

    lo: float
    hi: float
    score_lo: float
    score_hi: float

    def score(self, v: float) -> float:
        if v <= self.lo:
            return self.score_lo
        if v >= self.hi:
            return self.score_hi
        t = (v - self.lo) / (self.hi - self.lo)
        return self.score_lo + t * (self.score_hi - self.score_lo)


def band_score(value: float, bands: list[Band]) -> float:
    """依次尝试各区间，返回首个命中区间的插值分数。"""
    for b in bands:
        if b.lo <= value <= b.hi:
            return b.score(value)
    if value < bands[0].lo:
        return bands[0].score_lo
    return bands[-1].score_hi
