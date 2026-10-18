"""videval.scoring — 把原始指标映射到六维 0-5 分。

打分区间（Band）的设计原则
--------------------------
每个指标用分段线性函数映射到 0-5 分，而不是做全局 min-max 归一化。

原因在于：min-max 归一化会让分数随批次漂移——换一批素材，同一个模型的
得分就变了，横评结果无法跨批次比较。分段线性映射的锚点是绝对阈值，
"清晰度 947 分"在任何一批里都对应同一档，报告才站得住。

"适中最好"型指标
----------------
运动幅度、对比度、饱和度这类指标并非越高越好：
    运动幅度 = 0   → 画面凝固，AI 视频典型缺陷；
    运动幅度 过高  → 抽搐抖动，同样不可交付。
因此用「过低惩罚 + 峰值给分 + 过高回落」的三段式区间。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .design import DIMENSIONS, DIMENSION_BY_KEY, Band, band_score
from .design import weights_of, DEFAULT_PRESET

__all__ = [
    "MetricRule",
    "derive",
    "score_video",
    "weighted_total",
    "score_all",
    "dimension_report_card",
]


@dataclass
class MetricRule:
    """单条指标的打分规则。"""

    key: str
    bands: list[Band]
    weight: float = 1.0
    #: 指标缺失时（如该用例无参考图），是否跳过而非按 0 分计
    optional: bool = False
    note: str = ""


# ---------------------------------------------------------------- 派生指标


def derive(m: dict[str, Any]) -> dict[str, Any]:
    """补充派生指标，供打分规则直接引用。"""
    out = dict(m)

    w = float(m.get("width") or 0)
    h = float(m.get("height") or 0)
    fps = float(m.get("fps") or 0)
    br = float(m.get("bit_rate") or 0)

    px_per_sec = w * h * fps
    # 每像素每帧比特数：衡量「达到同样分辨率要付多少文件体积」，
    # 越低说明编码越高效、分发成本越低。
    out["bpp_frame"] = br / px_per_sec if px_per_sec > 0 else 0.0

    # 分辨率档位（按短边/高度归一到常见档）
    if h >= 1400:
        out["res_tier"] = 5.0
    elif h >= 1000:
        out["res_tier"] = 4.5
    elif h >= 700:
        out["res_tier"] = 4.0
    elif h >= 500:
        out["res_tier"] = 2.5
    else:
        out["res_tier"] = 1.0

    # 音轨评分：无音轨不能一律扣分。
    #   - 提示词要求了声音，成片无音轨 → 缺陷，0 分；
    #   - 提示词本就不需要声音（纯特效/字幕/报幕镜头）→ 该项不适用，
    #     置 None 让打分规则跳过，避免误扣。
    if m.get("requires_audio"):
        out["audio_score"] = 5.0 if m.get("has_audio") else 0.0
    else:
        out["audio_score"] = None

    # 切镜偏差取绝对值：多切与少切同样偏离指令，方向不影响扣分
    out["cut_intent_abs_delta"] = abs(float(m.get("cut_intent_delta") or 0))

    return out


# ---------------------------------------------------------------- 打分规则表

#: 说明：Band(lo, hi, score_lo, score_hi)
_RULES: dict[str, list[MetricRule]] = {
    "consistency": [
        # 以下阈值依据本次 96 条成片的实测分布标定：
        # 使中位数落在 3 分附近，p10 约 1 分、p90 约 4.5 分，保证分数有梯度。
        MetricRule(
            "ref_sim_mean",
            [Band(0.05, 0.70, 0.0, 5.0)],
            weight=2.0,
            optional=True,
            note="与参考图的整体吻合度",
        ),
        MetricRule(
            "ref_sim_std",
            [Band(0.02, 0.30, 5.0, 0.0)],
            weight=1.0,
            optional=True,
            note="帧间一致性波动，越小越稳",
        ),
        MetricRule(
            "subject_drift",
            [Band(-0.35, 0.05, 0.0, 5.0)],
            weight=1.0,
            optional=True,
            note="主体随时间漂移，负值表示越走越偏",
        ),
        MetricRule(
            "first_ref_sim",
            [Band(0.05, 0.90, 0.0, 5.0)],
            weight=1.0,
            optional=True,
            note="首帧与参考图的吻合度，反映起手是否跑偏",
        ),
        # 语义层：ArcFace 人脸一致性，直接测「是不是这个人」。
        # 权重给高，因为它是唯一直接量化角色一致的指标；
        # 阈值按 ArcFace 余弦相似度经验区间标定（<0.3 换脸/崩坏，
        # 0.5~0.7 基本同一人，>0.7 高度一致），使中位数落在 3 分附近。
        MetricRule(
            "face_sim_mean",
            [Band(0.15, 0.75, 0.0, 5.0)],
            weight=2.5,
            optional=True,
            note="成片人脸与参考角色的 ArcFace 余弦相似度",
        ),
    ],
    "motion": [
        MetricRule(
            "motion_mean",
            [
                Band(0.000, 0.008, 0.0, 1.5),   # 近乎凝固
                Band(0.008, 0.030, 1.5, 5.0),      # 理想区间
                Band(0.030, 0.090, 5.0, 3.0),      # 偏活跃尚可
                Band(0.090, 0.300, 3.0, 0.5),      # 剧烈
            ],
            weight=2.0,
            note="运动幅度，过低凝固、过高抽搐",
        ),
        MetricRule(
            "static_ratio",
            [Band(0.00, 0.50, 5.0, 0.0)],
            weight=1.5,
            note="静止帧占比，AI 视频典型缺陷",
        ),
        MetricRule(
            "motion_cv",
            [Band(0.30, 2.20, 5.0, 0.5)],
            weight=1.0,
            note="运动均匀性，越低越平滑",
        ),
        MetricRule(
            "motion_outlier_ratio",
            [Band(0.00, 0.35, 5.0, 0.0)],
            weight=1.0,
            note="运动突变帧占比，捕捉抖动抽搐",
        ),
    ],
    "frame_quality": [
        MetricRule(
            "sharpness_mean",
            [Band(150.0, 1100.0, 0.0, 5.0)],
            weight=2.0,
            note="拉普拉斯方差，衡量清晰度",
        ),
        MetricRule(
            "overexp_ratio",
            [Band(0.000, 0.120, 5.0, 0.0)],
            weight=1.5,
            note="过曝像素占比",
        ),
        MetricRule(
            "crushed_ratio",
            [Band(0.000, 0.300, 5.0, 0.0)],
            weight=1.0,
            note="死黑像素占比，暗部丢失细节",
        ),
    ],
    "continuity": [
        MetricRule(
            "cut_intent_abs_delta",
            [Band(0.0, 5.0, 5.0, 0.0)],
            weight=2.0,
            note="实际切点数与提示词期望的偏差",
        ),
        MetricRule(
            "temporal_sim_mean",
            [Band(0.900, 1.000, 0.0, 5.0)],
            weight=1.5,
            note="帧间连贯性，跳变会拉低该值",
        ),
    ],
    "color": [
        MetricRule(
            "contrast_mean",
            [
                Band(20.0, 55.0, 1.0, 5.0),
                Band(55.0, 85.0, 5.0, 3.5),
                Band(85.0, 115.0, 3.5, 1.5),
            ],
            weight=1.5,
            note="对比度，过低发灰、过硬刺眼",
        ),
        MetricRule(
            "saturation_mean",
            [
                Band(10.0, 60.0, 2.0, 5.0),
                Band(60.0, 110.0, 5.0, 3.5),
                Band(110.0, 180.0, 3.5, 1.0),
            ],
            weight=1.0,
            note="饱和度，过低寡淡、过高艳俗",
        ),
        MetricRule(
            "colorfulness",
            [
                Band(2.0, 18.0, 1.5, 5.0),
                Band(18.0, 45.0, 5.0, 2.5),
            ],
            weight=0.8,
            note="Hasler 色彩丰富度",
        ),
    ],
    "usability": [
        MetricRule(
            "audio_score",
            [Band(0.0, 5.0, 0.0, 5.0)],
            weight=2.0,
            optional=True,
            note="音轨完整性；提示词未要求声音时该项不适用",
        ),
        MetricRule(
            "res_tier",
            [Band(1.0, 5.0, 0.0, 5.0)],
            weight=1.5,
            note="分辨率档位",
        ),
        MetricRule(
            "bpp_frame",
            [Band(0.02, 0.45, 5.0, 1.0)],
            weight=0.8,
            note="码率效率，越低越省存储与带宽",
        ),
    ],
}


def _get(m: dict[str, Any], key: str) -> float | None:
    v = m.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def score_video(m: dict[str, Any]) -> dict[str, Any]:
    """对单个成片计算六维分数。

    返回包含各维度分数、总分与逐指标明细的字典。
    """
    d = derive(m)
    dim_scores: dict[str, float] = {}
    details: dict[str, list[dict[str, Any]]] = {}

    for dim in DIMENSIONS:
        rules = _RULES.get(dim.key, [])
        total_w = 0.0
        acc = 0.0
        rows: list[dict[str, Any]] = []

        for r in rules:
            v = _get(d, r.key)
            if v is None:
                if r.optional:
                    continue
                v = 0.0
            s = band_score(v, r.bands)
            acc += s * r.weight
            total_w += r.weight
            rows.append({
                "metric": r.key,
                "value": round(v, 4),
                "score": round(s, 2),
                "weight": r.weight,
                "note": r.note,
            })

        # 该维度所有指标都缺失（如无参考图、或参考图非设定图）时，
        # 记 None 表示「不可评估」，而非 0 分——0 分会污染横评均分。
        dim_scores[dim.key] = round(acc / total_w, 3) if total_w else None
        details[dim.key] = rows

    return {
        "path": m.get("path", ""),
        "model": m.get("model", ""),
        "case_id": m.get("case_id", ""),
        "duration": d.get("duration", 0.0),
        "height": d.get("height", 0),
        "width": d.get("width", 0),
        "has_audio": bool(d.get("has_audio", False)),
        "dims": dim_scores,
        "details": details,
        # 原始指标一并带出：缺陷归因要按绝对阈值直接定位问题样本，
        # 只看维度分数会丢掉「到底是过曝还是死黑」这类关键信息。
        "raw": d,
    }


def weighted_total(
    dims: dict[str, float | None],
    preset: str = DEFAULT_PRESET,
    weights: dict[str, float] | None = None,
) -> float | None:
    """按权重把六维分数合成总分。

    不可评估的维度（None）不参与合成，分母只计实际参与维度的权重；
    若所有维度都不可评估，返回 None。
    """
    w = weights if weights is not None else weights_of(preset)
    num = 0.0
    den = 0.0
    for k, wt in w.items():
        v = dims.get(k)
        if v is None:
            continue
        num += v * wt
        den += wt
    return round(num / den, 3) if den else None


def score_all(
    metric_rows: list[dict[str, Any]],
    preset: str = DEFAULT_PRESET,
) -> list[dict[str, Any]]:
    """批量打分，并附加权总分。"""
    out = []
    for m in metric_rows:
        s = score_video(m)
        s["total"] = weighted_total(s["dims"], preset)
        out.append(s)
    return out


def dimension_report_card(
    scored: list[dict[str, Any]],
    model: str,
) -> dict[str, float | None]:
    """某个模型在各维度上的平均分，用于雷达图与横向对比。

    跳过不可评估（None）的样本，只对有值的样本求均值。
    """
    rows = [s for s in scored if s.get("model") == model]
    if not rows:
        return {}
    card: dict[str, float | None] = {}
    for d in DIMENSIONS:
        vals = [r["dims"].get(d.key) for r in rows]
        vals = [v for v in vals if v is not None]
        card[d.key] = round(sum(vals) / len(vals), 3) if vals else None
    return card
