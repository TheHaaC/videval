"""videval.blind — 主观盲测准备：评分卡、分层抽样、匿名打分表。

盲测的用途
----------
客观指标（像素级、直方图级）与人眼感知的相关性天然有限。
要验证客观分数「测得对不对」，需要一轮人工盲测作为参照：

    客观分 ~ 人工分 的 Spearman 相关  →  相关性高的维度可信，低的要换指标或降权。

盲测必须「盲」：打分者不知道视频出自哪个模型，否则会带预设偏好。
因此这里把抽中的成片复制成匿名文件（B01、B02…），打分表里只出现匿名编号。

抽样原则
--------
从 96 条里按加权总分分层（低/中/高），每段每模型抽相同数量，
保证样本覆盖「好片」和「烂片」——若全是中不溜的样本，相关性根本算不出来。
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, Sequence

from .design import DIMENSIONS

__all__ = ["write_rubric", "sample_for_blind", "prepare_blind"]

# 盲测维度：前五维与客观六维一一对应，第六维取「可用性与成本」中
# 唯一可人眼判断的音轨部分（分辨率、码率无法盲测，交由客观指标）。
# 校准阶段前五维直接对客观分，第六维对客观的 has_audio。
BLIND_DIMS: list[str] = [
    "人物场景一致性",
    "运动合理性",
    "单帧画面正确性",
    "镜头衔接流畅性",
    "光影色彩和谐性",
    "音画与音效适配性",
]


# ---------------------------------------------------------------- 评分卡

_RUBRIC: dict[str, list[tuple[int, str]]] = {
    "人物场景一致性": [
        (5, "人物/场景与参考图吻合，跨帧稳定，无明显跑偏或身份漂移"),
        (4, "大体吻合，偶有轻微偏差但不影响辨识"),
        (3, "可辨识但有多处偏差（服饰、发型、场景细节对不上）"),
        (2, "明显不一致，人物或场景与参考图出入较大"),
        (1, "基本跑偏，难以与参考图建立联系"),
    ],
    "运动合理性": [
        (5, "运动自然流畅，幅度恰当，无抽搐、无冻结、无拖影"),
        (4, "整体流畅，个别帧轻微抖动或短暂滞涩"),
        (3, "有可见抽搐或局部凝固，但尚可接受"),
        (2, "明显抽搐/冻结，或运动幅度与内容不符"),
        (1, "严重抖动、冻结或运动失真，影响观看"),
    ],
    "单帧画面正确性": [
        (5, "清晰，曝光正常，无形变、无手部/面部崩坏"),
        (4, "基本正确，个别帧轻微模糊或小瑕疵"),
        (3, "部分帧可见形变、模糊或轻微崩坏"),
        (2, "多处崩坏、形变或曝光异常"),
        (1, "大面积崩坏/糊片/过曝死黑，无法使用"),
    ],
    "镜头衔接流畅性": [
        (5, "切镜干净利落，衔接连贯，节奏合理，符合提示词要求的镜头结构"),
        (4, "衔接基本流畅，个别切点略生硬"),
        (3, "切点可感知，或镜头结构与提示词有出入"),
        (2, "衔接明显断裂、跳变，或漏掉了提示词要求的切镜"),
        (1, "镜头混乱，衔接断裂，结构完全偏离"),
    ],
    "光影色彩和谐性": [
        (5, "光影层次丰富，色彩和谐，对比/饱和度舒适，暗部不死黑、亮部不过曝"),
        (4, "整体和谐，局部光影或色彩略有不足"),
        (3, "色彩寡淡或过艳，光影平淡，但不刺眼"),
        (2, "明显发灰/过艳/过曝或死黑，影响观感"),
        (1, "光影色彩严重失衡，观感差"),
    ],
    "音画与音效适配性": [
        (5, "有音轨，音画同步，氛围与画面匹配"),
        (4, "有音轨，基本同步，氛围尚可"),
        (3, "有音轨但同步欠佳，或氛围与画面略违和"),
        (2, "音轨明显不匹配，或本应有声却缺失"),
        (1, "无音轨（本应配音），或严重不同步"),
    ],
}


def write_rubric(path: str | Path) -> Path:
    """生成六维 0-5 档评分卡（Markdown）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    L = ["# 视频生成质量盲测评分卡", ""]
    L.append("对每条视频，在六个维度上各打 1–5 分（可用 0.5 步进）。")
    L.append("**打分时只依据画面本身，不要猜测或受文件名/顺序影响。**")
    L.append("")

    for name in BLIND_DIMS:
        L.append(f"## {name}")
        L.append("")
        L.append("| 分数 | 判据 |")
        L.append("|---|---|")
        for score, desc in _RUBRIC[name]:
            L.append(f"| {score} | {desc} |")
        L.append("")

    p.write_text("\n".join(L), encoding="utf-8")
    return p


# ---------------------------------------------------------------- 抽样


def _tier(total: float, lo: float, hi: float) -> str:
    if total <= lo:
        return "low"
    if total >= hi:
        return "high"
    return "mid"


def sample_for_blind(
    scored: Sequence[dict[str, Any]],
    n_per_model_per_seg: int = 6,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """按加权总分分层抽样，保证每段每模型覆盖低/中/高三档。

    每档取 n_per_model_per_seg // 3 条，不足则向上补。
    """
    import numpy as np

    from .design import SEGMENTS, SUBJECT_MODEL

    picked: list[dict[str, Any]] = []
    per_tier = max(1, n_per_model_per_seg // 3)

    for seg in SEGMENTS:
        for model in (SUBJECT_MODEL, seg.opponent):
            rows = [
                s for s in scored
                if str(s["case_id"]) in seg.case_ids and s["model"] == model
            ]
            if not rows:
                continue
            totals = np.array([r["total"] for r in rows])
            lo, hi = np.percentile(totals, [33, 67])
            # 按档次分组，组内按总分随机（用 seed 保证可复现）
            buckets: dict[str, list[dict[str, Any]]] = {
                "low": [], "mid": [], "high": []
            }
            for r in rows:
                buckets[_tier(r["total"], lo, hi)].append(r)
            rng = np.random.default_rng(seed)
            for tier in ("low", "mid", "high"):
                pool = buckets[tier]
                rng.shuffle(pool)
                picked.extend(pool[:per_tier])

    return picked


# ---------------------------------------------------------------- 准备产物


def prepare_blind(
    scored: Sequence[dict[str, Any]],
    outdir: str | Path,
    n_per_model_per_seg: int = 6,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    """准备盲测所需的全部产物。

    返回 (评分卡路径, 打分表路径, 盲测元数据)。
    盲测元数据含 blind_id 与原始路径的映射，校准阶段用来对回模型标签。
    """
    out = Path(outdir)
    blind_dir = out / "blind"
    blind_dir.mkdir(parents=True, exist_ok=True)

    rubric_path = write_rubric(out / "盲测评分卡.md")

    picked = sample_for_blind(scored, n_per_model_per_seg)
    picked.sort(key=lambda s: s["total"])  # 按质量升序，避免顺序暗示

    # 匿名复制
    blind_ids: list[str] = []
    rows: list[dict[str, Any]] = []
    for i, s in enumerate(picked, start=1):
        bid = f"B{i:02d}"
        blind_ids.append(bid)
        src = Path(s["path"])
        dst = blind_dir / f"{bid}{src.suffix}"
        if not dst.exists():
            shutil.copy2(src, dst)
        rows.append({
            "blind_id": bid,
            "video_file": f"blind/{dst.name}",
            "src_path": s["path"],
            "case_id": s["case_id"],
            "model": s["model"],  # 仅存于元数据，不写入打分表
            "total_objective": round(s["total"], 3),
        })

    # 打分表 CSV（不含模型名）
    sheet_path = out / "盲测打分表.csv"
    header = ["blind_id", "视频文件"] + BLIND_DIMS + ["备注"]
    with sheet_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r["blind_id"], r["video_file"]] + [""] * len(BLIND_DIMS) + [""])

    return rubric_path, sheet_path, rows
