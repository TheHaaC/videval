"""videval.calibrate — 用人工盲测分数校准客观指标。

校准的目的
----------
客观指标（像素级、直方图级）与人眼感知的相关性有限。
要判断客观分「测得对不对」，需要一轮人工盲测作为参照：

    对每个维度，算 客观分 ~ 人工分 的 Spearman 秩相关：

        r 高（如 > 0.6） → 客观指标可信，权重可保持或上调；
        r 低（如 < 0.2） → 该客观指标测的不是人眼关心的东西，换指标或降权。

校准不是给客观分「补一刀」，而是回答一个更根本的问题：
「我们自动化掉的那部分，到底有没有自动化对人？」
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .blind import BLIND_DIMS

__all__ = [
    "HUMAN_TO_OBJ",
    "load_human_scores",
    "align",
    "calibrate",
    "run_calibration",
]

#: 盲测维度名 → 客观维度 key
HUMAN_TO_OBJ: dict[str, str] = {
    "人物场景一致性": "consistency",
    "运动合理性": "motion",
    "单帧画面正确性": "frame_quality",
    "镜头衔接流畅性": "continuity",
    "光影色彩和谐性": "color",
    "音画与音效适配性": "usability",
}


def _to_float(v: Any) -> float | None:
    """打分表里可能混入 '\'、空串等，统一按缺失处理。"""
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "\\", "-", "—", "NA", "N/A", "nan"):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return f


def load_human_scores(sheet_path: str | Path) -> dict[str, dict[str, float | None]]:
    """读取盲测打分表，返回 blind_id → {维度名: 分数}。"""
    p = Path(sheet_path)
    out: dict[str, dict[str, float | None]] = {}
    with p.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            bid = row.get("blind_id", "").strip()
            if not bid:
                continue
            out[bid] = {d: _to_float(row.get(d)) for d in BLIND_DIMS}
    return out


def align(
    scored: Iterable[dict[str, Any]],
    blind_meta: list[dict[str, Any]],
    human: dict[str, dict[str, float | None]],
) -> list[dict[str, Any]]:
    """把客观分与人工分按视频路径对齐。

    scored 里每条含 path / dims / raw；
    blind_meta 提供 blind_id → src_path 的映射。
    """
    by_path = {s.get("path"): s for s in scored}
    aligned: list[dict[str, Any]] = []
    for b in blind_meta:
        s = by_path.get(b.get("src_path"))
        h = human.get(b.get("blind_id"))
        if s is None or h is None:
            continue
        aligned.append({
            "blind_id": b.get("blind_id"),
            "src_path": b.get("src_path"),
            "objective": s,
            "human": h,
        })
    return aligned


def _objective_value(s: dict[str, Any], dim: str) -> float | None:
    """取某维度的客观分数。

    音画维度（usability）的客观侧用「是否有音轨」——它才是
    人眼（耳朵）能感知、且与人工音画分对应的信号；
    分辨率、码率这类工程指标人无法盲测，不应拿来校准。
    """
    if dim == "usability":
        raw = s.get("raw") or {}
        return 1.0 if raw.get("has_audio") else 0.0
    dims = s.get("dims") or {}
    v = dims.get(dim)
    return float(v) if v is not None else None


def calibrate(
    aligned: list[dict[str, Any]],
    min_pairs: int = 8,
) -> dict[str, Any]:
    """对每个维度做 Spearman 相关，返回校准结论。"""
    from scipy.stats import spearmanr

    results: dict[str, Any] = {}
    for human_dim, obj_dim in HUMAN_TO_OBJ.items():
        xs: list[float] = []
        ys: list[float] = []
        for a in aligned:
            h = a["human"].get(human_dim)
            o = _objective_value(a["objective"], obj_dim)
            if h is None or o is None:
                continue
            xs.append(o)
            ys.append(h)

        if len(xs) < min_pairs:
            results[human_dim] = {
                "obj_dim": obj_dim, "n": len(xs),
                "spearman_r": None, "p": None,
                "verdict": "样本不足",
                "suggestion": "需更多盲测样本或该维度有效样本不足",
            }
            continue

        # 若任一变量方差为 0（如人工分全打同一个分），相关无定义
        if len(set(xs)) < 2 or len(set(ys)) < 2:
            results[human_dim] = {
                "obj_dim": obj_dim, "n": len(xs),
                "spearman_r": None, "p": None,
                "verdict": "无方差",
                "suggestion": "分数没有拉开，无法校准；请重新打分拉开档次",
            }
            continue

        r, p = spearmanr(xs, ys)
        if r is None or r != r:  # NaN 保护
            r, p = 0.0, 1.0

        if r >= 0.6:
            verdict, suggestion = "强相关", "客观指标可信，权重可保持或上调"
        elif r >= 0.4:
            verdict, suggestion = "中等相关", "客观指标基本可信，权重可保持"
        elif r >= 0.2:
            verdict, suggestion = "弱相关", "建议换指标或降低该客观指标权重"
        else:
            verdict, suggestion = "几乎不相关", "该客观指标与人工判断脱节，需更换或大幅降权"

        results[human_dim] = {
            "obj_dim": obj_dim, "n": len(xs),
            "spearman_r": round(float(r), 3), "p": round(float(p), 4),
            "verdict": verdict, "suggestion": suggestion,
        }

    return results


def run_calibration(
    scored: list[dict[str, Any]],
    outdir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """读打分表与映射，对齐并校准。

    返回 (对齐后的样本, 校准结果)。
    """
    out = Path(outdir)
    sheet = out / "盲测打分表.csv"
    meta = out / "blind_meta.json"

    if not sheet.exists() or not meta.exists():
        raise FileNotFoundError(
            "缺少打分表或映射文件，请先运行 `videval blind` 并回填打分表"
        )

    human = load_human_scores(sheet)
    blind_meta = json.loads(meta.read_text(encoding="utf-8"))
    aligned = align(scored, blind_meta, human)
    result = calibrate(aligned)
    return aligned, result
