"""videval.compare — 人工对比打分（配对选优）与客观指标校准。

为什么对比打分优于绝对打分
--------------------------
人做「哪个更好」的相对判断，远比「打几分」的绝对判断可靠——
后者会系统性偏松（leniency bias），前者天然有参照物。
这也正是 RLHF 用偏好对（chosen/rejected）而非评分的根本原因。

校准的目标
----------
客观指标在每个用例上会给出 Rally vs 对手的胜负判定（win/tie/lose）。
人工对比打分给出同样的胜负判定。两者的**一致率**与 **Cohen's κ**
衡量「客观指标在多大程度上模拟了人的判断」：

    一致率高 → 客观指标可信，可放心自动化；
    一致率低 → 客观指标测的东西与人关心的不一致，需换指标或降权。
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .design import SEGMENTS, SUBJECT_MODEL

__all__ = [
    "build_compare",
    "calibrate_compare",
    "load_judgments",
]


def _best(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """取总分最高的一条（None 视为 -inf）。"""
    valid = [r for r in rows if r.get("total") is not None]
    if not valid:
        return rows[0] if rows else None
    return max(valid, key=lambda r: r["total"])


def build_compare(
    scored: Sequence[dict[str, Any]],
    cases: Sequence[Any],
    outdir: str | Path,
    seed: int = 42,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    """生成人工对比打分的全部材料。

    每个用例一组：Rally 与对手各取客观总分最高的代表，匿名复制为
    `C{id}_A.mp4` / `C{id}_B.mp4`，A/B 顺序随机（可复现）。

    返回 (对比打分表, 映射文件, 对比记录)。
    """
    out = Path(outdir)
    cmp_dir = out / "compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for seg in SEGMENTS:
        opponent = seg.opponent
        for cid in seg.case_ids:
            rows = [s for s in scored if str(s["case_id"]) == cid]
            s_rows = [r for r in rows if r["model"] == SUBJECT_MODEL]
            o_rows = [r for r in rows if r["model"] == opponent]
            s_best = _best(s_rows)
            o_best = _best(o_rows)
            if not s_best or not o_best:
                continue

            rng = np.random.default_rng(seed + int(cid))
            if rng.random() < 0.5:
                a_src, a_model = s_best["path"], SUBJECT_MODEL
                b_src, b_model = o_best["path"], opponent
            else:
                a_src, a_model = o_best["path"], opponent
                b_src, b_model = s_best["path"], SUBJECT_MODEL

            cname = f"C{int(cid):02d}"
            dst_a = cmp_dir / f"{cname}_A.mp4"
            dst_b = cmp_dir / f"{cname}_B.mp4"
            if not dst_a.exists():
                shutil.copy2(a_src, dst_a)
            if not dst_b.exists():
                shutil.copy2(b_src, dst_b)

            records.append({
                "compare_id": cname,
                "case_id": cid,
                "segment": seg.key,
                "model_a": a_model,
                "model_b": b_model,
                "video_a": f"compare/{dst_a.name}",
                "video_b": f"compare/{dst_b.name}",
                "src_a": a_src,
                "src_b": b_src,
            })

    # 对比打分表：只给匿名视频，不给模型名
    sheet = out / "对比打分表.csv"
    by_case = {c.case_id: c for c in cases}
    with sheet.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["对比编号", "提示词摘要", "视频A", "视频B", "判定(A/B/平)"])
        for r in records:
            c = by_case.get(r["case_id"])
            summary = c.prompt[:60].replace("\n", " ") if c else ""
            w.writerow([
                r["compare_id"], summary,
                r["video_a"], r["video_b"], "",
            ])

    meta_path = out / "compare_meta.json"
    meta_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return sheet, meta_path, records


def load_judgments(sheet_path: str | Path) -> dict[str, str]:
    """读取人工对比判定，返回 compare_id → ('A'|'B'|'平')。"""
    p = Path(sheet_path)
    out: dict[str, str] = {}
    with p.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cid = row.get("对比编号", "").strip()
            if not cid:
                continue
            j = str(row.get("判定(A/B/平)", "")).strip().upper()
            if j in ("A", "B"):
                out[cid] = j
            elif j in ("平", "=", "TIE", "—"):
                out[cid] = "平"
    return out


def _objective_outcome(case_id: str, comps: Sequence[Any]) -> str | None:
    """从客观配对比较里取该用例总分的胜负判定。"""
    for c in comps:
        if str(c.case_id) == str(case_id) and c.dimension == "total":
            return c.outcome  # win / tie / lose
    return None


def _human_outcome(record: dict[str, Any], judgment: str) -> str | None:
    """把人工的 A/B/平 判定转成以 Rally 为主语的 win/tie/lose。"""
    if judgment == "平":
        return "tie"
    chosen = record["model_a"] if judgment == "A" else record["model_b"]
    if chosen == SUBJECT_MODEL:
        return "win"
    return "lose"


def calibrate_compare(
    records: Sequence[dict[str, Any]],
    comps: Sequence[Any],
    judgments: dict[str, str],
) -> dict[str, Any]:
    """计算人工判定与客观判定的一致率 + Cohen's κ。

    只统计人工明确给出 A/B/平 判定、且客观侧有 total 对比的用例。
    """
    obj: list[str] = []
    hum: list[str] = []
    detail: list[dict[str, Any]] = []

    for r in records:
        j = judgments.get(r["compare_id"])
        o = _objective_outcome(r["case_id"], comps)
        if j is None or o is None:
            continue
        h = _human_outcome(r, j)
        obj.append(o)
        hum.append(h)
        detail.append({
            "compare_id": r["compare_id"],
            "case_id": r["case_id"],
            "objective": o,
            "human": h,
            "agree": o == h,
        })

    n = len(obj)
    if n == 0:
        return {"n": 0, "agreement": None, "kappa": None, "detail": detail}

    agree = sum(o == h for o, h in zip(obj, hum))
    agreement = round(agree / n, 3)

    # Cohen's κ：扣除随机一致后的真实一致
    kappa = None
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = round(cohen_kappa_score(obj, hum), 3)
    except Exception:
        pass

    return {
        "n": n,
        "agreement": agreement,
        "kappa": kappa,
        "detail": detail,
    }
