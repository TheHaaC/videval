"""videval.report — 导出 Markdown 横评报告与 HTML 看板。

两份产物面向不同读者：
    Markdown —— 给工程师和面试官，可进版本库、可 diff、可贴进文档；
    HTML     —— 给决策者，一眼看出「哪个模型在哪些维度上赢」。

所有图表用内联 SVG 手绘，不引外部库，无外网依赖，邮件转发也不掉样式。
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Sequence

from . import design
from .design import DIMENSIONS, SEGMENTS, SUBJECT_MODEL
from . import prefs as P
from . import stats

__all__ = ["write_markdown", "write_html", "radar_svg"]


# ---------------------------------------------------------------- 工具


def _human_comparison(outdir: Path) -> dict[str, Any] | None:
    """读人工对比打分结果，返回 Rally 胜平负统计。

    这是当前评测里最可靠的主观结论——对比选优有参照物，
    比客观指标更能反映「人觉得哪个更好」。
    """
    from . import compare as _cmp

    meta_path = outdir / "compare_meta.json"
    sheet_path = outdir / "对比打分表.csv"
    if not meta_path.exists() or not sheet_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        judgments = _cmp.load_judgments(sheet_path)
    except Exception:
        return None

    win = tie = lose = 0
    for r in meta:
        j = judgments.get(r["compare_id"])
        if j is None:
            continue
        if j == "平":
            tie += 1
            continue
        chosen = r["model_a"] if j == "A" else r["model_b"]
        if chosen == SUBJECT_MODEL:
            win += 1
        else:
            lose += 1

    n = win + tie + lose
    return {"win": win, "tie": tie, "lose": lose, "n": n} if n else None


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _short(path: str, keep: int = 2) -> str:
    p = Path(path)
    parts = p.parts[-keep:]
    return "/".join(parts)


def _card(rows, model: str) -> dict[str, float]:
    from . import scoring as S

    return S.dimension_report_card(rows, model)


# ---------------------------------------------------------------- SVG 雷达图


def radar_svg(
    labels: Sequence[str],
    series: dict[str, list[float]],
    size: int = 420,
    maxv: float = 5.0,
    colors: dict[str, str] | None = None,
) -> str:
    """手绘雷达图，返回内联 SVG 字符串。"""
    import math

    palette = colors or {
        SUBJECT_MODEL: "#c0392b",
    }
    fallback = ["#2980b9", "#27ae60", "#8e44ad", "#d35400"]
    for i, name in enumerate(series):
        palette.setdefault(name, fallback[i % len(fallback)])

    cx = cy = size / 2
    r = size * 0.34
    n = len(labels)

    def pt(i: int, val: float) -> tuple[float, float]:
        ang = -math.pi / 2 + 2 * math.pi * i / n
        rr = r * (max(0.0, min(val, maxv)) / maxv)
        return cx + rr * math.cos(ang), cy + rr * math.sin(ang)

    parts: list[str] = [
        f'<svg viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">'
    ]

    # 网格
    for g in range(1, 6):
        pts = " ".join(
            f"{x:.1f},{y:.1f}"
            for x, y in (pt(i, maxv * g / 5) for i in range(n))
        )
        parts.append(
            f'<polygon points="{pts}" fill="none" stroke="#e3e6ea" stroke-width="1"/>'
        )

    # 轴线与标签
    for i, lab in enumerate(labels):
        x, y = pt(i, maxv)
        parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" '
            f'stroke="#e3e6ea" stroke-width="1"/>'
        )
        lx, ly = pt(i, maxv * 1.22)
        anchor = "middle"
        if lx > cx + 4:
            anchor = "start"
        elif lx < cx - 4:
            anchor = "end"
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="12" fill="#5b6570" '
            f'text-anchor="{anchor}" dominant-baseline="middle">{html.escape(lab)}</text>'
        )

    # 数据系列
    for name, vals in series.items():
        pts = " ".join(
            f"{x:.1f},{y:.1f}"
            for x, y in (pt(i, vals[i]) for i in range(n))
        )
        c = palette.get(name, "#666")
        parts.append(
            f'<polygon points="{pts}" fill="{c}" fill-opacity="0.16" '
            f'stroke="{c}" stroke-width="2"/>'
        )
        for i in range(n):
            x, y = pt(i, vals[i])
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{c}"/>'
            )

    parts.append("</svg>")
    return "".join(parts)


def _legend(series: dict[str, list[float]], colors: dict[str, str]) -> str:
    items = []
    for name in series:
        c = colors.get(name, "#666")
        items.append(
            f'<span class="lg"><i style="background:{c}"></i>'
            f"{html.escape(name)}</span>"
        )
    return f'<div class="legend">{"".join(items)}</div>'


# ---------------------------------------------------------------- Markdown


def write_markdown(
    scored: Sequence[dict[str, Any]],
    comps: Sequence[P.Comparison],
    cases: Sequence[Any],
    path: str | Path,
    preset: str = design.DEFAULT_PRESET,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    L: list[str] = []
    add = L.append

    total_videos = len(scored)
    total_dur = sum(s.get("duration", 0.0) for s in scored)

    add("# 视频生成模型横向评测报告")
    add("")
    add(f"> 评测对象：**{SUBJECT_MODEL}**　|　"
        f"权重预设：`{preset}`　|　"
        f"样本：{total_videos} 个成片 / {total_dur/60:.1f} 分钟")
    add("")

    # ---------- 实验设计 ----------
    add("## 一、实验设计")
    add("")
    add("采用**配对对比**：同一条提示词、同一套参考资产，仅模型不同，"
        "从而消除提示词难度这一混淆变量。")
    add("")
    add("| 分段 | 用例编号 | 对比对象 | 说明 |")
    add("|---|---|---|---|")
    for seg in SEGMENTS:
        add(f"| {seg.label} | {seg.case_ids[0]}–{seg.case_ids[-1]} | "
            f"{SUBJECT_MODEL} vs {seg.opponent} | {seg.note} |")
    add("")

    n_ref = sum(len(c.ref_images) for c in cases)
    n_aud = sum(len(c.ref_audios) for c in cases)
    add(f"素材规模：{len(cases)} 条用例、{total_videos} 个成片、"
        f"{n_ref} 张参考图、{n_aud} 段参考音频。")
    add("")

    # ---------- 核心结论（主观对比） ----------
    human = _human_comparison(Path(path).parent)
    if human:
        add("## 二、核心结论（主观对比评测）")
        add("")
        add(f"由执行导演对 {human['n']} 个用例做**并排对比选优**"
            f"（同一提示词、同一参考资产，仅模型不同，匿名判断）：")
        add("")
        add(f"- {SUBJECT_MODEL} 胜 **{human['win']}** 场、"
            f"平 {human['tie']} 场、负 **{human['lose']}** 场；")
        win_rate = human["win"] / max(1, human["win"] + human["lose"])
        add(f"- 去平局后胜率 **{win_rate:.0%}**，"
            f"**{SUBJECT_MODEL} 在整体生成质量上明显落后于对手**。")
        add("")
        add("> 这是本评测最可信的结论。对比选优有参照物，"
            "比客观指标更贴近人对视频整体质量的判断。")
        add("")

    # ---------- 客观指标结果 ----------
    add("## 三、客观指标结果")
    add("")
    add("> 客观指标用于**缺陷归因与技术质量体检**，而非「谁更好」的裁判。"
        "人工对比校准显示其加权总分与人的整体判断仅 ≈随机（Cohen's κ=0.139），"
        "故本节的胜负只描述单项技术质量差异，不代表整体优劣。")
    add("")

    for seg in SEGMENTS:
        seg_rows = [s for s in scored if str(s["case_id"]) in seg.case_ids]
        subj = [s for s in seg_rows if s["model"] == SUBJECT_MODEL]
        oppo = [s for s in seg_rows if s["model"] == seg.opponent]

        add(f"### {seg.label}：{SUBJECT_MODEL} vs {seg.opponent}")
        add("")
        add(f"每维度按 {len(seg.case_ids)} 个配对用例的分差做 Wilcoxon 符号秩检验，"
            "并用 bootstrap 估计 95% 置信区间。")
        add("")
        add("| 维度 | "
            f"{SUBJECT_MODEL} | {seg.opponent} | 分差 | p 值 | 95% CI | 判定 |")
        add("|---|---|---|---|---|---|---|")

        sig = {
            t.dimension: t
            for t in stats.segment_significance(comps, seg.key)
        }
        card_s = _card(seg_rows, SUBJECT_MODEL)
        card_o = _card(seg_rows, seg.opponent)

        for d in DIMENSIONS:
            t = sig.get(d.key)
            if not t:
                continue
            sv, ov = card_s.get(d.key, 0.0), card_o.get(d.key, 0.0)
            pstr = f"{t.wilcoxon_p:.3f}" if t.n else "—"
            add(f"| {d.name} | {sv:.2f} | {ov:.2f} | "
                f"{t.mean_margin:+.2f} | {pstr} | {t.ci_str} | {t.outcome} |")

        t = sig.get("total")
        if t:
            st = sum(s["total"] for s in subj) / max(1, len(subj))
            ot = sum(s["total"] for s in oppo) / max(1, len(oppo))
            pstr = f"{t.wilcoxon_p:.3f}" if t.n else "—"
            add(f"| **加权总分** | **{st:.2f}** | **{ot:.2f}** | "
                f"**{t.mean_margin:+.2f}** | **{pstr}** | "
                f"**{t.ci_str}** | **{t.outcome}** |")
        add("")

    # ---------- 逐用例明细 ----------
    add("## 四、逐用例明细")
    add("")
    for seg in SEGMENTS:
        add(f"### {seg.label}")
        add("")
        add("| 用例 | 提示词字数 | 参考图 | "
            f"{SUBJECT_MODEL} 总分 | {seg.opponent} 总分 | 分差 |")
        add("|---|---|---|---|---|---|")
        by_case = {c.case_id: c for c in cases}
        for cid in seg.case_ids:
            c = by_case.get(cid)
            rows = [s for s in scored if str(s["case_id"]) == cid]
            s_rows = [r for r in rows if r["model"] == SUBJECT_MODEL]
            o_rows = [r for r in rows if r["model"] == seg.opponent]
            if not s_rows or not o_rows:
                continue
            st = sum(r["total"] for r in s_rows) / len(s_rows)
            ot = sum(r["total"] for r in o_rows) / len(o_rows)
            nref = len(c.ref_images) if c else 0
            pchars = c.prompt_chars if c else 0
            add(f"| {cid} | {pchars} | {nref} | "
                f"{st:.2f} | {ot:.2f} | {st-ot:+.2f} |")
        add("")

    # ---------- 缺陷归因 ----------
    add("## 五、缺陷归因")
    add("")
    add("按客观指标的绝对阈值直接定位问题样本，不依赖人工标注：")
    add("")

    # (缺陷名, 指标, 方向, 阈值, 说明)
    #   方向 high：值越高越糟（如静止帧占比）
    #   方向 low ：值越低越糟（如清晰度）
    defect_rules = [
        ("画面凝固", "static_ratio", "high", 0.15, "静止帧占比过高"),
        ("运动抽搐", "motion_outlier_ratio", "high", 0.20, "运动突变帧过多"),
        ("清晰度不足", "sharpness_mean", "low", 450.0, "拉普拉斯方差偏低"),
        ("亮部过曝", "overexp_ratio", "high", 0.010, "高光溢出"),
        ("暗部死黑", "crushed_ratio", "high", 0.18, "暗部细节丢失"),
        ("参考一致性弱", "ref_sim_mean", "low", 0.30, "与参考图偏离较大"),
    ]

    any_hit = False
    for name, key, direction, thr, desc in defect_rules:
        hits = []
        for s in scored:
            v = (s.get("raw") or {}).get(key)
            if v is None:
                continue
            bad = (v > thr) if direction == "high" else (v < thr)
            if bad:
                hits.append((v, s))
        if not hits:
            continue
        any_hit = True
        # 严重程度高的排前面
        hits.sort(key=lambda t: t[0], reverse=(direction == "high"))
        add(f"- **{name}**（{desc}，`{key}` 阈值 {thr}）：{len(hits)} 条命中")
        for v, s in hits[:3]:
            add(f"    - `{_short(s['path'], 3)}` — {s['model']}，{key}={_fmt(v)}")
    if not any_hit:
        add("- 本批样本未触发任何硬性缺陷阈值。")
    add("")

    # ---------- 方法论 ----------
    add("## 六、方法论与局限")
    add("")
    add("**自动化部分**——分辨率归一化后计算像素级与直方图级指标，"
        "全部基于 CPU 经典算法，跨机器可复现。")
    add("")
    add("**客观指标的方法论定位**——人工对比校准表明，客观指标的加权总分"
        "与人对视频整体质量的判断仅 ≈随机（Cohen's κ=0.139），故客观指标"
        "只能用于缺陷归因与技术质量体检，不能替代人的整体判断。"
        "「谁更好」以人工对比选优为准。")
    add("")
    add("**需要人工复核的部分**——语义正确性、叙事逻辑、审美水准"
        "属于主观层，客观指标只能间接反映，须由多人盲测补齐。")
    add("")
    add("**已知局限**：")
    add("")
    add("1. 一致性指标依赖参考图，无参考图的用例该项不可用；")
    add("2. 切镜检测基于帧间差分突变，相邻镜头画面相近时会漏检；")
    add("3. 感知哈希与直方图对形变、手部崩坏等局部缺陷不敏感，"
        "这类问题仍需人工或语义模型识别；")
    add("4. 本评测基于商业平台黑盒生成，无法控制随机种子，"
        "以每用例多次采样（2 次）替代 seed 控制，方差估计偏保守；")
    add("5. 各模型成片生成时间错开（详见资产清单），"
        "若平台在采样期间更新版本，将构成未被控制的混淆变量；")
    add("6. 每分段仅 12 个配对用例，显著性检验功效有限，"
        "结论应视为阶段性的、待更大样本复核的观察；")
    add("7. 语义层仅覆盖人脸一致性（ArcFace），口型同步、图文一致性等"
        "尚未实现；要模拟人的整体判断，需引入多模态大模型打分。")
    add("")

    p.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------- HTML


_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--bd:#e3e6ea;--tx:#1f2933;--mut:#6b7684;--sub:#0b6b5e}
*{box-sizing:border-box}
body{margin:0;padding:28px;background:var(--bg);color:var(--tx);
 font:14px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,
 "PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:24px;margin:0 0 6px}
h2{font-size:17px;margin:30px 0 12px;padding-left:9px;border-left:3px solid var(--sub)}
h3{font-size:15px;margin:18px 0 8px;color:#374151}
.sub{color:var(--mut);font-size:13px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;
 padding:16px 18px;margin-bottom:14px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:7px 9px;text-align:right;border-bottom:1px solid var(--bd)}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600;font-size:12px;letter-spacing:.2px}
tr:last-child td{border-bottom:none}
.win{color:#c0392b;font-weight:600}
.lose{color:#27ae60;font-weight:600}
.tie{color:#6b7684}
.legend{margin-top:8px;font-size:12px;color:var(--mut)}
.legend .lg{margin-right:14px}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;
 margin-right:5px;vertical-align:middle}
.note{font-size:12px;color:var(--mut);margin-top:10px}
ul{margin:8px 0;padding-left:20px}
li{margin:3px 0}
code{background:#eef1f4;padding:1px 5px;border-radius:3px;font-size:12px}
"""


def write_html(
    scored: Sequence[dict[str, Any]],
    comps: Sequence[P.Comparison],
    cases: Sequence[Any],
    path: str | Path,
    preset: str = design.DEFAULT_PRESET,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    total_videos = len(scored)
    total_dur = sum(s.get("duration", 0.0) for s in scored)
    names = [d.name for d in DIMENSIONS]

    H: list[str] = []
    a = H.append

    a("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>")
    a("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    a("<title>视频生成模型横向评测看板</title>")
    a(f"<style>{_CSS}</style></head><body><div class='wrap'>")

    a("<h1>视频生成模型横向评测看板</h1>")
    a(f"<div class='sub'>评测对象 <b>{html.escape(SUBJECT_MODEL)}</b>　·　"
      f"权重预设 <code>{html.escape(preset)}</code>　·　"
      f"{total_videos} 个成片 / {total_dur/60:.1f} 分钟</div>")

    human = _human_comparison(Path(path).parent)
    if human:
        a(
            "<div class='card' style='border-left:4px solid #c0392b'>"
            "<b>核心结论（主观对比）</b>　"
            f"执行导演并排选优 {human['n']} 组："
            f"{html.escape(SUBJECT_MODEL)} 胜 {human['win']}、"
            f"负 {human['lose']}、平 {human['tie']}，"
            f"整体明显落后于对手。"
            "</div>"
        )
        a(
            "<div class='card' style='border-left:4px solid #854F0B'>"
            "<b>客观指标定位</b>　下方雷达图为客观指标，仅用于"
            "缺陷归因与技术质量体检，不代表整体优劣（人工对比校准 κ=0.139）。"
            "</div>"
        )

    for seg in SEGMENTS:
        seg_rows = [s for s in scored if str(s["case_id"]) in seg.case_ids]
        card_s = _card(seg_rows, SUBJECT_MODEL)
        card_o = _card(seg_rows, seg.opponent)

        series = {
            SUBJECT_MODEL: [card_s.get(d.key, 0.0) for d in DIMENSIONS],
            seg.opponent: [card_o.get(d.key, 0.0) for d in DIMENSIONS],
        }
        colors = {SUBJECT_MODEL: "#c0392b", seg.opponent: "#2980b9"}

        a(f"<h2>{html.escape(seg.label)}："
          f"{html.escape(SUBJECT_MODEL)} vs {html.escape(seg.opponent)}</h2>")

        a("<div class='card'><div class='grid'>")
        a("<div>" + radar_svg(names, series, 420, 5.0, colors) + "</div>")
        a("<div>")

        aggs = P.aggregate(comps, segment=seg.key)
        a("<table><thead><tr><th>维度</th>"
          f"<th>{html.escape(SUBJECT_MODEL)}</th>"
          f"<th>{html.escape(seg.opponent)}</th>"
          "<th>分差</th><th>胜/平/负</th></tr></thead><tbody>")
        for d in DIMENSIONS:
            ag = next((x for x in aggs if x.dimension == d.key), None)
            if not ag:
                continue
            cls = {"win": "win", "lose": "lose", "tie": "tie"}[ag.outcome]
            sv = card_s.get(d.key, 0.0)
            ov = card_o.get(d.key, 0.0)
            a(f"<tr><td>{html.escape(d.name)}</td>"
              f"<td>{sv:.2f}</td><td>{ov:.2f}</td>"
              f"<td class='{cls}'>{ag.mean_margin:+.2f}</td>"
              f"<td>{ag.win}:{ag.tie}:{ag.lose}</td></tr>")
        a("</tbody></table>")

        a(_legend(series, colors))
        a("</div></div></div>")
        a("</div>")

    a("</div></body></html>")
    p.write_text("\n".join(H), encoding="utf-8")
