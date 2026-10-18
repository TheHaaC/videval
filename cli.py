"""videval.cli — 命令行入口。

    python -m videval scan    --assets ./ai视频素材
    python -m videval metrics --assets ./ai视频素材
    python -m videval prefs   --assets ./ai视频素材
    python -m videval report  --assets ./ai视频素材
    python -m videval all     --assets ./ai视频素材
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import blind, calibrate, compare, design, face, meta, metrics as M, prefs as P, probe, report, scoring, stats

DEFAULT_ASSETS = "ai视频素材"
DEFAULT_OUT = "outputs"


# ---------------------------------------------------------------- 公共步骤


def load_cases(assets: str, limit: int | None = None):
    root = Path(assets)
    if not root.is_dir():
        print(f"素材目录不存在：{root}", file=sys.stderr)
        raise SystemExit(1)
    cases = probe.scan_assets(root)
    if limit:
        cases = cases[:limit]
    return cases


def ensure_out(out: str) -> Path:
    p = Path(out)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_metrics(
    cases,
    outdir: Path,
    sample_fps: float,
    work_width: int,
    workers: int | None,
    force: bool = False,
):
    """取得指标结果，优先复用已落盘的缓存。

    指标计算是整条链路里唯一的重活（96 个成片约 1 分钟）。
    报告、偏好数据等下游步骤都应复用同一份结果，
    既省时间，也保证各处口径一致——否则同一批素材
    在报告和偏好对里可能算出两套分。
    """
    import pandas as pd

    cache = outdir / "metrics_raw.csv"
    if cache.exists() and not force:
        df = pd.read_csv(cache)
        print(f"复用已缓存指标 {cache}（{len(df)} 条）")
        return df.to_dict("records")

    payloads = M.build_payloads(
        cases, sample_fps=sample_fps, work_width=work_width
    )
    print(
        f"待计算 {len(payloads)} 个成片"
        f"（采样 {sample_fps}fps，归一化宽 {work_width}px）"
    )
    rows = M.batch_compute(payloads, workers=workers)
    pd.DataFrame(rows).to_csv(cache, index=False, encoding="utf-8-sig")
    return rows


# ---------------------------------------------------------------- 子命令


def cmd_scan(args: argparse.Namespace) -> int:
    t0 = time.time()
    cases = load_cases(args.assets, args.limit)
    outdir = ensure_out(args.out)

    rows = meta.enrich_records(probe.to_records(cases))
    import pandas as pd

    df = pd.DataFrame(rows)
    csv_path = outdir / "manifest.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"扫描完成：{len(cases)} 条用例 / {len(rows)} 个成片  用时 {time.time()-t0:.1f}s")
    print(f"  资产清单 → {csv_path}")
    print()

    for seg in design.SEGMENTS:
        sub = df[df.case_id.isin(seg.case_ids)]
        print(f"  {seg.label}（用例 {seg.case_ids[0]}-{seg.case_ids[-1]}）  对手：{seg.opponent}")
        for model, g in sub.groupby("model"):
            res = sorted({f"{w}×{h}" for w, h in zip(g.width, g.height)})
            print(
                f"      {model:<14} {len(g):>2} 条 | "
                f"分辨率 {', '.join(res)} | "
                f"均时长 {g.duration.mean():.1f}s | "
                f"均码率 {g.bit_rate.mean()/1e6:.1f} Mbps | "
                f"含音轨 {int(g.has_audio.sum())}/{len(g)}"
            )
        print()

    # 数据质量提示
    empty = [c.case_id for c in cases if c.prompt_chars == 0]
    short = [c.case_id for c in cases if 0 < c.prompt_chars < 80]
    noref = [c.case_id for c in cases if not c.ref_images]
    if empty or short or noref:
        print("  数据质量提示：")
        if empty:
            print(f"      提示词为空的用例：{', '.join(empty)}（一致性指标将不可用）")
        if short:
            print(f"      提示词过短(<80字符)：{', '.join(short)}（指令遵循判定依据不足）")
        if noref:
            print(f"      无参考图的用例：{', '.join(noref)}（无法量化人物场景一致性）")
        print()

    # 生成执行元数据：时间错开是重要的混淆变量提示
    if "gen_time" in df.columns:
        t = pd.to_datetime(df["gen_time"], errors="coerce")
        known = t.notna()
        print(f"  生成元数据：{int(known.sum())}/{len(df)} 条可从文件名解析出生成时间")
        if known.any():
            for model, g in df.groupby("model"):
                gm = pd.to_datetime(g["gen_time"], errors="coerce").dropna()
                if not gm.empty:
                    span = f"{gm.min():%m-%d} ~ {gm.max():%m-%d}"
                    print(f"      {model:<14} {span}")
            rng = t.dropna()
            if rng.nunique() > 1 and (rng.max() - rng.min()).days > 3:
                print("      注意：不同模型成片的生成时间跨度超过 3 天，")
                print("      若平台在此期间更新版本，将构成混淆变量，报告需声明。")
        print()
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    import pandas as pd

    t0 = time.time()
    cases = load_cases(args.assets, args.limit)
    outdir = ensure_out(args.out)

    rows = get_metrics(
        cases, outdir, args.sample_fps,
        args.work_width, args.workers, force=args.force,
    )
    print()

    df = pd.DataFrame(rows)
    raw_path = outdir / "metrics_raw.csv"
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    scored = scoring.score_all(rows, preset=args.preset)
    flat = []
    for s in scored:
        r = {
            "case_id": s["case_id"],
            "model": s["model"],
            "duration": s["duration"],
            "resolution": f"{s['width']}×{s['height']}",
            "has_audio": s["has_audio"],
            "total": s["total"],
        }
        r.update({
            design.DIMENSION_BY_KEY[k].name: v
            for k, v in s["dims"].items()
        })
        flat.append(r)

    sdf = pd.DataFrame(flat)
    score_path = outdir / "scores.csv"
    sdf.to_csv(score_path, index=False, encoding="utf-8-sig")

    print(f"指标计算完成，用时 {time.time()-t0:.1f}s")
    print(f"  原始指标 → {raw_path}")
    print(f"  六维评分 → {score_path}")
    print()

    # 分段汇总
    for seg in design.SEGMENTS:
        print(f"  {seg.label}  对手：{seg.opponent}")
        seg_rows = [s for s in scored if str(s["case_id"]) in seg.case_ids]
        for model in (design.SUBJECT_MODEL, seg.opponent):
            rs = [s for s in seg_rows if s["model"] == model]
            if not rs:
                continue
            card = scoring.dimension_report_card(seg_rows, model)
            tot = sum(s["total"] for s in rs) / len(rs)
            detail = "  ".join(
                f"{design.DIMENSION_BY_KEY[d.key].name[:4]}{card.get(d.key, 0):.2f}"
                for d in design.DIMENSIONS
            )
            print(f"      {model:<14} 总分 {tot:.2f} | {detail}")
        print()
    return 0


def cmd_prefs(args: argparse.Namespace) -> int:
    import pandas as pd

    t0 = time.time()
    cases = load_cases(args.assets, args.limit)
    outdir = ensure_out(args.out)

    rows = get_metrics(
        cases, outdir, args.sample_fps,
        args.work_width, args.workers, force=args.force,
    )
    scored = scoring.score_all(rows, preset=args.preset)
    print()

    comps = P.build_preferences(
        scored, tie_margin=args.tie_margin, subject=design.SUBJECT_MODEL
    )

    # 偏好对：以用例为单位组织视频路径
    by_case: dict[str, dict[str, list[str]]] = {}
    for c in cases:
        d: dict[str, list[str]] = {}
        for v in c.videos:
            d.setdefault(v.model, []).append(v.path)
        by_case[c.case_id] = d

    prompt_by_case = {c.case_id: c.prompt for c in cases}

    records = P.to_dpo_records(
        comps,
        prompt_by_case=prompt_by_case,
        video_by=lambda cid, m: by_case.get(str(cid), {}).get(m, []),
        tie_margin=args.tie_margin,
    )

    jsonl_path = outdir / "preferences.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 比较明细
    cdf = pd.DataFrame([asdict_compat(c) for c in comps])
    comp_path = outdir / "comparisons.csv"
    cdf.to_csv(comp_path, index=False, encoding="utf-8-sig")

    print(f"配对比较完成，用时 {time.time()-t0:.1f}s")
    print(f"  比较明细   → {comp_path}  ({len(comps)} 条)")
    print(f"  偏好对     → {jsonl_path}  ({len(records)} 条)")
    print()

    for seg in design.SEGMENTS:
        print(f"  {seg.label}  {design.SUBJECT_MODEL} vs {seg.opponent}  "
              f"（n={len(seg.case_ids)} 配对用例，Wilcoxon 检验）")
        sig = stats.segment_significance(comps, seg.key)
        for t in sig:
            print(
                f"      {t.dimension_name:<10} "
                f"均分差 {t.mean_margin:+.3f}  "
                f"p={t.wilcoxon_p:.3f}  "
                f"95% CI {t.ci_str}  "
                f"→ {t.outcome}"
            )
        print()
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    t0 = time.time()
    cases = load_cases(args.assets, args.limit)
    outdir = ensure_out(args.out)

    rows = get_metrics(
        cases, outdir, args.sample_fps,
        args.work_width, args.workers, force=args.force,
    )
    scored = scoring.score_all(rows, preset=args.preset)
    comps = P.build_preferences(
        scored, tie_margin=args.tie_margin, subject=design.SUBJECT_MODEL
    )
    print()

    md_path = outdir / "评测报告.md"
    html_path = outdir / "评测看板.html"

    report.write_markdown(scored, comps, cases, md_path, preset=args.preset)
    report.write_html(scored, comps, cases, html_path, preset=args.preset)

    print(f"报告生成完成，用时 {time.time()-t0:.1f}s")
    print(f"  Markdown 报告 → {md_path}")
    print(f"  HTML 看板     → {html_path}")
    return 0


def cmd_blind(args: argparse.Namespace) -> int:
    """准备主观盲测：评分卡、匿名视频副本、打分表。"""
    t0 = time.time()
    cases = load_cases(args.assets, args.limit)
    outdir = ensure_out(args.out)

    rows = get_metrics(
        cases, outdir, args.sample_fps,
        args.work_width, args.workers, force=args.force,
    )
    scored = scoring.score_all(rows, preset=args.preset)
    print()

    rubric, sheet, bl = blind.prepare_blind(
        scored, outdir, n_per_model_per_seg=args.n_per_model_per_seg
    )

    meta_path = outdir / "blind_meta.json"
    meta_path.write_text(
        json.dumps(bl, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"盲测准备完成，用时 {time.time()-t0:.1f}s")
    print(f"  评分卡   → {rubric}")
    print(f"  打分表   → {sheet}  ({len(bl)} 条匿名样本)")
    print(f"  视频副本 → {outdir}/blind/")
    print(f"  元数据   → {meta_path}（校准用，打分前勿看）")
    print()
    print("  打分步骤：")
    print("    1. 打开「盲测评分卡.md」熟悉标准")
    print("    2. 打开 blind/ 目录，逐个播放视频（文件名已匿名，看不到模型）")
    print("    3. 在「盲测打分表.csv」里给六维各打 1–5 分")
    print("    4. 填完把 CSV 发回，我用人工分校准客观指标")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """校准客观指标。

    优先走「对比打分校准」（人工选优 vs 客观判定的一致率 + Cohen's κ）；
    若无对比材料，回退到「绝对打分校准」（Spearman 相关）。
    """
    t0 = time.time()
    cases = load_cases(args.assets, args.limit)
    outdir = ensure_out(args.out)

    rows = get_metrics(
        cases, outdir, args.sample_fps,
        args.work_width, args.workers, force=args.force,
    )
    scored = scoring.score_all(rows, preset=args.preset)
    print()

    compare_meta = outdir / "compare_meta.json"
    compare_sheet = outdir / "对比打分表.csv"
    if compare_meta.exists() and compare_sheet.exists():
        return _calibrate_compare(scored, outdir, args, t0)
    return _calibrate_absolute(scored, outdir, t0)


def _calibrate_compare(scored, outdir, args, t0) -> int:
    """对比打分校准：人工选优 vs 客观判定。"""
    import pandas as pd

    comps = P.build_preferences(
        scored, tie_margin=args.tie_margin, subject=design.SUBJECT_MODEL
    )
    records = json.loads(
        (outdir / "compare_meta.json").read_text(encoding="utf-8")
    )
    judgments = compare.load_judgments(outdir / "对比打分表.csv")

    result = compare.calibrate_compare(records, comps, judgments)

    pd.DataFrame(result["detail"]).to_csv(
        outdir / "对比校准结果.csv", index=False, encoding="utf-8-sig"
    )

    n_total = len(records)
    n = result["n"]
    print(f"对比校准完成，用时 {time.time()-t0:.1f}s")
    print(f"  有效判定 {n}/{n_total} 组")
    if n == 0:
        print("  尚无可用的对比判定，请先填写「对比打分表.csv」的判定列。")
        return 0
    print(f"  一致率   {result['agreement']:.1%}")
    k = result["kappa"]
    if k is not None:
        verdict = (
            "几乎完全一致" if k >= 0.8 else
            "高度一致" if k >= 0.6 else
            "中等一致" if k >= 0.4 else
            "较低一致" if k >= 0.2 else "≈随机（客观指标未模拟人的判断）"
        )
        print(f"  Cohen's κ {k:.3f}  → {verdict}")
    print()
    print("  不一致的用例（客观判定 vs 人工判定）：")
    for d in result["detail"]:
        if not d["agree"]:
            print(f"    用例 {d['case_id']:<3} 客观={d['objective']:<5} 人工={d['human']}")
    print(f"  明细 → {outdir / '对比校准结果.csv'}")
    return 0


def _calibrate_absolute(scored, outdir, t0) -> int:
    """绝对打分校准：客观分 vs 人工分的 Spearman 相关。"""
    import pandas as pd

    aligned, result = calibrate.run_calibration(scored, outdir)

    rows_out = []
    for human_dim, r in result.items():
        rows_out.append({
            "盲测维度": human_dim,
            "客观维度": r["obj_dim"],
            "样本数": r["n"],
            "Spearman_r": r["spearman_r"] if r["spearman_r"] is not None else "",
            "p": r["p"] if r["p"] is not None else "",
            "判定": r["verdict"],
            "建议": r["suggestion"],
        })
    pd.DataFrame(rows_out).to_csv(
        outdir / "校准结果.csv", index=False, encoding="utf-8-sig"
    )

    print(f"校准完成，用时 {time.time()-t0:.1f}s  对齐样本 {len(aligned)} 条")
    print(f"  校准结果 → {outdir / '校准结果.csv'}")
    print()
    print(f"  {'盲测维度':<12} {'n':>3} {'Spearman r':>10} {'p':>7}   判定")
    for human_dim, r in result.items():
        rv = f"{r['spearman_r']:.3f}" if r["spearman_r"] is not None else "  —  "
        pv = f"{r['p']:.3f}" if r["p"] is not None else "  —  "
        print(f"  {human_dim:<12} {r['n']:>3} {rv:>10} {pv:>7}   {r['verdict']}")
    print()
    print("  注意：以上是基于重跑客观分的校准；若某维度显示「无方差」，")
    print("  说明人工打分没拉开档次，需按评分卡重新打分后再校准。")
    return 0


def cmd_face(args: argparse.Namespace) -> int:
    """语义层：ArcFace 人脸一致性，补齐一致性维度的语义短板。"""
    import pandas as pd

    t0 = time.time()
    cases = load_cases(args.assets, args.limit)
    outdir = ensure_out(args.out)

    # 确保客观指标已就绪（人脸一致性结果要合并进去）
    rows = get_metrics(
        cases, outdir, args.sample_fps,
        args.work_width, args.workers, force=args.force,
    )

    # 只对「提示词 @引用了参考图」的用例做人脸比对
    payloads = []
    for c in cases:
        if not c.ref_images or not M.prompt_refs_image(c.prompt):
            continue
        for v in c.videos:
            payloads.append({
                "path": v.path,
                "ref_images": list(c.ref_images),
            })

    print(f"待计算人脸一致性的成片 {len(payloads)} 个")
    results = face.batch_face_sim(payloads, max_frames=args.max_frames)

    # 合并进指标缓存，供 scoring 使用
    fr = pd.DataFrame(results)
    raw_path = outdir / "metrics_raw.csv"
    raw = pd.read_csv(raw_path)
    raw = raw.merge(fr, on="path", how="left")
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

    face_path = outdir / "face_sim.csv"
    fr.to_csv(face_path, index=False, encoding="utf-8-sig")

    n_valid = int(fr["face_sim_mean"].notna().sum())
    print(f"人脸一致性计算完成，用时 {time.time()-t0:.1f}s")
    print(f"  有效样本 {n_valid}/{len(payloads)}（其余参考图/成片检不出人脸，视为不可评估）")
    print(f"  明细 → {face_path}，并已合并进 {raw_path}")
    if n_valid:
        vals = fr["face_sim_mean"].dropna()
        print(f"  face_sim 分布：min {vals.min():.3f} / 中位 {vals.median():.3f} / max {vals.max():.3f}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """生成人工对比打分材料：Rally vs 对手并排，选优。"""
    t0 = time.time()
    cases = load_cases(args.assets, args.limit)
    outdir = ensure_out(args.out)

    rows = get_metrics(
        cases, outdir, args.sample_fps,
        args.work_width, args.workers, force=args.force,
    )
    scored = scoring.score_all(rows, preset=args.preset)
    print()

    sheet, meta_path, records = compare.build_compare(scored, cases, outdir)

    print(f"对比打分材料生成完成，用时 {time.time()-t0:.1f}s")
    print(f"  打分表   → {sheet}  ({len(records)} 组对比)")
    print(f"  视频副本 → {outdir}/compare/")
    print(f"  映射     → {meta_path}（校准用，打分前勿看）")
    print()
    print("  打分步骤：")
    print("    1. 打开 compare/ 目录，每组并排看 A、B 两个视频")
    print("    2. 判断哪个更好（或平局），填进「对比打分表.csv」的判定列")
    print("    3. 填完发回，我用人工判定 vs 客观判定算一致率 + Cohen's κ")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    cmd_scan(args)
    cmd_metrics(args)
    cmd_prefs(args)
    cmd_report(args)
    return 0


def asdict_compat(obj: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(obj)


# ---------------------------------------------------------------- 入口


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="videval",
        description="视频生成模型批量评测工具链",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--assets", default=DEFAULT_ASSETS, help="素材根目录")
        sp.add_argument("--out", default=DEFAULT_OUT, help="输出目录")
        sp.add_argument("--limit", type=int, default=None, help="只处理前 N 条用例")
        sp.add_argument(
            "--preset", default=design.DEFAULT_PRESET,
            choices=list(design.WEIGHT_PRESETS),
            help="权重预设",
        )
        sp.add_argument("--sample-fps", type=float, default=12.0)
        sp.add_argument("--work-width", type=int, default=256)
        sp.add_argument("--workers", type=int, default=None)
        sp.add_argument("--tie-margin", type=float, default=0.25)
        sp.add_argument("--force", action="store_true", help="忽略已缓存的指标，强制重算")
        sp.add_argument(
            "--n-per-model-per-seg", type=int, default=6,
            help="盲测每段每模型抽样数（默认 6，两段两模型共 24 条）",
        )
        sp.add_argument(
            "--max-frames", type=int, default=15,
            help="人脸一致性每视频最多抽帧数（默认 15）",
        )

    for name, fn, helptext in (
        ("scan", cmd_scan, "扫描素材，生成资产清单"),
        ("metrics", cmd_metrics, "计算客观指标与六维评分"),
        ("face", cmd_face, "语义层：ArcFace 人脸一致性"),
        ("prefs", cmd_prefs, "配对对比 + 显著性检验 + 偏好数据"),
        ("blind", cmd_blind, "准备主观盲测：评分卡 + 匿名打分表"),
        ("compare", cmd_compare, "生成人工对比打分材料（选优）"),
        ("calibrate", cmd_calibrate, "校准客观指标（对比一致率 / Spearman）"),
        ("report", cmd_report, "导出 Markdown 与 HTML 报告"),
        ("all", cmd_all, "依次执行以上全部步骤"),
    ):
        sp = sub.add_parser(name, help=helptext)
        common(sp)
        sp.set_defaults(func=fn)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
