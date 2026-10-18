"""videval.meta — 从文件名提取生成执行的残留元数据。

商业视频生成平台不暴露随机种子、也不记录当次参数，这使「生成执行」
环节成为黑盒。但文件命名里往往残留两类信息：

    1. 生成时间戳 —— 毫秒级 epoch、ISO 8601、截图工具时间戳等；
    2. 重试序号   —— 文件名后缀 "(1)" / "-1" 标记同提示词的第几次采样。

前者可用来追溯生成批次、检验「同一用例多次采样是否同批跑出」，
后者可用来核对「每用例到底采样了几次」。两者虽不能替代 seed 控制，
但能把黑盒里能拿到的信息尽量拿干净，并在报告里如实声明局限。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

__all__ = ["extract_gen_meta", "enrich_records"]


#: 各类时间戳的识别模式（按优先级）
_TS_PATTERNS: list[tuple[str, str]] = [
    # (正则, 来源标签)
    (r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z)", "iso8601"),
    (r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", "snipaste"),
    (r"_(\d{13})\.", "epoch_ms"),
]


def _parse_ts(value: str, fmt: str) -> datetime | None:
    try:
        return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def extract_gen_meta(filename: str) -> dict[str, Any]:
    """解析单个文件名，返回生成时间与重试序号。

    识别不到时返回 gen_time=None, retry_idx=0，不抛异常——
    元数据是「能拿多少拿多少」，拿不到不应阻断主流程。
    """
    stem = filename.rsplit(".", 1)[0]

    gen_time: datetime | None = None
    source = "unknown"

    # 1. ISO 8601（如 ai-generated-2026-10-08T07-23-10-837Z.mp4）
    m = re.search(_TS_PATTERNS[0][0], filename)
    if m:
        gen_time = _parse_ts(m.group(1), "%Y-%m-%dT%H-%M-%S-%fZ")
        source = "iso8601"

    # 2. 截图工具时间戳（如 Snipaste_2026-10-09_15-10-43）
    if gen_time is None:
        m = re.search(_TS_PATTERNS[1][0], filename)
        if m:
            gen_time = _parse_ts(
                f"{m.group(1)}-{m.group(2)}", "%Y-%m-%d-%H-%M-%S"
            )
            source = "snipaste"

    # 3. 13 位毫秒 epoch（如 vid_xxx_1786424781245.mp4）
    if gen_time is None:
        m = re.search(_TS_PATTERNS[2][0], filename)
        if m:
            ms = int(m.group(1))
            gen_time = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
            source = "epoch_ms"

    # 重试序号：优先 " (1)" / "(1)" 括号形式，其次 "-1.mp4" 结尾形式。
    # 注意 "-1.mp4" 只匹配紧跟扩展名的数字，不会误吞 ISO 时间戳里的毫秒。
    retry_idx = 0
    m = re.search(r"\((\d+)\)", stem)
    if m:
        retry_idx = int(m.group(1))
    else:
        m = re.search(r"-(\d+)\.", filename)
        if m and source == "iso8601":
            retry_idx = int(m.group(1))

    return {
        "gen_time": gen_time.isoformat() if gen_time else None,
        "retry_idx": retry_idx,
        "source": source,
    }


def enrich_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给资产记录补充生成元数据列。

    rows 来自 probe.to_records，需含 "path" 字段；逐行解析文件名后
    追加 gen_time / retry_idx / gen_source 三列。
    """
    import os

    out: list[dict[str, Any]] = []
    for r in rows:
        base = os.path.basename(str(r.get("path", "")))
        meta = extract_gen_meta(base)
        r2 = dict(r)
        r2["gen_time"] = meta["gen_time"]
        r2["retry_idx"] = meta["retry_idx"]
        r2["gen_source"] = meta["source"]
        out.append(r2)
    return out
