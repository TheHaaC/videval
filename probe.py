"""videval.probe — 素材资产扫描与元数据提取。

一次遍历把「用例 → 提示词 → 参考资产 → 各模型成片」的关系织成一张表，
后续所有指标计算都基于这张表调度，避免重复 IO。

只调用 ffprobe 读取容器头信息，不解码任何视频帧，因此扫 96 个成片
通常在秒级完成。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}

# 目录名 → 展示用模型名。新增模型只需在此登记，无需改动其它模块。
MODEL_ALIASES: dict[str, str] = {
    "seedance2_0": "Seedance 2.0",
    "seedance2_5": "Seedance 2.5",
    "rally_video": "Rally",
}


class ProbeError(RuntimeError):
    """ffprobe 调用失败或返回结构异常时抛出。"""


def _require_ffprobe() -> str:
    exe = shutil.which("ffprobe")
    if exe is None:
        raise ProbeError(
            "未检测到 ffprobe，请先安装：brew install ffmpeg"
        )
    return exe


@dataclass
class VideoMeta:
    """单个成片的容器级元数据。"""

    path: str
    model: str
    case_id: str
    width: int
    height: int
    fps: float
    nb_frames: int
    duration: float
    bit_rate: int
    size_bytes: int
    vcodec: str
    has_audio: bool
    audio_codec: str | None = None
    audio_duration: float | None = None

    @property
    def megapixels(self) -> float:
        return round(self.width * self.height / 1_000_000, 2)


@dataclass
class CaseAsset:
    """一条评测用例的完整资产。"""

    case_id: str
    prompt: str = ""
    prompt_chars: int = 0
    ref_images: list[str] = field(default_factory=list)
    ref_audios: list[str] = field(default_factory=list)
    videos: list[VideoMeta] = field(default_factory=list)

    @property
    def models(self) -> list[str]:
        return sorted({v.model for v in self.videos})


def _ffprobe(exe: str, path: Path) -> dict[str, Any]:
    """调用 ffprobe 返回 JSON，失败时给出可读错误。"""
    cmd = [
        exe, "-v", "error",
        "-show_streams", "-show_format",
        "-of", "json", str(path),
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover
        raise ProbeError(f"ffprobe 超时：{path}") from exc

    if out.returncode != 0:
        raise ProbeError(f"ffprobe 失败（{path}）：{out.stderr.strip()}")
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise ProbeError(f"ffprobe 输出非 JSON（{path}）") from exc


def _parse_rate(rate: str | None) -> float:
    """把 '24/1' 这类有理数帧率转成 float，异常时回退 0。"""
    if not rate:
        return 0.0
    if "/" in rate:
        num, _, den = rate.partition("/")
        try:
            return round(float(num) / float(den), 3) if float(den) else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(rate)
    except ValueError:
        return 0.0


def probe_video(path: Path, model: str, case_id: str) -> VideoMeta:
    """提取单个视频文件的容器级元数据。"""
    exe = _require_ffprobe()
    info = _ffprobe(exe, path)

    streams = info.get("streams", [])
    fmt = info.get("format", {})

    vstream = next(
        (s for s in streams if s.get("codec_type") == "video"), None
    )
    if vstream is None:
        raise ProbeError(f"无视频流：{path}")

    astream = next(
        (s for s in streams if s.get("codec_type") == "audio"), None
    )

    nb_frames = vstream.get("nb_frames")
    if not nb_frames or nb_frames == "N/A":
        fps = _parse_rate(vstream.get("r_frame_rate"))
        dur = float(fmt.get("duration") or vstream.get("duration") or 0)
        nb_frames = int(round(fps * dur)) if fps else 0

    return VideoMeta(
        path=str(path),
        model=model,
        case_id=case_id,
        width=int(vstream.get("width", 0)),
        height=int(vstream.get("height", 0)),
        fps=_parse_rate(vstream.get("r_frame_rate")),
        nb_frames=int(nb_frames),
        duration=float(fmt.get("duration") or vstream.get("duration") or 0),
        bit_rate=int(fmt.get("bit_rate") or 0),
        size_bytes=int(fmt.get("size") or path.stat().st_size),
        vcodec=vstream.get("codec_name", "unknown"),
        has_audio=astream is not None,
        audio_codec=astream.get("codec_name") if astream else None,
        audio_duration=(
            float(astream.get("duration"))
            if astream and astream.get("duration") else None
        ),
    )


def _read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return ""


def scan_assets(assets_root: Path | str) -> list[CaseAsset]:
    """扫描素材根目录，返回按用例编号排序的资产列表。

    期望结构：
        <root>/prompt/<case_id>/p.txt          提示词
        <root>/prompt/<case_id>/*.(png|jpg)    参考图
        <root>/prompt/<case_id>/*.(mp3|wav)    参考音频
        <root>/<model>/<case_id>/*.mp4         该模型在该用例下的成片
    """
    root = Path(assets_root)
    if not root.is_dir():
        raise ProbeError(f"素材目录不存在：{root}")

    prompt_root = root / "prompt"
    cases: dict[str, CaseAsset] = {}

    # --- 用例侧：提示词与参考资产 ---
    if prompt_root.is_dir():
        for case_dir in sorted(prompt_root.iterdir()):
            if not case_dir.is_dir():
                continue
            cid = case_dir.name
            case = cases.setdefault(cid, CaseAsset(case_id=cid))

            pfile = case_dir / "p.txt"
            if pfile.is_file():
                case.prompt = _read_text(pfile).strip()
                case.prompt_chars = len(case.prompt)

            for f in sorted(case_dir.rglob("*")):
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                if ext in IMAGE_EXTS:
                    case.ref_images.append(str(f))
                elif ext in AUDIO_EXTS:
                    case.ref_audios.append(str(f))

    # --- 模型侧：成片 ---
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "prompt":
            continue
        model_name = MODEL_ALIASES.get(model_dir.name, model_dir.name)

        for case_dir in sorted(model_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            cid = case_dir.name
            case = cases.setdefault(cid, CaseAsset(case_id=cid))

            for vf in sorted(case_dir.rglob("*")):
                if vf.is_file() and vf.suffix.lower() in VIDEO_EXTS:
                    case.videos.append(probe_video(vf, model_name, cid))

    def sort_key(cid: str) -> tuple[int, str]:
        return (int(cid), "") if cid.isdigit() else (10**9, cid)

    return [cases[k] for k in sorted(cases, key=sort_key)]


def to_records(cases: Iterable[CaseAsset]) -> list[dict[str, Any]]:
    """把资产列表摊平成 DataFrame 友好的记录行（一视频一行）。"""
    rows: list[dict[str, Any]] = []
    for case in cases:
        for v in case.videos:
            row = asdict(v)
            row["prompt_chars"] = case.prompt_chars
            row["n_ref_images"] = len(case.ref_images)
            row["n_ref_audios"] = len(case.ref_audios)
            rows.append(row)
    return rows
