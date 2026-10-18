"""videval.metrics — 视频生成质量的客观指标计算。

全部基于 CPU 的经典图像算法，不依赖任何推理框架，
因此在笔记本上即可全量复现，结果跨机器一致。

指标分四组，对应六维体系中的可自动化部分：

    时序运动    motion_* / static_ratio    → 运动合理性
    单帧质量    sharpness_* / exposure_*   → 单帧画面正确性
    镜头结构    shot_cut_* / cut_intent_*  → 镜头衔接流畅性 + 指令遵循
    光影色彩    contrast / saturation / colorfulness → 光影色彩和谐性
    参考一致性  ref_sim_* / subject_drift  → 人物场景一致性

参考一致性说明
--------------
图生视频的提示词会 @引用参考图（角色、场景、道具）。
把成片各帧与这些参考图做 HSV 直方图相关性 + 感知哈希比对，
即可在没有 GPU 语义模型的情况下，量化「像不像、稳不稳」。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

__all__ = [
    "VideoMetrics",
    "compute_video_metrics",
    "batch_compute",
    "build_payloads",
    "count_shot_intents",
    "detect_shot_cuts",
    "hamming",
    "phash",
]


# ---------------------------------------------------------------- 基础工具


def phash(gray: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """DCT 感知哈希。对缩放、压缩、轻度色彩变化稳健。"""
    factor = 4
    size = hash_size * factor
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))
    low = dct[:hash_size, :hash_size]
    return (low > np.median(low)).flatten()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _resize_keep_aspect(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    nh = max(1, int(round(h * width / w)))
    return cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)


def _hsv_signature(bgr: np.ndarray, grid: int = 3) -> np.ndarray:
    """空间分块 HSV 直方图签名。

    把画面切成 grid×grid 的网格、每块各算一份 H-S 联合直方图再拼接。
    相比全图直方图，它对「内容搬到别处」这类变化敏感得多——
    实测全图直方图相关性在相邻帧上恒为 1.000，完全丧失区分度，
    而分块签名能拉开到 0.7~0.99 的可用区间。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    parts: list[np.ndarray] = []
    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = h * gy // grid, h * (gy + 1) // grid
            x0, x1 = w * gx // grid, w * (gx + 1) // grid
            tile = hsv[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            hist = cv2.calcHist(
                [tile], [0, 1], None, [8, 8], [0, 180, 0, 256]
            )
            # 注意：NORM_L1 下 alpha 是「目标 L1 范数」而非取值下界
            # （NORM_MINMAX 才是下界）。这里要让每块直方图和为 1，
            # 故 alpha=1.0；误写为 0 会把整张直方图清零，
            # 进而让所有相关性退化成 1.0，一致性指标彻底失效。
            cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
            parts.append(hist.flatten())
    return np.concatenate(parts) if parts else np.zeros(1)


def _hist_corr(a: np.ndarray, b: np.ndarray) -> float:
    """直方图相关性，对空/常量直方图做兜底。"""
    if a.size != b.size or a.size == 0:
        return 0.0
    if a.std() < 1e-9 or b.std() < 1e-9:
        # 退化分布：相关性无定义，退化为 L1 距离相似度
        return float(1.0 - np.abs(a - b).sum() / 2.0)
    return float(
        cv2.compareHist(
            a.astype(np.float32), b.astype(np.float32),
            cv2.HISTCMP_CORREL,
        )
    )


def detect_shot_cuts(
    motion: np.ndarray,
    spike_factor: float = 4.0,
    isolation: float = 3.0,
) -> list[int]:
    """从帧间差分序列中识别镜头切点。

    难点在于把「切镜」和「激烈运动」分开：

        切镜   —— 孤立单帧突变，前一帧和后一帧都正常，只有交界处崩一下；
        运动段 —— 连续多帧一起升高，比如打斗、爆炸这类持续大幅运动。

    因此这里用两道闸：
        1) 幅度闸：显著高于全局中位水平（中位数 × spike_factor，或中位数 + 6×MAD）；
        2) 孤立闸：同时显著强于前一帧和后一帧（isolation 倍）。
    只有两道都过才算切点，持续的运动段会被第二道闸挡掉。
    """
    if len(motion) < 3:
        return []

    med = float(np.median(motion))
    mad = float(np.median(np.abs(motion - med))) or 1e-6
    amp_gate = max(spike_factor * med, med + 6.0 * mad)

    cuts: list[int] = []
    for i in range(1, len(motion) - 1):
        v = float(motion[i])
        if v < amp_gate:
            continue
        if v > isolation * motion[i - 1] and v > isolation * motion[i + 1]:
            cuts.append(i)
    return cuts


def _colorfulness(bgr: np.ndarray) -> float:
    """Hasler–Süsstrunk 色彩丰富度度量。"""
    b, g, r = cv2.split(bgr.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, std_yb = rg.std(), yb.std()
    mean_rg, mean_yb = rg.mean(), yb.mean()
    return float(
        np.sqrt(std_rg**2 + std_yb**2)
        + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    )


#: 提示词中表达「切镜」意图的写法
_CUT_PATTERNS = (
    r"切镜",
    r"切\s*镜头",
    r"转场",
    r"分镜",
    r"SC[-\s]?\d+",
)


def count_shot_intents(prompt: str) -> int:
    """统计提示词里期望的镜头切分次数。

    用于把「镜头衔接」从主观感受变成可核对的指令遵循检查：
    提示词要求 3 次切镜，成片里就该找得到对应数量的切点。
    """
    if not prompt:
        return 0
    return sum(len(re.findall(p, prompt)) for p in _CUT_PATTERNS)


#: 提示词中表达「需要声音」的写法
_AUDIO_HINTS = (
    "台词", "配音", "对白", "人声", "旁白", "口型",
    "声音", "音频", "音乐", "音效", "起声", "收音",
)


def prompt_requires_audio(prompt: str) -> bool:
    """判断提示词是否要求声音。

    无音轨不能一律判为缺陷：纯特效/字幕/报幕镜头本就不该有声。
    只有提示词明确要求了声音（台词、配音、音效等）的用例，
    「成片无音轨」才应作为「音乐音效适配性」的扣分项。
    """
    if not prompt:
        return False
    return any(k in prompt for k in _AUDIO_HINTS)


#: 提示词中 @引用参考图/角色/场景的写法
_REF_HINTS = (
    "@图片", "@图", "@角色", "@人物", "@场景", "@道具", "@形象",
)


def prompt_refs_image(prompt: str) -> bool:
    """判断提示词是否 @引用了参考图作为人物/场景设定。

    参考图并不都是设定图——有些用例的参考图是报幕图、字幕排版图，
    与画面内容无关。把这类图拿去和成片比对相似度，会得到
    毫无意义的低分（如 ref_sim≈0.006）。只有 @引用了图片的用例，
    其参考图才参与「人物场景一致性」比对。
    """
    if not prompt:
        return False
    return any(k in prompt for k in _REF_HINTS)


# ---------------------------------------------------------------- 结果结构


@dataclass
class VideoMetrics:
    """单个成片的全部客观指标。"""

    path: str
    model: str
    case_id: str

    # 容器级
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0
    bit_rate: int = 0
    size_bytes: int = 0
    has_audio: bool = False

    # 采样
    n_sampled: int = 0

    # 时序运动
    motion_mean: float = 0.0
    motion_std: float = 0.0
    motion_cv: float = 0.0
    motion_p95: float = 0.0
    static_ratio: float = 0.0
    motion_outlier_ratio: float = 0.0

    # 单帧质量
    sharpness_mean: float = 0.0
    sharpness_min: float = 0.0
    sharpness_std: float = 0.0
    overexp_ratio: float = 0.0
    crushed_ratio: float = 0.0

    # 镜头结构
    shot_cut_count: int = 0
    cut_intent: int = 0
    cut_intent_delta: int = 0
    temporal_sim_mean: float = 0.0

    # 光影色彩
    contrast_mean: float = 0.0
    saturation_mean: float = 0.0
    colorfulness: float = 0.0

    # 参考一致性（无参考图或参考图非设定图时为 None）
    ref_sim_mean: float | None = None
    ref_sim_std: float | None = None
    subject_drift: float | None = None
    #: 首帧与参考图的相似度。图生视频的首帧通常最接近参考图，
    #: 用它可判断「起手是否跑偏」，与全程均值互补。
    first_ref_sim: float | None = None

    # 指令语义（从提示词解析，非视频内容）
    requires_audio: bool = False
    refs_image: bool = False

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------- 主计算


def compute_video_metrics(
    path: str | Path,
    model: str = "",
    case_id: str = "",
    ref_images: Sequence[str] | None = None,
    prompt: str = "",
    sample_fps: float = 12.0,
    work_width: int = 256,
    static_threshold: float = 0.004,
    container: dict[str, Any] | None = None,
) -> VideoMetrics | None:
    """计算单个成片的客观指标。

    Args:
        path: 视频路径。
        model: 模型名（用于结果标识）。
        case_id: 用例编号。
        ref_images: 参考图路径列表，用于一致性比对。
        prompt: 提示词原文，用于统计切镜意图。
        sample_fps: 抽帧频率。24fps 源按 12fps 抽，兼顾精度与耗时。
        work_width: 统一缩放宽度。跨模型分辨率不一致，必须先归一化。
        static_threshold: 帧间差分低于该值视为静止帧。
            切点判定改用自适应阈值（见 detect_shot_cuts），不再需要外部指定。
        container: ffprobe 已解析的容器级元数据。传入后直接采信，
            避免解码器自报值与容器头不一致导致的口径漂移。

    Returns:
        VideoMetrics；若无法解码返回 None。
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(native_fps / sample_fps))) if native_fps else 1

    m = VideoMetrics(path=str(path), model=model, case_id=case_id)
    m.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    m.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    m.fps = round(float(native_fps), 3)

    # 指令语义：从提示词解析，决定哪些客观指标「适用」
    m.requires_audio = prompt_requires_audio(prompt)
    m.refs_image = prompt_refs_image(prompt)

    # 音轨信息交给 ffprobe 侧补充，这里只记录是否有流
    prev_gray: np.ndarray | None = None
    prev_sig: np.ndarray | None = None

    diffs: list[float] = []
    sharpness: list[float] = []
    overexp: list[float] = []
    crushed: list[float] = []
    contrasts: list[float] = []
    sats: list[float] = []
    colorful: list[float] = []
    sims_seq: list[float] = []
    ref_frame_sims: list[float] = []
    first_ref_sim: float | None = None

    # 参考图签名（只需算一次）。
    # 仅当提示词 @引用了参考图时，参考图才作为「人物/场景设定」参与比对；
    # 报幕图、字幕排版图等非设定图不应参与，否则会得到无意义的低相似度。
    ref_sigs: list[np.ndarray] = []
    ref_hashes: list[np.ndarray] = []
    ref_images = list(ref_images or []) if m.refs_image else []
    for rp in ref_images:
        img = cv2.imread(str(rp))
        if img is None:
            continue
        img = _resize_keep_aspect(img, work_width)
        ref_sigs.append(_hsv_signature(img))
        ref_hashes.append(phash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step:
            idx += 1
            continue

        frame = _resize_keep_aspect(frame, work_width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        contrasts.append(float(gray.std()))
        overexp.append(float(np.mean(gray >= 250)))
        crushed.append(float(np.mean(gray <= 5)))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sats.append(float(hsv[:, :, 1].mean()))
        colorful.append(_colorfulness(frame))

        sig = _hsv_signature(frame)

        if prev_gray is not None and prev_sig is not None:
            d = cv2.absdiff(gray, prev_gray)
            diffs.append(float(d.mean() / 255.0))

            sims_seq.append(_hist_corr(prev_sig, sig))

        if ref_sigs:
            best = -2.0
            for rs, rh in zip(ref_sigs, ref_hashes):
                c = cv2.compareHist(
                    rs.astype(np.float32), sig.astype(np.float32),
                    cv2.HISTCMP_CORREL,
                )
                hd = hamming(rh, phash(gray)) / float(len(rh))
                # 直方图相关性做主判据（对景别、构图变化稳健），
                # 感知哈希距离只做小幅惩罚：参考图与成片帧的景别、
                # 构图本就不同，重罚会把一致性分整体压到地板上。
                best = max(best, float(c) - 0.15 * hd)
            ref_frame_sims.append(best)
            if first_ref_sim is None:
                first_ref_sim = best

        prev_gray, prev_sig = gray, sig
        idx += 1

    cap.release()

    if not sharpness:
        return None

    m.n_sampled = len(sharpness)
    m.size_bytes = path.stat().st_size

    # 容器级信息以 ffprobe 为准；未传入时回退到解码器自报值
    if container:
        m.width = int(container.get("width") or m.width)
        m.height = int(container.get("height") or m.height)
        m.fps = float(container.get("fps") or m.fps)
        m.duration = float(container.get("duration") or 0.0)
        m.bit_rate = int(container.get("bit_rate") or 0)
        m.has_audio = bool(container.get("has_audio", False))
    elif m.fps:
        m.duration = round(total_frames / m.fps, 3)

    if diffs:
        arr = np.asarray(diffs)
        m.motion_mean = float(arr.mean())
        m.motion_std = float(arr.std())
        m.motion_cv = float(arr.std() / arr.mean()) if arr.mean() > 1e-9 else 0.0
        m.motion_p95 = float(np.percentile(arr, 95))
        m.static_ratio = float(np.mean(arr < static_threshold))
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) or 1e-6
        m.motion_outlier_ratio = float(
            np.mean(np.abs(arr - med) / mad > 5.0)
        )

    m.sharpness_mean = float(np.mean(sharpness))
    m.sharpness_min = float(np.min(sharpness))
    m.sharpness_std = float(np.std(sharpness))
    m.overexp_ratio = float(np.mean(overexp))
    m.crushed_ratio = float(np.mean(crushed))
    m.contrast_mean = float(np.mean(contrasts))
    m.saturation_mean = float(np.mean(sats))
    m.colorfulness = float(np.mean(colorful))

    # 切镜遵循：成片实际切点数 对比 提示词期望切点数。
    # 这是「指令遵循」维度里少数可以全自动核对的一项。
    m.cut_intent = count_shot_intents(prompt)
    if diffs:
        m.shot_cut_count = len(detect_shot_cuts(np.asarray(diffs)))
    m.cut_intent_delta = m.shot_cut_count - m.cut_intent
    m.temporal_sim_mean = float(np.mean(sims_seq)) if sims_seq else 0.0

    if ref_frame_sims:
        ra = np.asarray(ref_frame_sims)
        m.ref_sim_mean = float(ra.mean())
        m.ref_sim_std = float(ra.std())
        third = max(1, len(ra) // 3)
        m.subject_drift = float(ra[-third:].mean() - ra[:third].mean())
        m.first_ref_sim = float(first_ref_sim) if first_ref_sim is not None else None

    return m


# ---------------------------------------------------------------- 批量调度


def _job(payload: dict[str, Any]) -> dict[str, Any] | None:
    m = compute_video_metrics(**payload)
    return m.as_dict() if m else None


def build_payloads(
    cases: Iterable[Any],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """把 CaseAsset 列表展开成 compute_video_metrics 可用的参数列表。

    在这里把每条用例的参考图与提示词注入到它的每个成片上，
    从而让「一致性」与「切镜遵循」这类需要上下文的指标得以计算。
    """
    payloads: list[dict[str, Any]] = []
    for case in cases:
        for v in case.videos:
            payloads.append({
                "path": v.path,
                "model": v.model,
                "case_id": case.case_id,
                "ref_images": list(case.ref_images),
                "prompt": case.prompt,
                "container": asdict(v),
                **kwargs,
            })
    return payloads


def batch_compute(
    payloads: Sequence[dict[str, Any]],
    workers: int | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """并行计算一批成片的指标。

    Args:
        payloads: build_payloads 产出的参数列表。
        workers: 并行进程数，默认取 CPU 核数。
        verbose: 是否打印进度。
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if workers is None:
        workers = min(8, os.cpu_count() or 4)

    results: list[dict[str, Any]] = []
    done = 0
    total = len(payloads)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, p) for p in payloads]
        for f in as_completed(futs):
            out = f.result()
            if out:
                results.append(out)
            done += 1
            if verbose and (done % 24 == 0 or done == total):
                print(f"  指标计算 {done}/{total}", flush=True)
    return results
