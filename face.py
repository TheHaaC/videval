"""videval.face — ArcFace 人脸一致性（语义层指标）。

客观层的 ref_sim 用直方图 + 感知哈希间接估计「与参考图像不像」，
但它测的是整体构图相似度，对「这个人是不是那个人」并不敏感——
同一个角色换了景别、换了机位，ref_sim 就掉得很厉害，而人眼一眼能认出
「还是这个人」。这里用 ArcFace（InsightFace buffalo_s）做人脸 embedding，
直接量化「成片里的人物是否与参考角色设定一致」，
是「人物场景一致性」维度在语义层的补强。

只在提示词 @引用了参考图、且参考图中能检出人脸的用例上计算；
其余返回 None（不可评估），不参与该维度打分。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

__all__ = [
    "get_app",
    "reference_embeddings",
    "compute_face_sim",
    "batch_face_sim",
]

_app = None
_app_lock = threading.Lock()


def get_app():
    """懒加载 + 单例。InsightFace 的 onnxruntime session 非线程安全，
    批处理统一单线程顺序调用，因此一个实例即可。"""
    global _app
    with _app_lock:
        if _app is None:
            from insightface.app import FaceAnalysis

            _app = FaceAnalysis(
                name="buffalo_s", providers=["CPUExecutionProvider"]
            )
            _app.prepare(ctx_id=-1, det_size=(320, 320))
        return _app


def _face_embedding(
    img: np.ndarray, app, min_det: float = 0.5
) -> np.ndarray | None:
    """检测图中置信度最高的人脸并返回其归一化 embedding。

    用 det_score 过滤误检（衣纹、背景元素常被误当成人脸）。
    """
    faces = app.get(img)
    faces = [f for f in faces if float(getattr(f, "det_score", 0.0)) >= min_det]
    if not faces:
        return None
    best = max(faces, key=lambda f: f.det_score)
    emb = np.asarray(best.embedding, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(emb))
    return emb / n if n > 0 else emb


def reference_embeddings(
    ref_images: Sequence[str | Path], app=None, min_det: float = 0.5
) -> list[np.ndarray]:
    """从参考图中提取所有角色的人脸 embedding（一图最多取一张）。"""
    app = app or get_app()
    embs: list[np.ndarray] = []
    for p in ref_images:
        img = cv2.imread(str(p))
        if img is None:
            continue
        e = _face_embedding(img, app, min_det)
        if e is not None:
            embs.append(e)
    return embs


def compute_face_sim(
    video_path: str | Path,
    ref_images: Sequence[str | Path],
    app=None,
    sample_fps: float = 2.0,
    max_frames: int = 15,
    min_det: float = 0.5,
) -> float | None:
    """计算成片人物与参考角色的平均人脸余弦相似度。

    返回 [0, 1]；参考图或视频均检不出人脸时返回 None（不可评估）。
    """
    app = app or get_app()
    ref_embs = reference_embeddings(ref_images, app, min_det)
    if not ref_embs:
        return None

    ref = np.stack(ref_embs)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, int(round(fps / sample_fps)))

    sims: list[float] = []
    idx = 0
    n_taken = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step:
            idx += 1
            continue
        idx += 1

        e = _face_embedding(frame, app, min_det)
        if e is None:
            continue
        # 归一化向量点积即余弦相似度；取与「最像的那个角色」的相似度
        sims.append(float(np.max(ref @ e)))
        n_taken += 1
        if n_taken >= max_frames:
            break

    cap.release()
    if not sims:
        return None
    return float(np.mean(sims))


def batch_face_sim(
    payloads: Sequence[dict],
    sample_fps: float = 2.0,
    max_frames: int = 15,
    verbose: bool = True,
) -> list[dict]:
    """批量计算人脸一致性。

    payloads 每项含 path / ref_images。只对「参考图能检出人脸」的
    用例有产出，其余返回 None。单线程顺序执行，避免 onnxruntime
    多线程竞争。
    """
    app = get_app()
    out: list[dict] = []
    done = 0
    total = len(payloads)
    for p in payloads:
        v = compute_face_sim(
            p["path"], p.get("ref_images", []),
            app=app, sample_fps=sample_fps, max_frames=max_frames,
        )
        out.append({"path": p["path"], "face_sim_mean": v})
        done += 1
        if verbose and done % 8 == 0:
            print(f"  人脸一致性 {done}/{total}", flush=True)
    return out
