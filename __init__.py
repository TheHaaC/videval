"""videval — 视频生成模型批量评测工具链。

面向多模态视频生成模型的横向评测场景，提供从素材扫描、客观指标计算、
评分聚合、偏好对构造到报告导出的完整链路。

设计原则：
    1. 零 GPU 依赖 —— 全部指标基于 CPU 可复现的经典图像算法，任何机器可跑；
    2. 跨模型可比 —— 统一分辨率归一化后再计算像素级指标，消除规格差异；
    3. 结果可追溯 —— 每个分数都能回溯到具体帧与中间量，便于缺陷归因。

典型用法：
    $ python -m videval scan     --assets ./ai视频素材
    $ python -m videval metrics  --assets ./ai视频素材
    $ python -m videval prefs    --assets ./ai视频素材
    $ python -m videval report   --assets ./ai视频素材
"""

__version__ = "0.1.0"
__all__ = [
    "probe",
    "metrics",
    "scoring",
    "prefs",
    "report",
]
