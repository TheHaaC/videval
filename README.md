# videval — 视频生成模型批量评测工具链

面向多模态视频生成模型的横向评测场景，覆盖**素材扫描 → 指标计算 → 六维评分 →
配对对比 → 偏好数据导出 → 报告生成**的完整链路。

## 快速开始

```bash
# 环境：Python 3.13 + ffmpeg
brew install ffmpeg
pip install numpy pandas opencv-python-headless scipy pillow

# 一键跑通全流程
python -m videval all --assets ./ai视频素材 --workers 8
```

单个步骤也可单独执行：

```bash
python -m videval scan     --assets ./ai视频素材   # 资产清单 + 生成元数据
python -m videval metrics  --assets ./ai视频素材   # 客观指标 + 六维评分
python -m videval face     --assets ./ai视频素材   # 语义层：ArcFace 人脸一致性
python -m videval prefs    --assets ./ai视频素材   # 配对对比 + 显著性检验 + 偏好对
python -m videval blind    --assets ./ai视频素材   # 盲测评分卡 + 匿名打分表
python -m videval calibrate --assets ./ai视频素材  # 用人工分校准客观指标
python -m videval report   --assets ./ai视频素材   # Markdown + HTML 报告
```

> 指标计算是唯一的重活，结果会缓存在 `outputs/metrics_raw.csv`。
> 后续步骤自动复用，避免同一批素材在报告和偏好对里算出两套分。
> 需要重算时加 `--force`。

## 输入目录结构

```
<assets>/
├── prompt/<case_id>/p.txt          提示词原文
├── prompt/<case_id>/*.(png|jpg)    参考图（角色/场景/道具）
├── prompt/<case_id>/*.(mp3|wav)    参考音频（声音克隆/口型）
├── <model_a>/<case_id>/*.mp4       模型 A 在该用例下的成片
└── <model_b>/<case_id>/*.mp4       模型 B 在该用例下的成片
```

新增模型只需在 `videval/probe.py` 的 `MODEL_ALIASES` 里登记目录名。

## 本次实验设计

评测对象为 **Rally**，按用例分段与不同对手配对比较：

| 分段 | 用例 | 对比 | 样本量 |
|---|---|---|---|
| A | 01–12 | Rally vs Seedance 2.0 | 48 个成片 |
| B | 13–24 | Rally vs Seedance 2.5 | 48 个成片 |

采用**配对对比**：同一条提示词、同一套参考资产，仅模型不同。
这样「提示词难度」这个混淆变量被消除，每次打分都是一次
「同一条件下 A 与 B 谁更好」的判断——也正是 RLHF 偏好数据的天然结构。

## 指标说明

| 分组 | 指标 | 对应维度 |
|---|---|---|
| 时序运动 | `motion_mean` `motion_cv` `static_ratio` `motion_outlier_ratio` | 运动合理性 |
| 单帧质量 | `sharpness_mean` `overexp_ratio` `crushed_ratio` | 单帧画面正确性 |
| 镜头结构 | `shot_cut_count` `cut_intent_delta` `temporal_sim_mean` | 镜头衔接流畅性 |
| 光影色彩 | `contrast_mean` `saturation_mean` `colorfulness` | 光影色彩和谐性 |
| 参考一致性 | `ref_sim_mean` `ref_sim_std` `subject_drift` `first_ref_sim` | 人物场景一致性 |
| 语义层 | `face_sim_mean`（ArcFace 人脸余弦相似度） | 人物场景一致性 |

几个值得说明的设计点：

**分辨率归一化。** 三个模型输出横跨 864×496 到 2560×1440，
不统一缩放到同一宽度再算像素级指标，高分辨率会天然占便宜。

**切镜检测要区分「切镜」和「运动」。** 提示词里常写「切镜」，
因此切点数量是可核对的指令遵循项。但帧间差分大也可能只是打斗激烈——
切镜是孤立单帧突变，运动段是连续多帧升高，
故用「幅度闸 + 孤立闸」两道判定，避免把打斗误判成切镜。

**一致性用分块直方图而非全图直方图。** 实测全图 HSV 直方图在相邻帧上
相关性恒为 1.000，毫无区分度；切成 3×3 网格后能拉开到 0.54–0.99。

## 产物

| 文件 | 说明 |
|---|---|
| `outputs/manifest.csv` | 资产清单，含分辨率/帧率/时长/码率/音轨 + 生成时间与重试序号 |
| `outputs/metrics_raw.csv` | 每个成片的原始客观指标 |
| `outputs/scores.csv` | 六维评分与加权总分 |
| `outputs/comparisons.csv` | 逐用例 × 逐维度的配对比较明细 |
| `outputs/preferences.jsonl` | DPO 格式偏好对，按分差降序 |
| `outputs/盲测评分卡.md` | 六维 1–5 档评分标准（主观盲测用） |
| `outputs/盲测打分表.csv` | 匿名打分表，填完发回用于校准客观指标 |
| `outputs/blind/` | 匿名化的视频副本（B01…，不暴露模型名） |
| `outputs/blind_meta.json` | 盲测编号与原始路径的映射（校准阶段用） |
| `outputs/评测报告.md` | Markdown 横评报告，含显著性检验 |
| `outputs/评测看板.html` | HTML 可视化看板（内联 SVG，无外部依赖） |

## 算力需求

全部指标基于 CPU 经典算法，无需 GPU。
本次 96 个成片、20.1 分钟素材，在 Apple M1（8 核 / 16GB）上
8 进程并行约 **60 秒**跑完。

> 需要 GPU 的是语义层指标（CLIP 图文相似度、VBench、LLM-as-judge），
> 而不是本工具覆盖的客观层。两者定位不同，见报告的「方法论与局限」。

## 统计严谨性

- **配对对比 + 显著性检验**：不再只数「几胜几负」，而是对每个维度的
  配对分差做 Wilcoxon 符号秩检验 + bootstrap 95% 置信区间，报告 p 值与 CI。
- **主观盲测校准**：`blind` 命令产出匿名视频副本与打分表，人工打分后
  与客观分算 Spearman 相关，相关性高的维度可信、低的需换指标或降权。

## 已知局限

1. 一致性指标依赖参考图，无参考图的用例该项不可用；
2. 切镜检测基于帧间差分突变，相邻镜头画面相近时会漏检；
3. 感知哈希与直方图对形变、手部崩坏等局部缺陷不敏感，
   这类问题仍需人工或语义模型识别；
4. 语义正确性、叙事逻辑、审美水准属主观层，须由多人盲测补齐，
   客观指标只能间接反映；
5. 商业平台不暴露随机种子，以每用例多次采样（2 次）替代 seed 控制；
6. 各模型成片生成时间错开，若平台在采样期间更新版本将构成混淆变量；
7. 每分段仅 12 个配对用例，显著性检验功效有限，结论宜视为阶段性观察。
