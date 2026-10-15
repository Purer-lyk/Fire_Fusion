# qualityDetection：火焰检测与空间一致性 D-S 融合实验工作区

本项目保存 RGB / 热成像火焰检测的数据审计、模型训练、真实推理缓存、空间配准、D-S 融合比较及中文报告。它是**可追溯的实验工作区**，不是已完成独立认证的报警产品。

**当前主线：** [continuous_iou_ds](fusion/continuous_iou_ds/README.md)，在已有真实 RGB 原图 320 + 热图多视角缓存上，受控比较固定可靠性与连续 IoU 可靠性；本轮不训练、不重新网络前向、预测不输入 UV。

**最新报告：** [连续IoU可靠性_DS复验报告.docx](reports/连续IoU可靠性_DS复验报告.docx)。完整指标以本分支的 [metrics.json](fusion/continuous_iou_ds/metrics.json)、冻结协议和逐帧结果为准，不要混用历史报告的标签或阈值。

## 1. 从哪里开始

| 目的 | 入口 | 是否产生写入 / GPU 计算 |
|---|---|---|
| 阅读最新结论 | 上述最新 DOCX、主线 README、metrics.json | 只读 |
| 检查连续可靠性公式和候选重选 | `fusion/continuous_iou_ds/test_continuous.py` | 纯数值单测，不运行模型 |
| 查看逐帧证据 | `fusion/continuous_iou_ds/frame_predictions.csv`、`frame_evidence.json` | 只读 |
| 重算最新缓存融合 | `fusion/continuous_iou_ds/run_continuous.py evaluate` | 覆盖该分支结果，无新前向 |
| 重建最新 DOCX | `fusion/continuous_iou_ds/run_continuous.py report` | 覆盖最新同名报告 |
| 独立全量核验 | `fusion/continuous_iou_ds/verify_continuous.py` | 读取全量源文件，并写核验回执和报告读回 |
| 对新图像做单对推理 | `fusion/adaptive_spatial_ds/infer_pair.py` | GPU 新前向；是历史固定可靠性版本，不是连续版 |
| 查旧 PPYOLOv2 训练 | `training/training_state.json`、`runtime/batchrunner.py` | 启动器可能训练，勿当成融合入口 |

建议先阅读结果和协议，再执行纯数值单测。**阅读 README 不需要重训或重跑实验。**

## 2. 目录导航

```text
qualityDetection/
├─ README.md                    本导航文档
├─ experiment_spec.json         初始实验约束；不能替代各分支当前冻结协议
├─ .venv/                       项目本地 Python 环境
├─ audit/                       原始数据、标注、哈希、分组及保留集审计
├─ datasets/                    rgb/thermal 的 COCO JSON、路径和数据配置
├─ configs/                     PPYOLOv2 RGB / thermal 配置
├─ runtime/                     PaddleDetection 副本、运行包装器、探针和训练启动器
├─ training/                    旧 PPYOLOv2 训练日志、权重、评估和故障恢复归档
├─ fusion/
│  ├─ continuous_iou_ds/        当前主线：连续可靠性受控缓存复验
│  ├─ adaptive_spatial_ds/      主线缓存来源；原图 RGB 锚点 + 热图多视角
│  ├─ corrected_uv_optimized_ds/ 修正参考后的有限优化与验证
│  ├─ supplied320_corrected_uv/ 固定 320 缓存的修正 UV 重评估
│  ├─ supplied320_paper_experiment/ 用户权重双模态固定 320 实验
│  ├─ supplied_models_affine_iou/ 用户权重与仿射多边形 IoU 分支
│  ├─ supplied_models_uv_reference/ 较早的 UV 参考分支
│  ├─ spatial_gate_ds/、spatial_reliability_ds/ 早期空间门控 / 可靠性分支
│  ├─ crossmodal_roi_ds/、calibrated_spatial_ds/ ROI / 校准分支
│  ├─ dual_baseline_target_ds/  双基线约束与候选覆盖诊断
│  ├─ uv_multiplicative_gate/   可选 UV 乘性门控历史分支
│  └─ core.py 等               最早 PPYOLOv2 / 原 C++ 对照产物
├─ newdirfusion/
│  └─ dataset_grounded_ds/     TorchDetection 来源审计及排重后 30 轮训练支线
└─ reports/                     各阶段中文 DOCX、图表与 archive/
```

文件名相同不表示协议相同。`fusion/metrics` 类结果必须连同所在分支、模型哈希、输入视角和标签版本一起解释。

## 3. 模型、数据与外部只读依赖

### 3.1 当前主线模型来源

当前缓存来自用户提供的两份 **YOLO26n end-to-end 单类检测权重**，不是本工作区 PPYOLOv2 的训练权重。

| 模态 | 原始权重路径 | SHA256 |
|---|---|---|
| RGB | `C:/Users/Administrator/AppData/Local/hermes/attachments/best.pt` | `1068cfddcca133ac2f6c66522d7a43f4988daa91023c1c8898bb6fbf4a894e32` |
| thermal | `C:/Users/Administrator/AppData/Local/hermes/attachments/best-2.pt` | `c068422549d5de50d532431892a940a44ce2300b1cb0fae2e12887749adc12e6` |

检查点元数据记录 `yolo26n.yaml`、训练 `imgsz=320`；RGB / thermal 原类名分别为 `fire` / `Fire`。完整结构、训练参数和源路径保存在 [当前协议](fusion/continuous_iou_ds/protocol.json) 与 [320 模型元数据](fusion/supplied320_paper_experiment/model_metadata.json)。检查点参数不等于已经证明其历史训练成员名单。

### 3.2 成对数据及当前参考标签

只读成对数据根目录：`C:/lyk/paper_data/paper_data_/paper_data`。其中 `paper_sample.txt` 按空白分隔四列：

```text
RGB文件名  thermal文件名  Tmax  UV
```

- 当前完整样本为 **1,528 对：第四列 UV=1 有 1,288 对，UV=0 有 240 对**。
- 当前参考文件 SHA256：`7fa9997f14d65213063651f8448dc56344f797911185a2d6170147f1960bd483`。
- 旧标签口径 **1,398 / 130 已过时**；旧报告可用于追溯，不能当作当前分母。
- 第四列是用户指定的帧级参考，不是独立人工审核的场景真值，更不是目标框标注。主线只在评估阶段使用它。
- 第三列 Tmax 不进入当前主线预测；不能把历史温度 BPA 说成已校准的物理温度证据。
- 原图 RGB 宽×高为 `320×320`，thermal 为 `192×256`。这与网络输入张量大小不同；缓存 `orig_shape` 按高、宽记录。

早期数据审计及 COCO 转换见 [audit/README.md](audit/README.md)。其中“尚未训练”“UV 不定义真值”等是当时任务范围的历史描述；当前帧级 UV 参考比较仍须保留非独立真值的限制。

### 3.3 其他源目录与保护边界

- `C:/lyk/fire/_paddleDetection/paddleDetection/dataset/fire/fireOrigin`：早期 RGB 标注源。
- `C:/lyk/fire/TorchDetection/datasets/thermal/totals`：早期热图标注源。
- `C:/lyk/fire/TorchDetection`：后续 YOLO 数据和训练来源审计对象，保持只读。
- `C:/lyk/paper/bpa_test`：原 C++ 算法与七点标定源，保持只读。

当前脚本 / JSON 中含本机绝对路径；**只复制仓库并不能直接迁移运行**。外部权重、原图、参考文件和上游缓存必须存在且哈希一致。不要通过修改参考数据、替换权重或伪造缓存来让核验通过。

## 4. 当前主线：输入、配准与融合规则

### 4.1 实际输入池

复用 `fusion/adaptive_spatial_ds/` 的真实预测：

- RGB：`rgb_all_full320.json`，只使用原图 `full320`；不把 RGB 多视角辅助对照偷换为主输入。
- thermal：`thermal_all_predictions.json`，包含原图 `full320`、`full640` 以及四个重叠 `tile*_320`，沿用原跨视角去重。
- 历史真实候选置信度下限为严格 `score > 0.005`。主线融合对完整真实候选池重新做匹配及最终质量重选。
- 因热图含 640 视角，不能将该主线称为“双模态全部仅 320 输入”。较早 `supplied320_paper_experiment` 才是固定双模态 320 分支。

YOLO26n end-to-end 输出的候选边界、`max_det` 截断与常规 NMS 不同；不能仅因接口有 `iou` 参数便声称执行了常规 NMS。真实 320 输入验证及后处理源快照保存在早期 320 分支。

### 4.2 几何与匹配

使用原七对应点拟合的 **FULL affine，thermal pts2 → RGB pts1，2×3 矩阵**。当前主线不是透视变换；热框四角仿射后计算凸多边形 IoU，不用轴对齐外接矩形代替。

先按硬门槛筛选，再进行允许 dummy 的 Hungarian 最大总 IoU 一对一匹配。匹配后重算全部候选质量，取最终 `mF` 最大者；同分保留原顺序首项。几何独立核验保留原框角点先转 float32、仿射运算使用 float64、投影多边形再转 float32 的数值路径。

**门槛纠正：原 `anchor_protocol.json` 和 `adaptive_core.anchor_fuse` 实际为 `IoU >= 0.001`，不是旧叙述中的 `0.01`。** 单变量主对照保留 `0.001`；`0.01` 和 `0.3` 必须作为额外硬门槛分支单列。

### 4.3 连续可靠性与决策

令 `s` 为匹配框多边形 IoU，`cR/cT` 为对应真实检测分数：

```text
g(s) = clip((s - 0.1) / 0.2, 0, 1)
rR = 1
rT(s) = 0.1 * g(s)
mR = (cR, 0, 1-cR)
mT = (rT*cT, 0, 1-rT*cT)
K = 0
mF = 1 - (1-cR)*(1-rT*cT)
mN = 0
mU = (1-cR)*(1-rT*cT)
Fire ⇔ 存在合格真实匹配，且 mF > 0.5
```

这是只支持 Fire 的简单支持 BPA，组合无冲突，具有正向 OR 强化效应；不是带负证据的冲突抑制模型，也不宣称检测分数为校准后验。阈值用 `mF`，不是 BetP。

缺任一真实模态证据或无合格匹配时输出 `(0,0,1)`，二值 Fire=0；这并不等于得到了 NoFire 的肯定证据。`s<=0.1` 时热增量为零，但保留弱硬门槛后，匹配 RGB 自身 `cR>0.5` 仍能报火；“必须强空间确认”是另一个硬门槛干预。

当前纯融合函数见 [continuous_core.py](fusion/continuous_iou_ds/continuous_core.py)。预测接口不接收 UV、Tmax、标签或文件名特征。

### 4.4 当前保存的全量结果

下表来自最新 `metrics.json`，均为**已反复暴露全量数据的帧级 UV 参考一致性统计**，不是独立认证，也不是检测 AP。

| 策略 | 硬门槛 | TP | FP | FN | TN | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原固定可靠性 | 0.001 | 1265 | 0 | 23 | 240 | 0% | 1.7857% |
| 连续可靠性主对照 | 0.001 | 1265 | 0 | 23 | 240 | 0% | 1.7857% |
| 固定可靠性附加对照 | 0.01 | 1261 | 0 | 27 | 240 | 0% | 2.0963% |
| 连续可靠性附加对照 | 0.01 | 1261 | 0 | 27 | 240 | 0% | 2.0963% |
| 连续可靠性严格确认 | 0.3 | 1190 | 0 | 98 | 240 | 0% | 7.6087% |

`FPR=FP/240`，`FNR=FN/1288`。同缓存 RGB 原图 320 基线为 `TP1262 FP0 FN26 TN240`，thermal 多视角为 `TP1199 FP180 FN89 TN60`。RGB 多视角辅助对照 `TP1279 FP59 FN9 TN181` 不能遗漏，也不能被主线宣称全面支配。

主门槛共有 3,090 个实际匹配，199 个候选 `g<1`，81 帧最终质量下降，但二值决策全部不变。因此**连续化本次未提高分类指标**；`0.01` 相对主线少 4 个 TP 来自额外硬门槛，不能归因于连续折扣。

## 5. Windows PowerShell 快速使用

以下命令按实际源码入口书写；不依赖系统默认 `python`。根目录为 `C:\lyk\fire\qualityDetection`。

### 5.1 环境与无副作用单测

```powershell
Set-Location 'C:\lyk\fire\qualityDetection'
& .\.venv\Scripts\python.exe -B -c "import sys; print(sys.executable); print(sys.version)"
Set-Location '.\fusion\continuous_iou_ds'
& ..\..\.venv\Scripts\python.exe -B -m unittest test_continuous -v
```

`test_continuous.py` 只检验纯数值公式、严格阈值和候选重选；本次 README 编写时实际执行为 **2 tests / OK**。不要将其等同于新做了完整模型推理或部署验证。

### 5.2 缓存重算与核验（显式写入，非日常阅读步骤）

**下列命令会覆盖已有产物。本次文档整理没有执行它们。** 先备份目标分支和最新 DOCX；需要长期保留比较时，先在独立实验副本中明确调整输出路径，当前脚本没有 `--output-dir` 开关。

```powershell
Set-Location 'C:\lyk\fire\qualityDetection\fusion\continuous_iou_ds'
& ..\..\.venv\Scripts\python.exe -B .\run_continuous.py evaluate
& ..\..\.venv\Scripts\python.exe -B .\run_continuous.py report
& ..\..\.venv\Scripts\python.exe -B .\verify_continuous.py
```

- `evaluate`：核验图像 / 权重，重算三个硬门槛的匹配、五策略及逐帧输出，覆盖本分支 JSON/CSV。
- `report`：覆盖 `reports/连续IoU可靠性_DS复验报告.docx`。
- `verify_continuous.py`：独立重算几何 / 匹配 / 质量，核验协议、受保护源、图像和权重哈希以及 DOCX CRC；覆盖本分支 `verification.json`、`report_readback.txt`。它**不是纯只读命令**。
- `run_continuous.py` 通过位置参数选择 `prepare/evaluate/report`，不是 argparse CLI；不要照搬其他分支参数。
- 现有协议已经冻结，**不要再次 `prepare`**；源码会拒绝覆盖现有 `protocol.json`。
- 重算依赖 `adaptive_spatial_ds` 及其上游模块 / 缓存和外部源路径，不是只有当前目录便可独立运行。

### 5.3 新图像入口与训练入口的区别

`fusion/adaptive_spatial_ds/infer_pair.py` 实际接受必填 `--rgb`、`--thermal`、`--output`，读取同目录 `deployment.json`，执行 GPU 0 上七个真实检测视角，并直接写指定 JSON；输出父目录必须预先存在。它使用**历史固定可靠性 anchor_fuse**，不能标注为连续可靠性部署版。新图运行前须检查图像、权重哈希及未验证几何警告；本任务不运行此入口。

`runtime/batchrunner.py` 是旧 PPYOLOv2 训练启动器，支持 `--validate-only` 与 `--resume-rgb`。无参数运行会进入训练流程，不是当前融合复现命令；即使 validate-only 也可能写状态 / 日志。训练及恢复请先核对现有权重、归档和对应源码，避免误覆盖。

## 6. 环境记录与迁移检查

编写本 README 时用项目 `.venv/Scripts/python.exe` 的包元数据实际查询：

| 项目 | 本机已安装版本 |
|---|---|
| Python | 3.12.10 |
| NumPy / SciPy | 1.26.4 / 1.17.1 |
| opencv-python | 4.11.0.86 |
| torch | 2.13.0+cu132 |
| ultralytics | 8.4.60 |
| python-docx | 1.2.0 |
| paddlepaddle-gpu | 3.3.1 |

这是现成环境的实测清单，**不是通用安装锁文件或新机器兼容保证**。当前连续融合 / 独立核验主要依赖 NumPy、SciPy、OpenCV；生成 / 读回报告需要 python-docx；新前向需要 PyTorch、Ultralytics 与相容 GPU；Paddle 用于旧 PPYOLOv2 管线。不要为读报告而重装全局环境。

迁移或重新计算前逐项检查：

1. 项目解释器存在、依赖可导入；需要新前向时另查 CUDA / GPU 可用性。
2. 两份外部权重仍存在，且匹配协议 SHA256。
3. 当前参考文件哈希、四列格式、1,528 对及 1,288 / 240 分布一致。
4. 3,056 张源图、文件名配对、原始尺寸与逐图哈希一致。
5. 上游真实预测池、视角绑定、几何源文件、冻结协议和保护清单完整。
6. 明确输出覆盖范围，备份历史结果；不要在有意改协议后仍冒称原协议复现。

## 7. 实验历史：不要把不同管线混成同一次实验

| 阶段 / 分支 | 用途与正确定位 |
|---|---|
| `audit` → `datasets` → `runtime` / `training` | 早期排除保留集同哈希的 COCO 数据与 PPYOLOv2 640 训练；RGB / thermal 的 model_final.pdparams 目前均存在，training_state.json 记录 completed、两任务退出码 0 |
| `fusion` 根目录 | 最早 PPYOLOv2 缓存与原 C++ 规则对照；参见 [执行说明](fusion/README_EXECUTION.md)，其固定 s=0.1 和历史透视候选不代表当前主线 |
| 用户权重 / `supplied320_paper_experiment` | YOLO26n 用户权重、双模态固定 320 真实输入及后处理审计；报告 [320 论文式实验](reports/320输入_空间一致性DS_论文式实验报告.docx) |
| `supplied320_corrected_uv` | 不重训、不重前向，核验源哈希后按修正第四列重算；报告 [320 修正 UV 复验](reports/320输入_修正UV标签_空间DS复验报告.docx) |
| `corrected_uv_optimized_ds` | 修正参考下有限搜索及分组结果，保留失败与覆盖上界；报告 [修正 UV 优化验证](reports/修正UV_空间DS优化验证报告.docx) |
| 空间可靠性 / ROI / 校准 / 双基线分支 | 方法探索和失败诊断；见各自 README / 协议，不应从其中挑一行当独立泛化结论 |
| `newdirfusion/dataset_grounded_ds` | 原 TorchDetection 来源审计、safe_datasets 与两模态 30 轮 YOLO26n 安全重训支线；[完成回执](newdirfusion/dataset_grounded_ds/safe_training_complete.json) 与 [该支线报告](newdirfusion/dataset_grounded_ds/dataset_grounded_spatial_DS_Chinese.docx)，不是当前用户权重主线 |
| `adaptive_spatial_ds` | 原图 RGB 锚点 + thermal 多视角；真实缓存来源，固定全量结果与分组 OOF 分开；见 [说明](fusion/adaptive_spatial_ds/README.md) |
| `continuous_iou_ds` | 当前受控缓存复验：保持实际 0.001 门槛，只连续化 thermal 可靠性；不是重新训练或新的 OOF |

`runtime/README_TRAINING.md` 保留了早期“未启动全训练”的交接快照；这已经不是当前状态。查训练完成情况优先读 `training/training_state.json`、真实权重和训练日志，不要仅引用旧交接文字。

旧 320 修正 UV 分支另有 `mUV=mF*(0.5+0.5*UV)`、严格 `>0.3` 的可选乘性分支；UV=0 只是减半，不保证否决。该公式与当前连续主线不同，不能跨分支套用。若同一 UV 既参与预测又充当参考，评估存在循环性，不能称为独立性能。

## 8. 产物追溯与可信范围

最新分支中建议一起保留：

- `protocol.json`、`freeze_receipt.json`、`protected_sources.json`：冻结参数、协议哈希及只读保护清单。
- `image_hash_verification.json`：源图哈希和尺寸。
- `frame_predictions.csv`、`frame_evidence.json`、`all_matched_candidates.csv`：逐帧输出、完整候选与实际匹配。
- `metrics.json`、`delta_cases.json`、`known_gains.json`：指标与变化定位；pair_index 为 0 起始审计标识。
- `verification.json`、测试日志、`docx_validation.json`、`report_readback.txt`：数值独立核验及文档核验回执。

现有主线核验回执记录 1,528 帧、五策略、9,078 行匹配、3,056 张图像哈希及两份模型哈希验证；它是已保存的实验回执，本次 README 编写未重新运行全量验证器。

必须保留的限制：

1. **全量已暴露、非独立测试。** 历史反复查看和方法选择不会被本轮冻结协议消除。较早 adaptive 分组 OOF 为 `TP1263 FP6 FN25 TN234`，没有满足零 FP 目标；不能用全量 `FP=0` 保证部署零误报。
2. **来源重叠风险。** [原始来源重叠审计](fusion/supplied320_paper_experiment/source_overlap_audit.json) 记录 RGB 原 train / val 与当前成对数据同哈希重叠 298 / 74 张，thermal 为 74 / 15 张。这证明当前源划分存在内容重叠，不单凭此证明用户检查点历史实际用了这些图。早期排重 PPYOLOv2、safe 重训支线和用户权重必须分别讨论。
3. **几何未获全视场独立验证。** 七点覆盖窄，历史留一误差约 5.77 像素；当前名义仿射不能自动修复场景漂移 / 视差。
4. **真实网络输出不等于真实火焰对应。** 部分低分 tile 框覆盖大面积暗背景；RGB / thermal 的烧录框、Fire 文字可能污染检测。候选覆盖上升不等于人工验证的火焰覆盖上升。
5. **阈值与批次敏感。** 历史新前向已有约 0.5 附近分数跨阈值的记录；相同权重不保证不同批次安排的逐位输出相同。
6. **没有完整独立框 / 场景真值。** 当前帧级 UV 参考一致性不能替代目标级 AP、独立 Fire/NoFire/Uncertain 审核或现场长期误报测试。

后续若面向部署，应另建未暴露场景 / 视频分组测试集、独立审核标注、验证跨视场配准及去除烧录标记污染，并在独立验证集上冻结阈值。本 README 只整理现有工作区，不新增实验、不改变源数据、脚本、模型或历史报告。
