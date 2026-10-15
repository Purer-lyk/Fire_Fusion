# 严格双模态空间门控 D-S（实际1528对）

## 已冻结的主规则

- 在场检测均为用户权重YOLO26n真实fire预测，严格score>0.5，复用缓存，不重训、不重新推理。
- thermal框四角FULL affine映射到RGB；真实polygon IoU；Hungarian最大总IoU一对一匹配。
- τL=0.1，τH=0.3：用户未给数值，明确作为可配置工程默认，而非验证最优值；全量指标前保存配置SHA256与UTC回执。
- g=clip((s−τL)/(τH−τL),0,1)，两种视觉BPA同时可靠性折扣至未知，再经典D-S。
- 缺框／无正IoU匹配／s≤τL：NoFire；τL<s<τH：Transition（二分类仍0）；s≥τH：FireCandidate，且BetP(F)>0.5才最终1。
- UV瞬时可靠性0.9；UV不参与配准、匹配或选框，绝不能推翻硬门控。
- 候选依次按IoU、无UV视觉BetP、最低在场检测分数、候选顺序选一个，避免多框相关证据累加。
- 原始D-S质量与最终硬标签分开保存，硬无火是用户政策，不声称物理无火证明。

## 实际主结果（相对于第四列UV参考）

无UV及含瞬时UV的严格门控输出相同：TP168 / FP0 / FN1230 / TN130。
Accuracy19.5026%，Precision100%，Recall12.0172%，F1 21.4559%，Specificity100%，Balanced Accuracy56.0086%。
此前及同候选普通DS：Accuracy63.4817%，F1 77.5904%。严格政策消除这里的误报，但严重损失召回；不能报告准确率提升。
状态计数：NoFire1359、Transition1、FireCandidate168。1230个原始含UV判火被硬门控阻止。
UV1有1398帧、UV0有130帧。含UV融合输入与标签相同，是循环评价／UV符合度，非独立真实火灾性能。
候选仿射LOO RMSE5.7718px，无独立全视场／跨场景验证。

## 文件

- `fusion_results.csv` / `.json`：1528唯一逐帧行，原始质量、IoU/g、状态、最终hard label和对照。
- `metrics.csv` / `.json`：混淆矩阵、P/R/F1/Accuracy、Balanced Accuracy、Specificity、FPR/FNR。
- `sensitivity_metrics.csv`：预先冻结τH=0.1/0.2/0.3/0.5、τL=τH/3，不选择测试最优。
- `geometry_evidence.json`：全部候选、配对索引、基础／折扣／融合BPA、冲突、最终标签。
- `frozen_config.json` / `config_freeze_receipt.json`：主参数与运行前SHA256／UTC。
- `execution_summary.json` / `deliverable_verification.json` / `protected_before_hashes.json`：来源、状态、旧文件保留及校验。
- `examples/*.png`：真实NoFire／Transition／FireCandidate叠加图。
- `test_gate.py` / `gate_core.py` / `run_experiment.py`：边界、缺模态、无交叠、UV不能越门、几何匹配及全量执行测试。
- `red_*.log` / `green_*.log` / `tests_and_execution.log`：实际TDD失败／通过及运行日志。
- `report_readback.txt` / `docx_package_validation.json`：中文DOCX文本和包校验。
- 报告：`C:/lyk/fire/qualityDetection/reports/spatial_gate_DS_1528_最终报告.docx`。

## 重跑（Git Bash，工作目录qualityDetection）

```bash
.venv/Scripts/python.exe fusion/spatial_gate_ds/test_gate.py > fusion/spatial_gate_ds/tests_and_execution.log 2>&1
# 或仅执行已冻结完整实验：
.venv/Scripts/python.exe fusion/spatial_gate_ds/run_experiment.py
```

重跑会更新本新目录与本新报告，不修改旧实验／旧报告／原C++／源图像／权重；同一冻结配置输出CSV指标应完全一致。不要为了匹配UV测试而修改冻结配置。如做外部定义的新参数实验，另建目录与新的运行前冻结回执。

## 已完成验证

1528唯一索引、3056源图SHA256／尺寸／行序、两份权重SHA256、仿射来源SHA256；全部候选一对一；所有非FireCandidate硬标签均0；CSV回读独立混淆计数／指标重算；203个旧文件SHA256前后不变；DOCX ZIP CRC／文本表格／4嵌图／SHA256与包关系检查通过。
