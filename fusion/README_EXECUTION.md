# 实际融合交付

主要算法是用户只读 `C:/lyk/paper/bpa_test/src/ds_fusion.cpp` 的 `fuse3` 返回分支：两次固定 s=0.1 空间规则。不是将建议算法替换为原算法。`core.py` 保持原 temporal BPA 的参数和初始化，新增显式几何方法与明确不同的建议变体。真实NN逐帧最大分数和paper_sample温度/UV替换原main随机输入。

## 重跑（Git Bash，原C++不修改）

```bash
g++ -O2 -std=c++17 -I 'C:/lyk/paper/bpa_test/include' 'C:/lyk/fire/qualityDetection/fusion/cpp_reference_harness.cpp' 'C:/lyk/paper/bpa_test/src/bpa.cpp' 'C:/lyk/paper/bpa_test/src/vision_bpa.cpp' 'C:/lyk/paper/bpa_test/src/temp_bpa.cpp' 'C:/lyk/paper/bpa_test/src/uv_bpa.cpp' 'C:/lyk/paper/bpa_test/src/ds_fusion.cpp' -o 'C:/lyk/fire/qualityDetection/fusion/cpp_reference_harness.exe'
cd /c/lyk/fire/qualityDetection/fusion
'C:/lyk/fire/qualityDetection/.venv/Scripts/python.exe' -B -m unittest -v test_core test_execution_geometry
'C:/lyk/fire/qualityDetection/.venv/Scripts/python.exe' -B execute_fusion_report.py
```

报告脚本会覆盖同名融合输出与报告；原始数据、模型和原C++只读。C++参考程序运行需要可用的MSYS/UCRT C++运行库（当前Git Bash环境已实际运行exit0）。神经网络推理已由父任务执行，融合脚本读取真实 `rgb_predictions.json` 和 `thermal_predictions.json`，不会重新训练或使用模拟输出。

## 产物
- `fusion_results.csv` / `.json`：1528行，各方法质量值、决策、几何、边界和置信度。
- `fusion_summary.json`：方法计数、分布、来源与模型哈希、重置规则、实际子集指标。
- `cpp_reference_inputs.txt` / `cpp_reference_results.txt` / `cpp_parity_verification.json`：原C++真实1528行参考，15值/行与Python最大差3.3306690738754696e-16。
- `candidate_calibration.json`：原homo_main七对应点、thermal→RGB候选H、fit/LOO误差；跨场景未验证。
- `geometry_pairing.json`：RGB最高分锚框和最佳polygon IoU热框，同时保存最高分对IoU。
- `current_truth_audit.json`：当前原标签/人工reviewed记录检查快照。此目录可能由用户继续标注，快照覆盖率不应当永久现状。
- `thermal_verified_subset.json` / `thermal_subset_metrics.json` / `.csv`：89张同哈希和当前XML核验热图，162目标。
- `blank_scene_label_template_1528.csv`：1528行，所有真值字段为空，未生成标签。
- `fusion_deliverable_verification.json`：CSV行数、空标签检查、DOCX CRC/读回/哈希。
- `fusion_tests.log`：7项单元测试通过。
- `../reports/PPYOLOv2_DS融合实际测试报告.docx` / `_readback.txt`：中文实际结果报告与文本读回。

## 必要限制
完整1528对Accuracy/Precision/Recall/F1全部N/A；缺少完整独立场景真值。当前数据库只有1条reviewed场景记录，原源RGB TXT只是局部未验证候选标签。UV和温度绝不当真值。热89图目标级score≥0.5、IoU≥0.5结果TP=0、FP=0、FN=162，Precision未定义/N/A，Recall=0、F1=0；不是完整融合指标，也不是质量提升结论。

候选H拟合RMSE约2.771px，但留一RMSE约60.853px，不能称为已验证配准。时间边界按文件名间隔>120s或非递增启发式重置，共6个段；不是已确认场景边界。原空间normalize取消共同分母，固定0.1只是火支持缩放，非实测IoU。UV无空间信息，建议变体与原fuse3区别见报告。
