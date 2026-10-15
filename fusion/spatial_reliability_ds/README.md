# 空间可靠性折扣 D-S：固定方案真实实验

工作目录：`C:/lyk/fire/qualityDetection/fusion/spatial_reliability_ds`。

执行：

```bash
C:/lyk/fire/qualityDetection/.venv/Scripts/python.exe -m unittest test_spatial -v
C:/lyk/fire/qualityDetection/.venv/Scripts/python.exe run_experiment.py
```

主交付：`C:/lyk/fire/qualityDetection/reports/spatial_reliability_DS_1528_最终对比报告.docx`。

- 参数在计算全量指标之前冻结，`frozen_config.json` + `config_freeze_receipt.json` 哈希绑定；重跑检查哈希，不按测试UV标签调参。
- 复用用户两个YOLO26n真实预测；重新核验3056图片、1528唯一行及权重SHA256；不重新推理。
- FULL affine四角polygon IoU、带dummy的Hungarian最大总IoU一对一匹配；门限0.01；缺失vacuous，未匹配在场可靠性0.5。
- 同一预UV候选用于新方案所有消融；最高普通DS mF聚合，不累计相关框。
- 标准可靠性折扣后经典DS结合RGB/thermal，UV用经典DS而不是空间融合。
- 基础BPA/UV/二分类口径均有显式算法变化；BetP>0.5主结果与统一mF>=0.5次结果均报告。原C++旧结果保留旧发布口径。
- 第三列温度单位不明，主方案温度门控中性1；可配置外部校准sigmoid，但本次不启用。原热温度BPA受控消融另报。
- 真实结果：无UV的空间折扣没有BetP二分类增益；瞬时UV方案100%与UV标签符合，但UV单独及无空间+UV也100%，属于循环评价，不是独立真实性能提升。保留旧UV及mF阈值时可出现退化，全部报告。

输出：`fusion_results.csv/json`（全1528行）、`geometry_evidence.json`（全部候选、索引、IoU、BPA）、`metrics.json`（13方案指标、两种统一阈值、0.8/0.95弃权和覆盖）、`execution_summary.json`（来源/模型/配准/缺失/冻结/C++重跑）、`deliverable_verification.json`、`report_readback.txt`、5幅真实叠加PNG。

`red_evidence.log`、`red_geometry.log`、`red_pipeline.log`记录三个纵向实现切片缺少实现时预期失败；`tests_green.log`记录最终通过。DOCX另经docx_validate健康检查。

限制：UV输入兼参考循环性；配准7点候选未独立验证（LOO RMSE约5.77px）；温度单位和模型训练/测试重叠未核实；没有独立场景标签。原算法/数据/报告只读，哈希保护核验保持不变。
