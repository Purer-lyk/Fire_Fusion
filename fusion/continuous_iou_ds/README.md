# 连续IoU可靠性：受控缓存复验

## 实测结果

全部1528对：1288阳性、240阴性。输入保持真实RGB原图full320 + 真实热图多视角（full320/full640/四tile320），不重新前向、不训练、预测不输入UV。

|策略|硬门槛|TP|FP|FN|TN|FPR|FNR|
|---|---:|---:|---:|---:|---:|---:|---:|
|原固定可靠性|.001|1265|0|23|240|0%|1.7857%|
|连续可靠性主对照|.001|1265|0|23|240|0%|1.7857%|
|固定可靠性文字门槛对照|.01|1261|0|27|240|0%|2.0963%|
|连续可靠性文字门槛对照|.01|1261|0|27|240|0%|2.0963%|
|连续可靠性严格确认|.3|1190|0|98|240|0%|7.6087%|

**门槛纠正：** 实际原anchor_protocol.json与adaptive_core.anchor_fuse都使用.001，而非叙述中的.01。单变量主对照保留实际源码门槛；另行执行.01固定/连续配对与.3严格确认，避免隐藏改变。

唯一主改动：`rT(s)=.1*clip((s-.1)/(.3-.1),0,1)`；`mF=1-(1-cR)*(1-rT(s)*cT)>.5`。sL=.1、sH=.3是预声明的历史区间默认值，不是按本次标签搜索结果。Hungarian最大总IoU一对一匹配后重算所有候选，按最大mF重新选取；不只更新历史已选证据。

主对照3090个实际匹配，199个g<1；81帧最终质量下降，二值决策全部不变；727/747/879的IoU均>.3，g=1，新增TP保持。严格确认降低召回，不能把其额外门槛影响归因于连续折扣。保留弱硬门槛时g=0仍可由匹配RGB自身cR>.5触发。

## 产物

- `protocol.json` / `freeze_receipt.json`：本次计算指标前冻结；协议SHA256 `54bfe2546d5e272a34b805e4958d5eb3300642fa332a0482e01d5876b42ec2d8`。
- `continuous_core.py`：纯数值融合，无UV/标签/文件名输入。
- `run_continuous.py`：prepare / evaluate / report。
- `frame_predictions.csv`：全1528帧五策略的选中证据及mF/mN/mU/决策。
- `frame_evidence.json`：全部真实RGB/thermal候选及匹配池、选中结果。
- `all_matched_candidates.csv`：三个门槛共9078行匹配、固定与连续质量。
- `metrics.json` / `delta_cases.json` / `known_gains.json`：指标及全部变化ID。
- `image_hash_verification.json`：3056图像的SHA256及维度。
- `verify_continuous.py` / `verification.json`：独立多边形、Hungarian、质量及CSV读回、权重/源保护、文档CRC核验。
- `test_continuous.py`、RED/GREEN日志：连续公式、重选非历史候选、严格阈值测试。
- `docx_validation.json` / `report_readback.txt`：DOCX包验证及读回正文。
- 报告：`C:/lyk/fire/qualityDetection/reports/连续IoU可靠性_DS复验报告.docx`。

## 重现

使用项目现成解释器（默认python缺numpy，未改全局环境）：

```
C:/lyk/fire/qualityDetection/.venv/Scripts/python.exe run_continuous.py evaluate
C:/lyk/fire/qualityDetection/.venv/Scripts/python.exe run_continuous.py report
C:/lyk/fire/qualityDetection/.venv/Scripts/python.exe verify_continuous.py
C:/lyk/fire/qualityDetection/.venv/Scripts/python.exe -m unittest test_continuous -v
```

protocol已有冻结时不要再次prepare，不要覆盖原缓存或历史报告。独立核验必须保留原float32框角点在仿射前的舍入，以匹配原几何数值路径。

全部数据历史已暴露、存在训练/测试内容重叠风险，七点配准约5.77px LOO及烧录文字污染仍未解决。这是全量描述性受控比较，不是独立认证或新OOF测试。
