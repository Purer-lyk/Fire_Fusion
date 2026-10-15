# Crossmodal ROI + DS genuine 1528-pair evaluation

Primary target FAILED: TP158 FP0 FN1240 TN130; FPR0%, FNR88.69814%. Optional UV multiplicative gate identical. All metrics use UV reference, not independent reviewed scene truth.

Run with C:/lyk/fire/qualityDetection/.venv/Scripts/python.exe and this directory scripts. test_core.py has seven unit tests; test_smoke.py performs three actual GPU pairs; run_experiment.py performs all 1528 pairs and ROI reviews; test_report.py builds and validates report. Full run repeats inference and writes only this experiment directory.

Fixed configuration was SHA256-receipted before full inference. No label tuning, no retraining, no substituted weights. Independent score calibration NOT fitted (identity interface only); geometry candidate unvalidated; quality nominal; temperature and temporal smoothing OFF.

Actual 3056 full-frame +1525 ROI forwards, 422 ROI detections, CUDA RTX5070. All 3056 image hashes reverified. Full logs and source/weight SHA receipts are local.

Fair ablation uses identical deduplication and matching; zero final ROI rescues. An intermediate unequal-dedup comparison misleadingly showed five gains; corrected before report by applying identical dedup to both branches and rerunning the entire dataset. Primary predictions were unchanged. The first attempted report failed case selection because no ROI-based pair ever crossed the alarm threshold; final examples honestly show a real low-score ROI match, full-frame successful match, and no-response crop.

Historical baselines recomputed from genuine frozen previous predictions. Extra new_conf01 metrics describe retained full-frame evidence after final dedup (not a pristine new standalone inference baseline); use the documented historical baselines for requested comparison.

Report: C:/lyk/fire/qualityDetection/reports/跨模态ROI复核_DS_1528验证报告.docx. DOCX ZIP/readback/three images and docx_validate.py passed.
