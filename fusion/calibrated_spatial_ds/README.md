# Calibrated spatial DS — internal retained exploratory evaluation

All 1528 pairs were previously seen. This is NOT independent validation. Reference labels are UV column 4; optional UV branch uses that same UV and is circular.

## Outcome
- Full: primary vision TP1032 FP0 FN366 TN130, FPR0, FNR0.2618025751; meets requested full-dataset target.
- Internal retained: TP0 FP0 FN180 TN22, FNR1; same-subset thermal baseline TP176 FP0 FN4 TN22. Joint full+retained target FAILS.
- UV suppression branch matches primary metrics exactly; no added gain.
- Selected: RGB>=.2, thermal>=.1, polygon IoU>=.01, rmin=1, mass>.3, nominal quality1. rmin=1 means NO dynamic spatial reliability discount; spatial consensus remains via dual real detector boxes.

## Protocol
3056 source image SHA256 reverified; all unique. Filename gap>10s temporal proxies, not verified scenes. Whole group, class-stratified chronological closest-to-60/20/20 split: train920, calibration406, retained202, both classes present in each. Groups not split. Entire data previously seen; unknown near-duplicate/source leakage remains.

480 fixed-grid configurations searched ONLY on calibration labels. Minimum FP then FN then trial ID. `split_allocation.csv` and `search_plan.json` precede search; `frozen_selected.json` precedes retained evaluation. No retained retuning. No supervised probability calibrator fitted; train used only for descriptive evaluation.

Fire-only BPA `(r*c,0,1-r*c)`, r=rmin+(1-rmin)*IoU, K=0. Classic DS becomes OR reinforcement. Scores max over one-to-one Hungarian real matches, recomputed AFTER detector-confidence filtering. Missing either modality ->0. UV secondary mF'=mF*(.5+.5*UV), no UV-forced fire.

Genuine existing conf=.1, imgsz640 full/ROI predictions reused from ../crossmodal_roi_ds/full_results.json. No new detector forwards; all swept candidate thresholds >=.1. ROI trigger remains .25, top3/modality; projections propose crops, not detections. Current retained positive block has zero RGB candidates despite 360 old ROI attempts; no BPA setting can fix this cache coverage. This bound says nothing about fresh conf=.05 inference.

## Entry points
Original experiment (refuses if freeze exists; do not erase/research retained data):
`C:/lyk/fire/qualityDetection/.venv/Scripts/python.exe C:/lyk/fire/qualityDetection/fusion/calibrated_spatial_ds/run_calibration.py`

Tests:
`C:/lyk/fire/qualityDetection/.venv/Scripts/python.exe -m unittest discover -s C:/lyk/fire/qualityDetection/fusion/calibrated_spatial_ds -p test_*.py -v`

Report: ../../reports/校准空间DS_优化验证报告.docx

See metrics.json, frame_predictions.csv, calibration_trials.csv, split_metadata.json, source_audit.json, execution.json and verification.json for counted outcomes, provenance, frozen receipts and hashes. Prior files, reports and weights remain unchanged.
