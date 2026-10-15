# Adaptive paired spatial DS — real implementation and verification

## Outcome (not an independent success claim)
- Fixed exposed-data asymmetric model: **TP1265 FP0 FN23 TN240**. Native320 RGB anchors + actual thermal full320/full640/fixed2x2 overlapping tiles320. Real nominal mapped polygon IoU>=.001, simple-support DS rT=.1, strict mass>.5. Runtime has no UV input.
- Training-selected six temporal-proxy-group OOF: **TP1263 FP6 FN25 TN234 — FAIL** for FP0/FN<=25.
- Same selected-view baselines: RGB320 **TP1262 FP0 FN26 TN240**; thermal multiview **TP1199 FP180 FN89 TN60**.
- Symmetric multiview RGB control is **TP1279 FP59 FN9 TN181**; fixed anchor model does NOT dominate its FN. Report all controls, not only older/weaker baseline.
- Original thermal320 prior benchmark716TP becomes717TP on this mixed-view-batch fresh forward: pair141 max .499995708 -> .500807524. Disclose boundary drift rather than claiming exact reproduction.
- All1528 pairs repeatedly seen; followup anchor variant was proposed AFTER initial results. Fixed result is descriptive/exposed-data, not pre-registered OOF or independent scene validity.

## Geometry: not solved by fake alignment
Native RGB H/W320/320, thermal256/192. Exact original homo_main.cpp calibration frame exists at its original path and matches current data copies by SHA256. Original seven points lie around narrow August stool/ring band. Inspected original raw source+points, zero-overlap pairs60/95/817, actual new boxes, and three additional positives727/747/879.

Actual81 background-image feature-registration attempts (27frames x gradientSIFT/intensitySIFT/gradientORB) yielded at most5 mutual Lowe-filtered matches, zero qualified proposals. No new transform promoted, no firebox/UV-fitted correction. Exclude image borders, existing box regions and red/yellow burned-in markings for feature extraction.

Seeded32 affine bootstrap distribution from seven calibration points only; expected actual-box polygon IoU and q10 are distinct diagnostic semantics, never per-frame maximizing transform. It does not validate systematic scene change.

Full multiview numerical coverage1287/1288 is misleading if interpreted as true flame coverage: pairs60/95 select actual low-score tile outputs spanning dark lower-right background, area~70x/66x RGB box. They are real detections, not artificial padding, but visually spurious matches. Pair817 has plausible full640 local bright-region rescue, IoU~.180. Three added positives727/747/879 have tighter plausible flame-like supports, IoU .449/.328/.334, but source images include burned-in annotations. **No independently validated full-field geometry or flame-level truth.**

## What ran
- Actual fresh GPU detector views18336 across3056 original images,64.76s RTX5070; plus36-view smoke and7-view standalone pair817 rerun.
- Initial bounded32 diagnostic configs: 2geometries x2BPA modes x8thresholds, training-only weak frame-label logistic calibration. All failed metrics retained.
- Adaptive nativeRGB-anchor followup32 configs; finite64 total, no unlimited hunt. Full selection and sixfold selection recorded separately.
- Independent verifier recomputes all32 anchor configurations,6fold selections, actual selected IoUs, all1528 mass rows, all original image/model hashes,18336view bindings and protected previous artifacts.
- Eight tests PASS with preserved RED/GREEN logs. DOCX ZIP/relationships/styles validation PASS, table readback verified.

## Commands (Windows native paths, git-bash)
Interpreter: `C:/lyk/fire/qualityDetection/.venv/Scripts/python.exe`
Directory: `C:/lyk/fire/qualityDetection/fusion/adaptive_spatial_ds`

1. `python run_adaptive.py prepare` (does not change an existing frozen protocol)
2. `python run_adaptive.py geometry`
3. `python run_adaptive.py fresh` (real full forwards; overwrites ONLY this experiment outputs)
4. `python run_adaptive.py evaluate` (finite first32 diagnostics)
5. `python run_adaptive.py anchor_evaluate` (finite adaptive32 diagnostics)
6. `python verify_adaptive.py`
7. `python build_report.py`, then `python verify_adaptive.py`
8. `python -m unittest test_adaptive test_deliverable` (includes real smoke and full cached evaluation)

Standalone runtime, seven fresh detector views:
`python infer_pair.py --rgb ABSOLUTE_RGB_PATH --thermal ABSOLUTE_THERMAL_PATH --output ABSOLUTE_OUTPUT_JSON`
No sample.txt, UV, filename feature or label needed by this entrypoint. Returns real selected boxes/masses/IoU/provenance and unvalidated-geometry warning. Batch-size numerical variations are possible.

## Artifacts
- `protocol.json`, receipt, `anchor_protocol.json`, all per-fold freezes
- `*_all_predictions.json`, `*_all_full320.json`, actualview receipt/bindings
- `bootstrap_geometry.json`, `registration_audit.json`, `geometry_cache.npz`
- `candidate_bounds_before_calibration.json`, perframe coverage and predictions
- `frame_evidence.json`, `anchor_evidence.json`, metrics/trials JSON/CSV
- native pixel visual overlays and `visual_audit.json`
- `verify_adaptive.py`, `verification.json`, RED/GREEN/test logs
- standalone `infer_pair.py`, `deployment.json` and actual runtime receipt
- Report: `C:/lyk/fire/qualityDetection/reports/自适应空间DS_双基线复验报告.docx`

Prior experiments, original source data and checkpoints unchanged. No independent-generalization or reliable object-consensus success claim is supported.
