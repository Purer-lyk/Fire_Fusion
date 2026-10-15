# Dataset audit and reuse instructions

## Outcome

Source directories were read only. No model training or inference was performed. All task scripts, outputs, tests, temporary test fixtures, and explicit logs are under C:/lyk/fire/qualityDetection/audit or datasets.

| Modality | Source images | Train images / boxes / negatives | Val images / boxes / negatives | Excluded |
|---|---:|---|---|---:|
| RGB | 2599 | 2079 / 3260 / 42 | 520 / 794 / 9 | 0 |
| Thermal | 2449 | 1848 / 2341 / 11 | 462 / 498 / 6 | 139 |

Thermal exclusions: 89 exact reserved-test matches, 50 invalid annotation images. All 89 timestamped source images overlap the reserved test and are excluded. The retained thermal set consists of 640x640 images with Roboflow-style names; actual sensor provenance must be checked independently. The 192x256 timestamped subset is unavailable for training under the reserved-test constraint.

Observed XML classes only: RGB `fire` (4054 objects), thermal `Fire` (3129 objects). Explicit case-only normalization maps these to COCO category 1 `fire`. YOLO class IDs are only 0. No smoke or other XML class was mapped. Dataset filenames containing smoke are not class or scene truth.

No missing XML/TXT labels, orphan labels, or XML-versus-JPEG dimension mismatches were found. Thermal TXT and XML counts/coordinates agree for all 2449 images within 0.01 pixel. Fifty images have actual XML and TXT boxes outside image bounds (e.g. ymax=641 for height=640); these were excluded without clipping. Forty additional images have only tiny TXT normalization-rounding overflow; these are accepted at 0.01-pixel tolerance because their XML boxes are strictly valid and unchanged in COCO.

SHA256: 73 internal source duplicate groups (68 RGB, 5 thermal), covering 148 images and 75 redundant copies. Exact duplicate hashes never straddle train/val. Duplicates remain within one split; consider deduplicating for unbiased weighting/evaluation. No cross-modality source duplicate groups. No same-basename/different-hash test/source conflicts. No retained training/validation hash overlaps any of the 3056 reserved RGB/thermal images.

Reserved test: 1528 complete four-column pairs, no missing or unreferenced images, no invalid rows. RGB GT coverage 0/1528. Thermal GT coverage 89/1528, 162 boxes, no validated negatives; remaining 1439 thermal images are unlabeled, not negative. `test_existing_gt.json` contains only covered images. The RGB JSON is intentionally empty; do not run RGB detection metrics against it. Fourth-column UV is sensor input only; it never defines detection or scene truth. No scene ground truth is provided.

## Files

- audit/audit.json: summary, policies, exact-hash overlaps, basename conflicts, duplicate paths, limitations.
- audit/manifest.json: every source image, SHA256, actual dimensions, XML boxes/classes/issues, TXT agreement, grouping, split and exclusions.
- audit/test_manifest.json: every reserved image, hash, status and exact candidate/provenance paths; unlabeled boxes=null.
- audit/test_pairs.json: original pair identifiers, Tmax and UV_input.
- audit/verification.json: independent serialized-output checks and rehash of all 8104 images.
- datasets/dataset_paths.json: machine-readable absolute paths.
- datasets/{rgb,thermal}/{train,val,test_existing_gt}.json: COCO datasets referencing absolute read-only source image paths.
- datasets/{rgb,thermal}/ppyolov2_dataset.yml: PaddleDetection dataset fragments only. Merge with an actual PPYOLOv2 model/reader config. `allow_empty: true`, training `empty_ratio: 1.0` preserve all valid negative images. Source COCODataSet implementation was inspected to confirm these keys.
- audit/audit_datasets.py: stdlib-only reproducible audit/conversion.
- audit/test_audit.py: 6 tests developed through observed RED/GREEN cycles.
- audit/verify_outputs.py: independent output and hash verification.
- audit/{tests,run,verification,rounding_red}.log: execution evidence.

## Split and limitations

Deterministic seed string `qualityDetection-v1`: SHA256-sorted group names, greedy nearest 20% validation assignment. Union exact hashes across groups before assignment. Thermal Roboflow augmentation variants are grouped by stripped augmentation/hash suffix. Timestamp bursts group consecutive frames until a gap greater than 120 seconds; any group touching a reserved hash is fully excluded. `group` is the inferred original group and `split_group` is its duplicate-union component.

RGB numerical filenames lack identifiable video/time metadata. Thus exact-hash leakage prevention is verified, but RGB temporal/scene independence cannot be certified. Do not call this a fully scene-independent experimental benchmark. Timestamp and augmentation grouping is heuristic; near-duplicate images with different bytes are not detected. JPEG header sizes were inspected, not full pixel decoding.

Recommended before experiments: (1) obtain independently annotated, paired RGB/thermal box GT and separately reviewed scene labels for reserved tests; (2) establish real capture/video provenance and revise RGB grouping; (3) inspect thermal source modality/provenance, especially retained public/augmented-looking 640x640 images; (4) review the 50 invalid boxes and regenerate labels under an explicit reviewed policy if desired; (5) deduplicate same-hash weighting; (6) load-test these fragments with the chosen PaddleDetection environment and transforms. No training was attempted.

Reproduce in Windows bash, cwd C:/lyk/fire/qualityDetection/audit:

```bash
python -B -m unittest test_audit -v > tests.log 2>&1
python -B audit_datasets.py > run.log 2>&1
python -B verify_outputs.py > verification.log 2>&1
```
