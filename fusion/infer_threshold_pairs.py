"""Real PPYOLOv2 inference only, via the local official PaddleDetection Trainer.predict."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
# Restrict all runtime caches and temporary writes to the approved workspace.
for key, value in {'TMPDIR':'tmp', 'TEMP':'tmp', 'TMP':'tmp', 'HOME':'home', 'USERPROFILE':'home', 'XDG_CACHE_HOME':'cache', 'PADDLE_HOME':'cache/paddle', 'PIP_CACHE_DIR':'cache/pip', 'MPLCONFIGDIR':'cache/matplotlib', 'CUDA_CACHE_PATH':'cache/cuda'}.items():
    p = ROOT / 'runtime' / value
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'runtime/PaddleDetection'))

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modality', choices=['rgb', 'thermal'], required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--confidence', type=float, default=0.5)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    assert 0 < args.confidence < 1
    threshold = args.confidence
    output = Path(args.output_dir).resolve()
    assert output.is_relative_to(ROOT), 'Writes outside approved workspace prohibited'
    pairs = json.loads((ROOT / 'audit/test_pairs.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / 'audit/test_manifest.json').read_text(encoding='utf-8'))
    assert len(pairs) == 1528 and len(manifest) == 3056
    source = Path('C:/lyk/paper_data/paper_data_/paper_data/paper_sample.txt')
    source_rows = [x.split() for x in source.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    assert len(source_rows) == len(pairs)
    for row, pair in zip(source_rows, pairs):
        assert row[0] == pair['rgb'] and row[1] == pair['thermal']
        assert float(row[2]) == pair['Tmax'] and float(row[3]) == pair['UV_input']
    lookup = {(x['modality'], x['basename']): x for x in manifest}
    assert len(lookup) == len(manifest)
    mapping = []
    missing = []
    for pair_index, pair in enumerate(pairs):
        entry = {'pair_index': pair_index, 'source_line': pair['line'], 'rgb_name': pair['rgb'], 'thermal_name': pair['thermal']}
        for modality in ['rgb', 'thermal']:
            info = lookup[(modality, pair[modality])]
            entry[modality + '_path'] = info['path']
            entry[modality + '_sha256'] = info['sha256']
            if not Path(info['path']).is_file():
                missing.append(info['path'])
        mapping.append(entry)
    assert not missing, missing
    if args.limit:
        mapping = mapping[:args.limit]
    images = [x[args.modality + '_path'] for x in mapping]
    assert len(set(images)) == len(images)
    # Confirm exact source content against the audit before each model run.
    for entry in mapping:
        path = entry[args.modality + '_path']
        assert digest(path) == entry[args.modality + '_sha256'], path
    save(output / 'image_mapping.json', mapping)
    import paddle
    import numpy as np
    from ppdet.core.workspace import load_config
    from ppdet.engine import Trainer
    from ppdet.utils.check import check_config, check_gpu, check_version
    cfg_path = ROOT / ('configs/ppyolov2_' + args.modality + '.yml')
    cfg = load_config(str(cfg_path))
    # This copied ImageFolder unconditionally opens a COCO index. Supply only
    # real image metadata and category vocabulary, with NO annotations/truth.
    index_path = output / (args.modality + '_inference_image_index.json')
    save(index_path, {'images': [{'id': i, 'file_name': x[args.modality + '_path'], 'width': lookup[(args.modality, x[args.modality + '_name'])]['width'], 'height': lookup[(args.modality, x[args.modality + '_name'])]['height']} for i, x in enumerate(mapping)], 'annotations': [], 'categories': [{'id': 1, 'name': 'fire'}]})
    cfg.TestDataset['dataset_dir'] = str(ROOT)
    cfg.TestDataset['anno_path'] = str(index_path)
    cfg.use_gpu = True
    cfg.worker_num = 0
    cfg.TestReader['batch_size'] = 1
    cfg.TestReader['use_shared_memory'] = False
    # Explicit confidence threshold at decode and MatrixNMS stages.
    cfg.BBoxPostProcess['decode']['conf_thresh'] = threshold
    cfg.BBoxPostProcess['nms']['score_threshold'] = threshold
    cfg.BBoxPostProcess['nms']['post_threshold'] = threshold
    assert cfg.architecture == 'YOLOv3' and cfg.YOLOv3['neck'] == 'PPYOLOPAN'
    device = paddle.set_device('gpu:0')
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()
    checkpoint = Path(str(cfg.weights) + '.pdparams')
    metadata = {'modality': args.modality, 'python': sys.executable, 'paddle_version': paddle.__version__, 'device': str(device), 'gpu_name': paddle.device.cuda.get_device_name(0), 'config': str(cfg_path), 'config_sha256': digest(cfg_path), 'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size, 'source_manifest': str(ROOT / 'audit/test_manifest.json'), 'source_pairs': str(ROOT / 'audit/test_pairs.json'), 'paper_sample_sha256': digest(source), 'architecture': cfg.architecture, 'backbone': dict(cfg.ResNet), 'neck': 'PPYOLOPAN', 'iou_aware': cfg.YOLOv3Head['iou_aware'], 'post_process': dict(cfg.BBoxPostProcess), 'minimum_score': threshold, 'coordinate_format': 'xyxy in original-image pixels', 'input_shape': [3, 640, 640], 'UV_used_by_network': False, 'truth_used': False, 'expected_images': len(images)}
    save(output / (args.modality + '_run_metadata.json'), metadata)
    print('INFERENCE_START ' + json.dumps(metadata), flush=True)
    trainer = Trainer(cfg, mode='test')
    trainer.load_weights(str(cfg.weights))
    started = time.perf_counter()
    with paddle.no_grad():
        results = trainer.predict(images, output_dir=str(output / (args.modality + '_official')), save_results=True, visualize=False, save_threshold=threshold)
    paddle.device.cuda.synchronize()
    elapsed = time.perf_counter() - started
    imid2path = {int(k): str(v) for k, v in trainer.dataset.get_imid2path().items()}
    path_to_entry = {str(Path(x[args.modality + '_path']).resolve()): x for x in mapping}
    records = []
    raw_records = []
    seen = set()
    for batch in results:
        ids = np.asarray(batch['im_id']).reshape(-1)
        counts = np.asarray(batch['bbox_num']).reshape(-1)
        boxes = np.asarray(batch['bbox'])
        assert sum(int(x) for x in counts) == len(boxes)
        start = 0
        for image_id, count in zip(ids, counts):
            image_id, count = int(image_id), int(count)
            path = str(Path(imid2path[image_id]).resolve())
            entry = path_to_entry[path]
            assert path not in seen
            seen.add(path)
            raw = boxes[start:start + count]
            start += count
            assert np.isfinite(raw).all()
            retained = [{'class_id': int(row[0]), 'class_name': 'fire', 'score': float(row[1]), 'bbox_xyxy': [float(v) for v in row[2:6]]} for row in raw if int(row[0]) >= 0 and float(row[1]) >= threshold]
            assert all(x['class_id'] == 0 for x in retained)
            info = lookup[(args.modality, entry[args.modality + '_name'])]
            record = {'pair_index': entry['pair_index'], 'source_line': entry['source_line'], 'image_id': image_id, 'image_path': path, 'image_name': entry[args.modality + '_name'], 'image_sha256': info['sha256'], 'width': info['width'], 'height': info['height'], 'detections': retained}
            records.append(record)
            raw_records.append({'pair_index': entry['pair_index'], 'image_id': image_id, 'image_path': path, 'bbox_rows_class_score_xyxy': raw.tolist()})
    records.sort(key=lambda x: x['pair_index'])
    raw_records.sort(key=lambda x: x['pair_index'])
    missing_images = sorted(set(path_to_entry) - seen)
    assert not missing_images and len(records) == len(images)
    save(output / (args.modality + '_predictions.json'), {'metadata': metadata, 'images': records})
    save(output / (args.modality + '_raw_model_outputs.json'), raw_records)
    save(output / (args.modality + '_image_id_to_path.json'), imid2path)
    summary = {'modality': args.modality, 'processed_images': len(records), 'expected_images': len(images), 'missing_images': missing_images, 'duplicate_images': len(records) - len(seen), 'retained_boxes': sum(len(x['detections']) for x in records), 'images_with_detections': sum(bool(x['detections']) for x in records), 'images_without_detections': sum(not x['detections'] for x in records), 'elapsed_predict_seconds': elapsed, 'batches_returned': len(results), 'prediction_json': str(output / (args.modality + '_predictions.json')), 'checkpoint_sha256': metadata['checkpoint_sha256'], 'complete': True}
    save(output / (args.modality + '_summary.json'), summary)
    print('INFERENCE_COMPLETE ' + json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
