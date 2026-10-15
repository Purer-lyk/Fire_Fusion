"""Run the already-smoke-tested full RGB then thermal inference sequentially."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv/Scripts/python.exe'

def main():
    evidence = {'python': str(PYTHON), 'sequential_gpu_order': ['rgb', 'thermal'], 'smoke_verified': {}, 'runs': []}
    for modality in ['rgb', 'thermal']:
        smoke = json.loads((ROOT / 'fusion/smoke' / (modality + '_summary.json')).read_text(encoding='utf-8'))
        assert smoke['complete'] and smoke['processed_images'] == 3 and not smoke['missing_images']
        evidence['smoke_verified'][modality] = smoke
    destination = ROOT / 'fusion'
    state = destination / 'inference_execution.json'
    for modality in ['rgb', 'thermal']:
        command = [str(PYTHON), '-B', str(destination / 'infer_trained_pairs.py'), '--modality', modality, '--output-dir', str(destination)]
        log = destination / 'logs' / (modality + '_full.log')
        run = {'modality': modality, 'command': command, 'log': str(log), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'exit_code': None}
        evidence['runs'].append(run)
        state.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
        with log.open('wb') as handle:
            proc = subprocess.run(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
        run['exit_code'] = proc.returncode
        run['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        state.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
        print(json.dumps(run), flush=True)
        if proc.returncode:
            return proc.returncode
        summary = json.loads((destination / (modality + '_summary.json')).read_text(encoding='utf-8'))
        assert summary['complete'] and summary['processed_images'] == 1528 and not summary['missing_images']
        run['verified_summary'] = summary
        state.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    evidence['all_completed_exit_zero'] = True
    state.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print('BOTH_MODALITIES_COMPLETED_EXIT_0', flush=True)
    return 0

if __name__ == '__main__':
    sys.exit(main())
