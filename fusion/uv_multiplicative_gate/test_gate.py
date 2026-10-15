import unittest
import importlib.util
from pathlib import Path

class FormulaTest(unittest.TestCase):
    def test_literal_uv_suppression(self):
        path=Path(__file__).with_name('multiplicative_core.py')
        self.assertTrue(path.exists(), 'new multiplicative implementation missing')
        spec=importlib.util.spec_from_file_location('mc',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        result=mod.apply((.8,.1,.1),0,.5)
        self.assertEqual(result['mass'],(.4,.1,.5))
        self.assertEqual(result['fire'],0)

class IntegrationTest(unittest.TestCase):
    def test_full_runner_and_invariants(self):
        runner=Path(__file__).with_name('run_experiment.py')
        self.assertTrue(runner.exists(), 'full 1528 runner missing')
        import subprocess, sys, json
        proc=subprocess.run([sys.executable,str(runner)],capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
        v=json.loads(runner.with_name('deliverable_verification.json').read_text(encoding='utf-8'))
        self.assertEqual(v['csv_rows'],1528)
        self.assertTrue(v['metrics_recomputed_from_csv'])
        self.assertTrue(v['docx_zip_crc_ok'])
        self.assertTrue(v['original_sources_and_old_reports_unchanged'])

if __name__=='__main__': unittest.main(verbosity=2)
