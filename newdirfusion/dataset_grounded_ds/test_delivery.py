import unittest
try:
 import grounded_core as c
except ImportError:
 c=None
class CoreTests(unittest.TestCase):
 def test_actual_same_denominator_target(self):
  self.assertIsNotNone(c,'grounded_core missing')
  m=c.metrics([1,1,0,0],[1,1,0,0]); t=c.metrics([1,1,0,0],[1,0,0,0]); r=c.metrics([1,1,0,0],[1,0,1,0])
  self.assertEqual((m['TP'],m['FP'],m['FN'],m['TN']),(2,0,0,2))
  self.assertTrue(c.target(m,t,r))
  self.assertFalse(c.target(t,t,r))
 def test_hash_exclusion_not_filename(self):
  self.assertTrue(hasattr(c,'safe_records'),'safe_records missing')
  rows=[{'sha256':'test','valid':True,'box_count':1},{'sha256':'safe','valid':True,'box_count':0},{'sha256':'bad','valid':False,'box_count':1}]
  self.assertEqual(c.safe_records(rows,{'test'}),[rows[1]])
 def test_final_real_predictions_and_report(self):
  from pathlib import Path
  import json,csv,zipfile
  from docx import Document
  out=Path(__file__).resolve().parent
  self.assertTrue((out/'safe_metrics.json').is_file(),'new real full inference missing')
  metrics=json.loads((out/'safe_metrics.json').read_text())
  with (out/'safe_frame_predictions.csv').open(encoding='utf-8-sig') as f:
   rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),1528)
  self.assertEqual(len({r['pair_index'] for r in rows}),1528)
  for name,field in [('fusion','fusion_fire'),('rgb_baseline','rgb_baseline_fire'),('thermal_baseline','thermal_baseline_fire')]:
   self.assertEqual(c.metrics([int(r['UV_reference']) for r in rows],[int(r[field]) for r in rows]),metrics['results']['full'][name])
  report=out/'dataset_grounded_spatial_DS_Chinese.docx'
  self.assertTrue(report.is_file(),'new Chinese DOCX missing')
  with zipfile.ZipFile(report) as z:self.assertIsNone(z.testzip())
  text='\n'.join(p.text for p in Document(report).paragraphs)
  self.assertIn('不属于独立验证',text)
  self.assertIn('精确哈希',text)
if __name__=='__main__':unittest.main()
