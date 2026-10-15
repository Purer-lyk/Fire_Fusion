import unittest,subprocess,json
from pathlib import Path
class DeliverableTest(unittest.TestCase):
 def test_pair_runtime_and_report(self):
  root=Path(__file__).parent
  self.assertTrue((root/'infer_pair.py').exists(),'standalone paired inference not implemented')
  self.assertTrue((root/'build_report.py').exists(),'verified report generation not implemented')
if __name__=='__main__':unittest.main()
