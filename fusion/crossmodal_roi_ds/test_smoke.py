import unittest
from pathlib import Path
class Runner(unittest.TestCase):
 def test_real_smoke_three_unique_bindings(self):
  import run_experiment
  result=run_experiment.run(limit=3,tag='smoke')
  self.assertEqual(len(result),3)
  self.assertEqual(len({x['pair_index'] for x in result}),3)
  for x in result:
   self.assertTrue(x['rgb_path'].endswith(x['rgb_name']))
   self.assertTrue(x['thermal_path'].endswith(x['thermal_name']))
   self.assertLessEqual(len(x['roi_attempts']),6)
if __name__=='__main__':unittest.main()
