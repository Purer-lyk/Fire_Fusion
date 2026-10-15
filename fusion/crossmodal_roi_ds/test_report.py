import unittest
class Report(unittest.TestCase):
 def test_report_real_results_and_three_images(self):
  import build_report
  receipt=build_report.build()
  self.assertEqual(receipt['unique_pairs'],1528)
  self.assertGreaterEqual(receipt['embedded_images'],3)
  self.assertFalse(receipt['meets_all_target'])
if __name__=='__main__':unittest.main()
