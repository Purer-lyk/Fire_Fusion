import unittest
import calibrated_core as c
class FusionTest(unittest.TestCase):
 def test_fire_only_mass_and_missing_gate(self):
  self.assertAlmostEqual(c.fire_mass(.4,.5,.2,.5),1-(1-.6*.4)*(1-.6*.5))
  self.assertEqual(c.frame_score([],[],[],.1,.1,.01,.5),0)
 def test_uv_never_forces_fire(self):
  self.assertEqual(c.uv_score(0,1,.5),0)
  self.assertEqual(c.uv_score(.8,0,.5),.4)
 def test_hungarian_recomputed_after_filtering(self):
  rgb=[{'score':.1},{'score':.8}]; thermal=[{'score':.8}]
  self.assertAlmostEqual(c.frame_score(rgb,thermal,[[.9],[.2]],.2,.2,.1,1),.96)
 def test_target_uses_subset_baseline(self):
  self.assertFalse(c.target({'FP':0,'FNR':.5},{'FPR':0,'FNR':.4},{'FPR':.9}))
  self.assertTrue(c.target({'FP':0,'FNR':.3},{'FPR':0,'FNR':.4},{'FPR':.9}))
if __name__=='__main__': unittest.main()
