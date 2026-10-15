import unittest
import numpy as np
import core
class Geometry(unittest.TestCase):
 def test_affine_inverse_corners(self):
  a=np.array([[.7,.04,78],[-.04,.96,-16.]])
  p=core.corners([10,20,40,50]); q=core.transform(p,a)
  np.testing.assert_allclose(core.transform(q,core.inverse(a)),p,atol=1e-5)
 def test_crop_offset(self):
  self.assertEqual(core.offset_box([1,2,3,4],[10,20,30,40]),[11,22,13,24])
  self.assertEqual(core.crop_bounds(np.array([[-10,-20],[30,40]]),(100,100),.25,0),[0,0,40,55])
class Fusion(unittest.TestCase):
 def test_hungarian_one_to_one_gate(self):
  r=[{'bbox':[0,0,10,10],'score':.9},{'bbox':[0,0,10,10],'score':.8}]; t=[r[0]]
  pairs=core.match(r,t,np.array([[1,0,0],[0,1,0]]),.3)
  self.assertEqual(len(pairs),1); self.assertAlmostEqual(pairs[0][2],1)
 def test_missing_evidence_hard_gate(self):
  self.assertEqual(core.decide([],[],[],0)[0],False)
 def test_uv_no_resurrection(self):
  self.assertEqual(core.uv_score(0,1),0); self.assertEqual(core.uv_score(.8,0),.4)
 def test_ds_conservative(self):
  m,k=core.combine(.9,.8,1)
  self.assertGreater(m[0],.5); self.assertAlmostEqual(sum(m),1); self.assertGreater(k,0)
 def test_known_metrics(self):
  m=core.metrics([1,1,0,0],[1,0,1,0]); self.assertEqual([m[k] for k in ['TP','FP','FN','TN']],[1,1,1,1]); self.assertEqual(m['FPR'],.5)
if __name__=='__main__': unittest.main()
