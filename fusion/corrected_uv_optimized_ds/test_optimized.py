import unittest, importlib, numpy as np
class FusionTests(unittest.TestCase):
 def test_discount_mass(self):
  try: c=importlib.import_module('optimized_core')
  except ModuleNotFoundError: self.fail('optimized_core BPA reliability implementation missing')
  np.testing.assert_allclose(c.bpa(.8,0,.1),[0,0,1])
  np.testing.assert_allclose(c.bpa(.8,1,.1),[.72,0,.28])
  for r in [0,.1,.7,1]: self.assertAlmostEqual(sum(c.bpa(.8,r,.1)),1)
 def test_no_uv_override_missing_detection_and_threshold(self):
  import optimized_core as c
  self.assertTrue(hasattr(c,'frame'),'frame decision implementation missing')
  cfg=dict(tR=.2,tT=.05,gate=.01,rR=1.,rT=.1,u0=0.,soft=False,threshold=.5)
  self.assertEqual(c.frame([],[],np.zeros((0,0)),cfg)['fire'],0)
  self.assertEqual(c.frame([dict(score=.5,bbox=[0,0,1,1])],[],np.zeros((1,0)),cfg)['fire'],0)
  rb=[dict(score=.6,bbox=[0,0,1,1])];tb=[dict(score=.7,bbox=[0,0,1,1])]
  out=c.frame(rb,tb,np.array([[.2]]),cfg)
  self.assertEqual(out['fire'],1);self.assertAlmostEqual(out['mF']+out['mN']+out['mU'],1)
  self.assertEqual(c.frame(rb,tb,np.array([[.001]]),cfg)['fire'],0)
  self.assertEqual(c.frame(rb,tb,np.array([[.2]]),dict(cfg,threshold=out['mF']))['fire'],0)
  self.assertNotIn('UV',c.frame.__code__.co_varnames)
 def test_one_to_one_matches(self):
  import optimized_core as c
  rb=[dict(score=.8,bbox=[0,0,1,1])]*2;tb=rb
  cfg=dict(tR=.2,tT=.05,gate=.01,rR=1.,rT=.1,u0=0.,soft=False,threshold=.5)
  out=c.frame(rb,tb,np.array([[.9,.2],[.3,.8]]),cfg)
  self.assertEqual(len(out['matches']),2)
  self.assertEqual(len(set(x['r'] for x in out['matches'])),2)
  self.assertEqual(len(set(x['t'] for x in out['matches'])),2)
 def test_selection_only_training_indices(self):
  import optimized_core as c
  self.assertTrue(hasattr(c,'select'),'selection implementation missing')
  p=np.array([[1,0,1,0],[0,0,1,1]],bool); y=np.array([1,0,0,1]); train=np.array([0,1])
  k=c.select(p,y,train)
  y[2:]=1-y[2:]
  self.assertEqual(k,c.select(p,y,train));self.assertEqual(k,0)
 def test_protocol_before_search_and_temporal_group_isolation(self):
  import pathlib
  p=pathlib.Path(__file__).parent/'run_optimization.py'
  self.assertTrue(p.exists(),'end-to-end runner missing')
  import run_optimization as r
  groups=r.temporal_groups(['2026_05_23_07_00_00_0000_bgr.jpg','2026_05_23_07_00_01_0000_bgr.jpg','2026_05_23_07_06_00_0000_bgr.jpg'])
  self.assertEqual(groups,[0,0,1])
  for tr,te in r.folds(np.array(groups)):
   self.assertFalse(set(tr)&set(te));self.assertFalse(set(np.array(groups)[tr])&set(np.array(groups)[te]))
 def test_complementary_ds_conflict_and_soft_discount(self):
  import optimized_core as c
  m,k=c.combine(np.array([.8,.1,.1]),np.array([.1,.8,.1]))
  self.assertGreater(k,0);self.assertAlmostEqual(sum(m),1)
  cfg=dict(rR=1,rT=.3,u0=.1,soft=True)
  a=c.pair_mass(.8,.8,.01,cfg);b=c.pair_mass(.8,.8,.3,cfg)
  self.assertLess(a['rT_effective'],b['rT_effective']);self.assertLess(a['mF'],b['mF'])
 def test_verifier_independent_confusion_and_geometry(self):
  import pathlib
  p=pathlib.Path(__file__).parent/'verify_optimization.py'
  self.assertTrue(p.exists(),'independent readback validator missing')
  import verify_optimization as v
  out=v.independent_metrics([1,1,0,0],[1,0,1,0])
  self.assertEqual((out['TP'],out['FP'],out['FN'],out['TN']),(1,1,1,1))
  from fuse320 import transform,polygon_iou,corners
  box=corners([0,0,10,10]);mapped=transform(box,[[1,0,2],[0,1,3]])
  np.testing.assert_allclose(mapped,corners([2,3,12,13]));self.assertAlmostEqual(polygon_iou(box,box),1)
if __name__=='__main__': unittest.main()
