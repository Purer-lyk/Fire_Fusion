import unittest, importlib.util
import numpy as np
from pathlib import Path
class Geometry(unittest.TestCase):
 def test_inverse_crop_offsets(self):
  p=Path(__file__).with_name('dual_core.py');self.assertTrue(p.exists(),'geometry implementation missing')
  sp=importlib.util.spec_from_file_location('core',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
  A=np.array([[.7,.01,78],[-.04,.96,-16]])
  pts=np.array([[0,0],[100,20],[320,320],[12,311]],float)
  np.testing.assert_allclose(m.project(m.project(pts,A),m.inverse(A)),pts,atol=1e-10)
  self.assertEqual(m.offset([1,2,10,12],[30,40,60,80]),[31,42,40,52])
class Montage(unittest.TestCase):
 def test_unequal_sensor_heights(self):
  import dual_core as m
  self.assertTrue(callable(getattr(m,'montage',None)),'sensor dimensions display helper missing')
  z=m.montage([np.zeros((320,320,3),np.uint8),np.zeros((256,320,3),np.uint8)])
  self.assertEqual(z.shape,(320,640,3))
class Execution(unittest.TestCase):
 def test_runner_stages_exist(self):
  p=Path(__file__).with_name('run_dual.py');self.assertTrue(p.exists(),'real-data runner missing')
  sp=importlib.util.spec_from_file_location('runner',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
  for name in ['diagnose','fresh_roi','candidate_bounds','evaluate','report']:self.assertTrue(callable(getattr(m,name,None)))
class Gate(unittest.TestCase):
 def test_missing_modality_cannot_fire(self):
  import run_dual as m
  cfg=dict(tR=.005,tT=.005,gate=.001,rR=1.,rT=.3,soft=False,u0=0,threshold=.4)
  b=[dict(bbox=[0,0,10,10],score=.99)]
  for r,t in [(b,[]),([],b),([],[])]:
   o=m.frame(r,t,np.zeros((len(r),len(t))),cfg);self.assertEqual(o['fire'],0);self.assertEqual(o['mU'],1.)
   self.assertFalse(o['mF']*(.5+.5*1)>.5 and o['selected'] is not None)
 def test_DS_mass_cornercases(self):
  import run_dual as m
  from optimized_core import pair_mass,combine
  cfg=dict(rR=1.,rT=.3,soft=False,u0=0)
  o=pair_mass(.8,.5,.01,cfg);self.assertAlmostEqual(o['mF'],.83);self.assertAlmostEqual(o['mF']+o['mU']+o['mN'],1.)
  z,k=combine([1.,0.,0.],[0.,1.,0.]);self.assertEqual(list(z),[0.,0.,1.]);self.assertEqual(k,1.)
class Verification(unittest.TestCase):
 def test_real_verifier_exists(self):
  self.assertTrue(Path(__file__).with_name('verify_dual.py').exists(),'independent verifier missing')
if __name__=='__main__':unittest.main()
