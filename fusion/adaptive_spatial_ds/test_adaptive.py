import unittest, importlib.util
from pathlib import Path
import numpy as np
class TileTest(unittest.TestCase):
 def test_fixed_tiles_cover_native_without_resize_or_uv(self):
  spec=importlib.util.spec_from_file_location('core',Path(__file__).with_name('adaptive_core.py'))
  self.assertIsNotNone(spec, 'adaptive core module required')
  self.assertTrue(Path(spec.origin).exists(),'fixed scene-independent tiled detector is not implemented')
  c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
  for h,w in [(320,320),(256,192)]:
   tiles=c.tiles((h,w));self.assertEqual(len(tiles),4);mask=np.zeros((h,w),int)
   for x,y,X,Y in tiles:mask[y:Y,x:X]+=1
   self.assertTrue(np.all(mask>0));self.assertGreater(np.mean(mask>1),0)
  self.assertEqual(c.map_box([1,2,5,8],[10,20,100,100]),[11,22,15,28])
class FusionTest(unittest.TestCase):
 def test_actual_polygon_gate_and_normalized_conflict_masses(self):
  import adaptive_core as c
  self.assertTrue(hasattr(c,'fuse'),'strict no-UV dual-detection fusion missing')
  A=np.array([[1.,0,0],[0,1.,0]])
  r=[dict(bbox=[0,0,10,10],score=.8)];t=[dict(bbox=[30,30,40,40],score=.9)]
  o=c.fuse(r,t,A);self.assertEqual(o['fire'],0);self.assertEqual(o['masses'],[0.,0.,1.])
  t[0]['bbox']=[0,0,10,10];o=c.fuse(r,t,A)
  self.assertGreater(o['iou'],.99);self.assertAlmostEqual(sum(o['masses']),1)
  self.assertEqual(c.fuse(r,[],A)['fire'],0)
  m,k=c.combine([1.,0,0],[0,1.,0]);self.assertEqual(m,[0.,0.,1.]);self.assertEqual(k,1.)
  self.assertAlmostEqual(c.expected_iou(r[0]['bbox'],t[0]['bbox'],[A,A])['mean'],1.)
class RegistrationTest(unittest.TestCase):
 def test_feature_masks_and_holdout_acceptance(self):
  import adaptive_core as c
  self.assertTrue(hasattr(c,'feature_proposal'),'image-only robust registration missing')
  blank=np.zeros((256,192),np.uint8)
  o=c.feature_proposal(blank,blank,[],[],'gradient_sift');self.assertFalse(o['accepted'])
  mask=c.background_mask(blank.shape,[dict(bbox=[50,60,80,90],score=.9)])
  self.assertEqual(mask[70,60],0);self.assertEqual(mask[0,100],0);self.assertEqual(mask[40,40],255)
  p=np.array([[0.,0],[10,0],[0,10],[10,10],[5,5],[8,8],[2,8]])
  mats,receipt=c.bootstrap_affines(p,p,np.array([[1.,0,0],[0,1.,0]]),32)
  self.assertEqual(len(mats),32);self.assertLess(receipt['loo_rmse'],1e-5)
  self.assertTrue(all(np.isfinite(a).all() for a in mats))
class PipelineTest(unittest.TestCase):
 def test_protocol_and_real_smoke(self):
  p=Path(__file__).with_name('run_adaptive.py')
  self.assertTrue(p.exists(),'frozen real multiscale forward runner missing')
  import run_adaptive as r
  r.prepare();receipt=r.fresh(limit=3,suffix='smoke')
  self.assertEqual(receipt['actual_detector_views'],36)
  self.assertTrue(receipt['all_image_hashes_verified'])
  self.assertEqual(set(x[-1] for x in receipt['input_shapes']),{320,640})
class OverlayMaskTest(unittest.TestCase):
 def test_burned_in_red_yellow_excluded_without_flame_alignment(self):
  import adaptive_core as c
  self.assertTrue(hasattr(c,'overlay_mask'),'burned-in text/box pixel mask missing')
  im=np.zeros((80,80,3),np.uint8);im[30:35,30:60]=(0,0,255);im[50:55,30:60]=(0,255,255)
  m=c.overlay_mask(im);self.assertEqual(m[32,40],0);self.assertEqual(m[52,40],0);self.assertEqual(m[65,65],255)
class EvaluationTest(unittest.TestCase):
 def test_full_evaluation_has_32_bounded_trials_and_weak_supervision(self):
  import run_adaptive as r
  self.assertTrue(hasattr(r,'evaluate'),'groupwise calibrated real spatial model evaluation missing')
  r.evaluate()
  m=r.load(r.OUT/'metrics.json');self.assertEqual(m['trials'],32);self.assertFalse(m['independent_test']);self.assertFalse(m['UV_runtime'])
  self.assertEqual(len(r.load(r.OUT/'frame_evidence.json')),1528)
  self.assertTrue((r.OUT/'candidate_bounds_before_calibration.json').exists())
class AnchorTest(unittest.TestCase):
 def test_anchor_retains_real_full320_rgb_and_requires_tiled_thermal_overlap(self):
  import adaptive_core as c
  self.assertTrue(hasattr(c,'anchor_fuse'),'asymmetric native RGB and multiview thermal fusion missing')
  a=np.array([[1.,0,0],[0,1.,0]]);r=[dict(bbox=[0,0,10,10],score=.49,view='full320')];t=[dict(bbox=[0,0,10,10],score=.9,view='tile0_320')]
  o=c.anchor_fuse(r,t,a);self.assertEqual(o['fire'],1);self.assertAlmostEqual(o['masses'][0],1-.51*.91)
  t[0]['bbox']=[20,20,30,30];self.assertEqual(c.anchor_fuse(r,t,a)['fire'],0)
if __name__=='__main__':unittest.main()
