import unittest, inspect
import numpy as np
import run_affine_iou as r
class Tests(unittest.TestCase):
 def test_translation(self):
  p=r.affine_polygon([0,0,10,10],[[1,0,5],[0,1,0]])
  self.assertAlmostEqual(r.c.polygon_iou(r.c.box_polygon([0,0,10,10]),p),1/3,places=7)
  np.testing.assert_array_equal(p,[[5,0],[15,0],[15,10],[5,10]])
 def test_rotation_polygon_not_aabb(self):
  # 45-degree rotation creates diamond, contained in its AABB.
  s=np.sqrt(.5); a=[[s,-s,0],[s,s,0]]; p=r.affine_polygon([-1,-1,1,1],a)
  box=[-np.sqrt(2),-np.sqrt(2),np.sqrt(2),np.sqrt(2)]
  actual=r.c.polygon_iou(r.c.box_polygon(box),p)
  self.assertAlmostEqual(actual,.5,places=6)
  aabb=[*p.min(axis=0),*p.max(axis=0)]
  self.assertAlmostEqual(r.c.box_iou(box,aabb),1,places=6)
 def test_reject_projective(self):
  with self.assertRaises(ValueError): r.affine_polygon([0,0,1,1],np.eye(3))
 def test_matching_confidence(self):
  d=lambda b,s:dict(bbox_xyxy=b,score=s)
  rgb={'detections':[d([0,0,10,10],.8)]}; thermal={'detections':[d([30,30,40,40],.99),d([0,0,10,10],.7)]}
  p=r.pairing(rgb,thermal,[[1,0,0],[0,1,0]])
  self.assertEqual(p['s'],1); self.assertEqual(p['thermal_confidence'],.7); self.assertEqual(p['thermal_detection_index'],1)
 def test_missing_and_no_overlap(self):
  d=dict(bbox_xyxy=[20,20,30,30],score=.9); p=r.pairing({'detections':[]},{'detections':[d]},[[1,0,0],[0,1,0]])
  self.assertEqual((p['s'],p['status'],p['thermal_confidence']),(0,'missing_rgb',.9))
  p=r.pairing({'detections':[dict(bbox_xyxy=[0,0,10,10],score=.8)]},{'detections':[d]},[[1,0,0],[0,1,0]])
  self.assertEqual((p['s'],p['status']),(0,'both_present_no_overlap'))
 def test_ties(self):
  ds=[dict(bbox_xyxy=[0,0,2,2],score=.7),dict(bbox_xyxy=[0,0,2,2],score=.9)]
  p=r.pairing({'detections':[ds[0]]},{'detections':ds},[[1,0,0],[0,1,0]])
  self.assertEqual(p['thermal_detection_index'],1)
 def test_metrics(self):
  m=r.metrics([1,1,0,0],[1,0,1,0]); self.assertEqual([m[k] for k in ['TP','FP','FN','TN']],[1,1,1,1]); self.assertEqual(m['F1'],.5)
 def test_no_projective_or_fixed_spatial_primary(self):
  source=inspect.getsource(r.run)
  self.assertNotIn('findHomography',source); self.assertNotIn('project_polygon',source); self.assertNotIn("'spatial',.1",source)
  self.assertIn("'spatial',pair['s']",source)
 def test_full_affine(self):
  a,cal=r.calibration(); self.assertEqual(a.shape,(2,3)); self.assertEqual(len(cal['inlier_mask']),7); self.assertEqual(len(cal['leave_one_out_errors_px']),7)
if __name__=='__main__': unittest.main(verbosity=2)
