import unittest
import numpy as np
from fuse320 import corners, transform, polygon_iou, match, masses, aligned
class Tests(unittest.TestCase):
 def test_mass(self):
  np.testing.assert_allclose(masses(.2,.1),(.28,0,.72)); self.assertAlmostEqual(sum(masses(.7,.8)),1)
 def test_polygon(self):
  p=corners([0,0,2,2]); self.assertAlmostEqual(polygon_iou(p,p),1);self.assertEqual(polygon_iou(p,corners([3,3,4,4])),0)
  np.testing.assert_allclose(transform(p,np.array([[1,0,2],[0,1,3]])),p+[2,3])
 def test_match(self):
  r=[{'score':.2},{'score':.199999}];t=[{'score':.1},{'score':.9}]
  out=match(r,t,np.array([[.01,0],[1,1]]));self.assertEqual([(x['r'],x['t']) for x in out],[(0,0)])
  self.assertFalse(masses(.2,.1)[0]>.3)
  out=match([{'score':.8},{'score':.7}],[{'score':.6},{'score':.9}],np.array([[.9,.8],[.7,0]]));self.assertEqual(len(out),2);self.assertEqual(len({x['t'] for x in out}),2)
 def test_align(self):
  aligned([{'pair_index':0,'image_name':'a'}],['a'])
  with self.assertRaises(ValueError):aligned([{'pair_index':1,'image_name':'a'}],['a'])
if __name__=='__main__':unittest.main()
