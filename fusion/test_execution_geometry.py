import unittest
import numpy as np
import core as c

class ExecutedGeometryTests(unittest.TestCase):
    def test_polygon_iou_not_axis_aligned_envelope(self):
        square=c.box_polygon((0,0,2,2))
        diamond=np.array([[1,0],[2,1],[1,2],[0,1]],float)
        self.assertAlmostEqual(c.polygon_iou(square,diamond),.5)
        self.assertAlmostEqual(c.box_iou((0,0,2,2),(0,0,2,2)),1.)
        self.assertEqual(c.polygon_iou(square,c.box_polygon((4,4,5,5))),0.)
        self.assertAlmostEqual(c.polygon_iou(square,square),1.)

    def test_homography_translation_and_horizon(self):
        np.testing.assert_allclose(c.project_polygon((0,0,2,2),np.array([[1,0,3],[0,1,4],[0,0,1]])),c.box_polygon((3,4,5,6)))
        with self.assertRaises(ValueError): c.project_polygon((0,0,2,2),[[1,0,0],[0,1,0],[1,0,-1]])
        with self.assertRaises(ValueError): c.project_polygon((0,0,2,2),None)

    def test_spatial_denominator_cancels_after_normalize(self):
        a=c.Mass(.6,.3,.1); b=c.Mass(.2,.5,.3)
        s=.4
        expected=c.Mass.normalized(s*.32,.29,.03)
        out=c.combine(a,b,'spatial',s)
        np.testing.assert_allclose(out.mass.values(),expected.values(),rtol=0,atol=1e-15)
        self.assertAlmostEqual(out.conflict,(1-s)*.36)

    def test_legacy_and_recommended_are_not_same_algorithm(self):
        a=c.Mass(.7,.2,.1); b=c.Mass(.8,.1,.1); uv=c.Mass(.2,.6,.2)
        legacy=c.fuse3(a,b,uv,'spatial',.1)
        expected=c.combine(c.combine(a,b,'spatial',.1).mass,uv,'spatial',.1)
        np.testing.assert_allclose(legacy.mass.values(),expected.mass.values())
        recommended=c.fuse_rgb_thermal_uv(a,b,uv,'spatial',.1)
        self.assertNotAlmostEqual(legacy.mass.fire,recommended.mass.fire)
        self.assertFalse(legacy.degenerate)

if __name__=='__main__': unittest.main()
