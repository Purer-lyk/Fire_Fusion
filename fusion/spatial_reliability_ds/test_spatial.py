import unittest
import importlib.util
import numpy as np

class EvidenceTests(unittest.TestCase):
    def test_discount_missing_and_ds_identity(self):
        self.assertIsNotNone(importlib.util.find_spec('spatial_core'), 'new evidence implementation missing')
        import spatial_core as s
        a=s.base(.8,.2)
        np.testing.assert_allclose(a,[.64,.16,.2])
        np.testing.assert_allclose(s.discount(a,0),[0,0,1])
        np.testing.assert_allclose(s.discount(a,1),a)
        np.testing.assert_allclose(s.discount(a,.5),[.32,.08,.6])
        self.assertAlmostEqual(sum(s.discount(a,.37)),1)
        np.testing.assert_allclose(s.base(None,.2),[0,0,1])
        np.testing.assert_allclose(s.ds(a,s.base(None,.2))[0],a)
        self.assertEqual(s.binary([0,0,1]),0)
        self.assertEqual(s.decision([0,0,1],.8),'uncertain')
        np.testing.assert_allclose(s.ds([1,0,0],[0,1,0])[0],[0,0,1])
        self.assertEqual(s.decision([0,.9,.1],.8),'nofire')

class GeometryTests(unittest.TestCase):
    def test_polygon_matching_translation_rotation_one_to_one(self):
        import spatial_core as s
        self.assertTrue(hasattr(s,'match'), 'one-to-one geometry missing')
        identity=np.array([[1,0,0],[0,1,0]])
        d=lambda b,c=.9: {'bbox_xyxy':b,'score':c}
        rgb=[d([0,0,2,2]),d([3,0,5,2])]
        thermal=[d([0,0,2,2]),d([3,0,5,2])]
        result=s.match(rgb,thermal,identity,.01)
        self.assertEqual([(x['rgb_index'],x['thermal_index']) for x in result],[(0,0),(1,1)])
        self.assertEqual(len({x['thermal_index'] for x in result}),2)
        self.assertAlmostEqual(result[0]['iou'],1)
        translation=np.array([[1,0,1],[0,1,0]])
        self.assertAlmostEqual(s.iou(s.polygon([0,0,2,2]),s.affine([0,0,2,2],translation)),1/3,places=6)
        rotation=np.array([[0,-1,2],[1,0,0]])
        self.assertAlmostEqual(s.iou(s.polygon([0,0,2,2]),s.affine([0,0,2,2],rotation)),1)
        self.assertEqual(s.match(rgb,[],identity,.01)[0]['thermal_index'],None)
        self.assertEqual(s.match([],[],identity,.01),[])
        self.assertEqual(len(s.match([d([0,0,2,2])],[d([10,0,12,2])],identity,.01)),2)
        # Hungarian optimum is global, not duplicate independent best edges.
        repeated=s.match([rgb[0],rgb[0]],thermal[:1],identity,.01)
        self.assertEqual(sum(x['thermal_index'] is not None for x in repeated),1)

class PipelineTests(unittest.TestCase):
    def test_fixed_candidate_ablation_and_confusion(self):
        self.assertIsNotNone(importlib.util.find_spec('run_experiment'), 'fixed experiment pipeline missing')
        import run_experiment as r
        cfg=r.load_config()
        rgb=[{'bbox_xyxy':[0,0,2,2],'score':.9}]
        thermal=[{'bbox_xyxy':[0,0,2,2],'score':.8}]
        masses,detail=r.frame(rgb,thermal,[[1,0,0],[0,1,0]],42,1,[.001,.7992,.1998],cfg)
        self.assertEqual(detail['selected']['rgb_index'],0)
        self.assertEqual(detail['selected']['thermal_index'],0)
        np.testing.assert_allclose(masses['ordinary_ds'],masses['spatial_ds'])
        self.assertGreater(masses['spatial_uv'][0],masses['spatial_ds'][0])
        empty,detail=r.frame([],[],[[1,0,0],[0,1,0]],42,0,[.001,.7992,.1998],cfg)
        np.testing.assert_allclose(empty['spatial_ds'],[0,0,1])
        np.testing.assert_allclose(empty['spatial_uv'],[0,.9,.1])
        m=r.metrics([1,1,0,0],[1,0,1,0])
        self.assertEqual([m[k] for k in ['TP','FP','FN','TN']],[1,1,1,1])
        self.assertEqual(m['accuracy'],.5)
        self.assertEqual(m['F1'],.5)
        self.assertEqual(r.temperature_reliability(123,cfg),1)

if __name__=='__main__': unittest.main()
