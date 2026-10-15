import importlib
import math
import random
import unittest

class FusionTests(unittest.TestCase):
    def test_binary_rules_and_conflict(self):
        self.assertIsNotNone(importlib.util.find_spec('core'), 'Pure fusion module missing')
        c = importlib.import_module('core')
        a, b = c.Mass(.6,.3,.1), c.Mass(.2,.5,.3)
        ds = c.combine(a,b,'ds')
        self.assertAlmostEqual(ds.conflict,.36)
        self.assertAlmostEqual(ds.mass.fire,.32/.64)
        self.assertAlmostEqual(ds.mass.nofire,.29/.64)
        self.assertAlmostEqual(ds.mass.uncertain,.03/.64)
        sp = c.combine(a,b,'spatial',.1)
        self.assertAlmostEqual(sp.conflict,.324)
        self.assertAlmostEqual(sp.mass.fire,.032/.352)
        p = c.combine(c.Mass(1,0,0),c.Mass(0,1,0),'pcr')
        self.assertEqual(p.mass,c.Mass(.5,.5,0))
        for method in ('ds','spatial'):
            out=c.combine(c.Mass(1,0,0),c.Mass(0,1,0),method,.5)
            self.assertTrue(out.degenerate)
            self.assertEqual(out.mass,c.Mass(0,0,1))
        self.assertTrue(c.combine(c.Mass(1,0,0),c.Mass(1,0,0),'spatial',0).degenerate)
        rng=random.Random(8)
        for _ in range(500):
            a=c.Mass.normalized(*(rng.random() for _ in range(3)))
            b=c.Mass.normalized(*(rng.random() for _ in range(3)))
            for method in ('ds','pcr','spatial'):
                m=c.combine(a,b,method,rng.random()).mass
                self.assertAlmostEqual(sum(m.values()),1)
                self.assertTrue(all(math.isfinite(x) and 0<=x<=1 for x in m.values()))
        with self.assertRaises(ValueError): c.Mass(.2,.2,.2)
        with self.assertRaises(ValueError): c.Mass(float('nan'),0,1)
        with self.assertRaises(ValueError): c.combine(a,b,'spatial',None)
        self.assertAlmostEqual(c.uv_gate(.8,.4),.8*(.2+.8*.4))
        self.assertEqual(c.fuse3(a,b,c.Mass(0,0,1),'ds').mass,c.combine(a,b,'ds').mass)

class SensorTests(unittest.TestCase):
    def test_temporal_source_evidence(self):
        import core as c
        self.assertTrue(hasattr(c,'sensor_step'),'Source-evidence transition missing')
        state=c.SensorState()
        row=c.parse_sample_line('rgb.jpg thermal.jpg 394.038 0')
        self.assertEqual(row.temperature,394.038)
        self.assertEqual(row.uv,0)
        out=c.sensor_step(state,row,.8,.7,reliability=.3)
        self.assertAlmostEqual(out.rgb.fire,.7*.5+.3*.8*.3)
        self.assertAlmostEqual(out.thermal.fire,.7*.5+.3*(1-math.exp(-7))*(1-math.exp(-.1*(394.038-50))))
        q=.001-.1*.001*(1-.001)
        self.assertAlmostEqual(out.uv.fire,(.7*.1+.3*q)/(.7*.85+.3))
        self.assertEqual(state.rgb_history,())
        self.assertEqual(out.state.rgb_history,(.8,))
        cold=c.sensor_step(state,c.parse_sample_line('a.jpg b.jpg 40 1'),.8,.7)
        self.assertAlmostEqual(cold.thermal.fire,.35)
        self.assertGreater(out.thermal.fire,cold.thermal.fire)
        self.assertGreater(cold.uv.fire,out.uv.fire)
        for _ in range(30):
            out=c.sensor_step(out.state,row,.8,.7)
        self.assertEqual(len(out.state.rgb_history),10)
        self.assertEqual(len(out.state.thermal_history),10)
        for m in (out.rgb,out.thermal,out.uv): self.assertAlmostEqual(sum(m.values()),1)
        for text in ('a b 20 2','a b nan 0','a b 20','../a b 20 0'):
            with self.assertRaises(ValueError): c.parse_sample_line(text)
        with self.assertRaises(ValueError): c.sensor_step(state,row,1.1,.7)

class GeometryTests(unittest.TestCase):
    def test_explicit_registration_and_nonspatial_uv(self):
        import core as c
        self.assertTrue(hasattr(c,'project_box'),'Registration geometry missing')
        box=(1,2,4,6)
        h=((2,0,5),(0,3,-1),(0,0,1))
        self.assertEqual(c.project_box(box,h),(7,5,13,17))
        self.assertAlmostEqual(c.box_iou((0,0,2,2),(1,1,3,3)),1/7)
        self.assertEqual(c.box_iou((0,0,2,2),(3,3,4,4)),0)
        with self.assertRaises(ValueError): c.project_box(box,None)
        with self.assertRaises(ValueError): c.project_box(box,((1,0,0),(0,1,0),(0,0,0)))
        with self.assertRaises(ValueError): c.box_iou((2,2,1,1),box)
        with self.assertRaises(ValueError): c.project_box(box,((1,0,0),(0,1,0),(1,0,-2)))
        a,b,u=c.Mass(.7,.2,.1),c.Mass(.8,.1,.1),c.Mass(.1,.7,.2)
        result=c.fuse_rgb_thermal_uv(a,b,u,method='spatial',spatial_consistency=.4)
        expected=c.combine(c.combine(a,b,'spatial',.4).mass,u,'ds')
        self.assertEqual(result.mass,expected.mass)
        with self.assertRaises(ValueError): c.fuse_rgb_thermal_uv(a,b,u,method='spatial')
        self.assertEqual(c.decide(c.Mass(.96,.03,.01),.95),'fire')
        self.assertEqual(c.decide(c.Mass(.01,.96,.03),.95),'nofire')
        self.assertEqual(c.decide(c.Mass(.4,.3,.3),.95),'uncertain')

if __name__=='__main__': unittest.main()
