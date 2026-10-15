import unittest
import continuous_core as c

class ContinuousTests(unittest.TestCase):
 def test_predeclared_continuous_mass(self):
  self.assertTrue(hasattr(c,'gain'),'continuous reliability is missing')
  for s,g in [(0,0),(.01,0),(.1,0),(.2,.5),(.3,1),(.9,1)]:
   self.assertAlmostEqual(c.gain(s),g)
   m=c.mass(.49,.8,s)
   self.assertAlmostEqual(sum(m),1)
   self.assertEqual(m[1],0)
   self.assertAlmostEqual(m[0],1-(1-.49)*(1-.1*g*.8))
  self.assertAlmostEqual(c.mass(.49,.8,.3)[0],c.mass(.49,.8,1)[0])
  self.assertEqual(c.mass(.8,.9,.01)[0],.8)

class SelectionTests(unittest.TestCase):
 def test_reselect_all_matches_not_previous_winner(self):
  self.assertTrue(hasattr(c,'select'),'full-candidate reselection is missing')
  pairs=[dict(rgb_index=0,thermal_index=0,cR=.49,cT=.9,s=.05),dict(rgb_index=1,thermal_index=1,cR=.5,cT=.5,s=.4)]
  self.assertEqual(c.select(pairs,continuous=False)['rgb_index'],0)
  self.assertEqual(c.select(pairs)['rgb_index'],1)
  self.assertEqual(c.select([dict(rgb_index=0,thermal_index=0,cR=.5,cT=.9,s=.05)])['fire'],0)
  self.assertEqual(c.select([dict(rgb_index=0,thermal_index=0,cR=.8,cT=.9,s=.05)])['fire'],1)
  self.assertIsNone(c.select([])['rgb_index'])

if __name__=='__main__':unittest.main()
