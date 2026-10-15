import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'spatial_reliability_ds'))
import spatial_core as spatial
try:
    import gate_core as gate
except ImportError:
    class MissingPolicy:
        @staticmethod
        def decide(iou, matched, mass, tau_l=.1, tau_h=.3):
            return {'label':spatial.binary(mass),'state':'FireCandidate'}
    gate=MissingPolicy()

class GateTests(unittest.TestCase):
    def test_missing_cannot_fire_even_uv_dogmatic(self):
        self.assertEqual(gate.decide(0,False,[1,0,0])['label'],0)

    def test_spatial_transition_never_binary_fire(self):
        for value in [0,.05,.1,.100001,.2,.299999]:
            result=gate.decide(value,True,[1,0,0])
            self.assertEqual(result['label'],0)
            self.assertEqual(result['state'],'Transition' if .1<value<.3 else 'NoFire')
        self.assertEqual(gate.decide(.3,True,[.7,.1,.2])['label'],1)
        self.assertEqual(gate.decide(.8,True,[.1,.7,.2])['label'],0)
        self.assertEqual(gate.decide(.8,True,[0,0,1])['label'],0)

    def test_frame_discount_matching_and_uv_cannot_override(self):
        import numpy as np
        A=[[1,0,0],[0,1,0]]
        box=lambda xy,c=.9: {'bbox_xyxy':xy,'score':c,'class_id':0}
        frame=getattr(gate,'frame',lambda *args,**kwargs: {'final_visual_uv':1})
        absent=frame([box([0,0,10,10])],[],A,1)
        self.assertEqual(absent['final_visual_uv'],0)
        zero=frame([box([0,0,10,10])],[box([20,20,30,30])],A,1)
        self.assertEqual(zero['final_visual_uv'],0)
        trans=frame([box([0,0,10,10])],[box([0,0,2,10])],A,1)
        self.assertEqual(trans['state'],'Transition')
        self.assertEqual(trans['final_visual_uv'],0)
        self.assertAlmostEqual(trans['g'],.5,places=6)
        self.assertAlmostEqual(sum(trans['visual_mass']),1)
        yes=frame([box([0,0,10,10])],[box([0,0,10,10])],A,1)
        self.assertEqual(yes['final_visual'],1)
        self.assertEqual(yes['final_visual_uv'],1)
        multiple=frame([box([0,0,10,10]),box([0,0,10,10])],[box([0,0,10,10])],A,1)
        matched=[c for c in multiple['candidates'] if c['status']=='matched']
        self.assertEqual(len(matched),1)
        self.assertEqual(len({c['rgb_index'] for c in matched}),1)
        self.assertEqual(len({c['thermal_index'] for c in matched}),1)
        shear=s=spatial.affine([0,0,10,10],[[1,.5,0],[0,1,0]])
        self.assertEqual(s.shape,(4,2))
        self.assertNotEqual(spatial.iou(spatial.polygon([0,0,10,10]),shear),10/15)

    def test_full_run_deliverable_contract(self):
        import subprocess,json,csv,zipfile
        out=Path(__file__).resolve().parent
        runner=out/'run_experiment.py'
        self.assertTrue(runner.exists(),'new full-run implementation missing')
        subprocess.run([sys.executable,str(runner)],check=True)
        with (out/'fusion_results.csv').open(encoding='utf-8-sig') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),1528)
        self.assertEqual([int(r['pair_index']) for r in rows],list(range(1528)))
        for r in rows:
            if r['semantic_state']!='FireCandidate':
                self.assertEqual(int(r['gate_visual_binary']),0)
                self.assertEqual(int(r['gate_visual_uv_binary']),0)
        verification=json.loads((out/'deliverable_verification.json').read_text(encoding='utf-8'))
        self.assertTrue(verification['metrics_recomputed_from_csv'])
        self.assertTrue(verification['old_files_unchanged'])
        with zipfile.ZipFile(verification['report_path']) as z: self.assertIsNone(z.testzip())

if __name__=='__main__': unittest.main(verbosity=2)
