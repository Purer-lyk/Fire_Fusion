import unittest
import importlib.util
from pathlib import Path

class MetricsTests(unittest.TestCase):
    def test_known_confusion(self):
        path=Path(__file__).with_name('run_supplied_models.py')
        self.assertTrue(path.exists(), 'UV reference evaluator not implemented')
        spec=importlib.util.spec_from_file_location('runner',path)
        m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        r=m.binary_metrics([1,1,1,0,0,0],[1,1,0,1,0,0])
        self.assertEqual((r['TP'],r['FP'],r['FN'],r['TN']),(2,1,1,2))
        for k in ['accuracy','precision','recall','F1']: self.assertAlmostEqual(r[k],2/3)
        self.assertIsNone(m.binary_metrics([0],[0])['precision'])
        with self.assertRaises(ValueError): m.binary_metrics([1],[1,0])

class AlignmentTests(unittest.TestCase):
    def test_exact_ids_names_and_order(self):
        path=Path(__file__).with_name('run_supplied_models.py')
        spec=importlib.util.spec_from_file_location('runner',path)
        m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        self.assertTrue(hasattr(m,'check_alignment'), 'alignment validation missing')
        samples=[m.c.Sample('a.jpg','b.jpg',50,1),m.c.Sample('c.jpg','d.jpg',51,0)]
        rows=[dict(pair_index=0,image_name='a.jpg'),dict(pair_index=1,image_name='c.jpg')]
        m.check_alignment(samples,rows,'rgb')
        with self.assertRaises(ValueError): m.check_alignment(samples,rows[::-1],'rgb')
        with self.assertRaises(ValueError): m.check_alignment(samples,[rows[0],rows[0]],'rgb')
        with self.assertRaises(ValueError): m.check_alignment(samples,rows[:1],'rgb')
        with self.assertRaises(ValueError): m.check_alignment(samples,rows,'thermal')

class RealLoaderTests(unittest.TestCase):
    def test_real_predictions_keep_original_paths_and_pixels(self):
        path=Path(__file__).with_name('run_supplied_models.py')
        spec=importlib.util.spec_from_file_location('runner',path)
        m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        model=m.YOLO(str(m.ATT/'best.pt'))
        samples=[m.c.parse_sample_line(l) for l in (m.SOURCE/'paper_sample.txt').read_text().splitlines()][:2]
        paths=[str(m.SOURCE/'bgr'/s.rgb) for s in samples]
        results=m.predict_images(model,paths,'0' if m.torch.cuda.is_available() else 'cpu')
        self.assertEqual([str(Path(r.path).resolve()) for r in results],[str(Path(p).resolve()) for p in paths])
        for r,p in zip(results,paths): m.np.testing.assert_array_equal(r.orig_img,m.cv2.imread(p))

if __name__=='__main__': unittest.main()
