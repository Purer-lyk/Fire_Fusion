import unittest, tempfile, struct
from pathlib import Path
import audit_datasets as a

class AuditTests(unittest.TestCase):
    def test_jpeg_dimensions(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            p=Path(d)/'x.jpg'
            p.write_bytes(b'\xff\xd8\xff\xc0'+struct.pack('>H',17)+b'\x08'+struct.pack('>HH',256,192)+b'\x03'+b'\x00'*9+b'\xff\xd9')
            self.assertTrue(callable(getattr(a,'jpeg_size',None)), 'jpeg_size feature missing')
            self.assertEqual(a.jpeg_size(p),(192,256))
    def test_split_excludes_reserved_hash_and_temporal_neighbors(self):
        self.assertTrue(callable(getattr(a,'split_records',None)), 'split_records feature missing')
        records=[{'path':str(i),'sha256':str(i),'valid':True,'group':g,'boxes':[]} for i,g in enumerate(['a','a','b','b','c','d'])]
        records.append({'path':'dup','sha256':'2','valid':True,'group':'e','boxes':[]})
        train,val,excluded=a.split_records(records,{'0'})
        self.assertNotIn('0',{r['sha256'] for r in train+val})
        self.assertFalse({r['sha256'] for r in train}&{r['sha256'] for r in val})
        self.assertFalse({r['group'] for r in train}&{r['group'] for r in val})
        self.assertEqual(len(train+val+excluded),len(records))
        self.assertEqual(a.split_records(records,{'0'}),(train,val,excluded))
    def test_annotation_rejects_bad_bounds_and_preserves_negatives(self):
        self.assertTrue(callable(getattr(a,'read_annotation',None)), 'read_annotation feature missing')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            p=Path(d)/'a.xml'
            p.write_text('<annotation><size><width>192</width><height>256</height></size></annotation>')
            r=a.read_annotation(p,(192,256))
            self.assertTrue(r['valid']); self.assertEqual(r['boxes'],[])
            p.write_text('<annotation><size><width>192</width><height>256</height></size><object><name>smoke</name><bndbox><xmin>0</xmin><ymin>1</ymin><xmax>200</xmax><ymax>5</ymax></bndbox></object></annotation>')
            r=a.read_annotation(p,(192,256))
            self.assertFalse(r['valid']); self.assertIn('unexpected_class:smoke',r['issues']); self.assertIn('invalid_box',r['issues'])
    def test_yolo_rounding_agreement(self):
        self.assertTrue(callable(getattr(a,'yolo_agreement',None)), 'yolo_agreement feature missing')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            p=Path(d)/'a.txt'; p.write_text('0 0.388021 0.400391 0.057292 0.035156\n')
            self.assertTrue(a.yolo_agreement(p,(192,256),[[69,98,80,107]])['agrees'])
            p.write_text('1 0.388021 0.400391 0.057292 0.035156\n')
            self.assertFalse(a.yolo_agreement(p,(192,256),[[69,98,80,107]])['agrees'])
    def test_end_to_end_exact_hash_gt_only_and_coco_negatives(self):
        self.assertTrue(callable(getattr(a,'build',None)), 'build feature missing')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            root=Path(d); src=root/'src'; test=root/'test'
            for p in [src/'images',src/'annotations',test/'bgr',test/'thermal']: p.mkdir(parents=True)
            jpeg=b'\xff\xd8\xff\xc0'+struct.pack('>H',17)+b'\x08'+struct.pack('>HH',256,192)+b'\x03'+b'\x00'*9+b'\xff\xd9'
            for i in range(12):
                (src/'images'/f'{i}.jpg').write_bytes(jpeg+bytes([i]))
                (src/'annotations'/f'{i}.xml').write_text('<annotation><size><width>192</width><height>256</height></size></annotation>')
            (test/'bgr'/'same.jpg').write_bytes(jpeg+b'\x00')
            (test/'thermal'/'different.jpg').write_bytes(jpeg+b'\xff')
            (test/'paper_sample.txt').write_text('same.jpg different.jpg 400 1\n')
            result=a.build(root/'out',{'rgb':src},test)
            self.assertEqual(result['test']['rgb']['gt_covered'],1)
            self.assertEqual(result['test']['thermal']['gt_covered'],0)
            self.assertEqual(result['sources']['rgb']['train_images']+result['sources']['rgb']['val_images'],11)
            for split in ['train','val']:
                coco=__import__('json').loads((root/'out'/'datasets'/'rgb'/f'{split}.json').read_text())
                self.assertEqual(coco['annotations'],[])
                self.assertTrue(coco['images'])
    def test_yolo_boundary_rounding_is_not_real_out_of_bounds(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            p=Path(d)/'a.txt'; p.write_text('0 0.493750 0.867188 0.228125 0.265625\n')
            self.assertTrue(a.yolo_agreement(p,(640,640),[[243,470,389,640]])['agrees'])
if __name__=='__main__': unittest.main()
