"""Independent serialized-output verification; no source writes."""
from pathlib import Path
import json, hashlib, collections
root=Path('C:/lyk/fire/qualityDetection')
report=json.loads((root/'audit/audit.json').read_text())
manifest=json.loads((root/'audit/manifest.json').read_text())
test=json.loads((root/'audit/test_manifest.json').read_text())
reserved={r['sha256'] for r in test}
by_path={r['path']:r for r in manifest+test}
checks={}
for modality in ['rgb','thermal']:
    splitsets={}
    for split in ['train','val','test_existing_gt']:
        c=json.loads((root/'datasets'/modality/(split+'.json')).read_text())
        ids={i['id']:i for i in c['images']}; assert len(ids)==len(c['images'])
        assert len({i['id'] for i in c['annotations']})==len(c['annotations'])
        for i in c['images']:
            r=by_path[i['file_name']]; assert Path(i['file_name']).is_file()
            assert (i['width'],i['height'])==(r['width'],r['height'])
            if split!='test_existing_gt': assert r['valid'] and r['split']==split and r['sha256'] not in reserved
            else: assert r['gt_status']=='exact_hash_existing_annotation' and r['annotation_provenance']
        for b in c['annotations']:
            i=ids[b['image_id']]; x,y,w,h=b['bbox']
            assert b['category_id']==1 and 0<=x<x+w<=i['width'] and 0<=y<y+h<=i['height'] and b['area']==w*h
        annotated={b['image_id'] for b in c['annotations']}
        checks[modality+'_'+split]={'images':len(ids),'boxes':len(c['annotations']),'negatives':len(ids)-len(annotated)}
        splitsets[split]={by_path[i['file_name']]['sha256'] for i in c['images']}
    assert not splitsets['train']&splitsets['val']
    assert not (splitsets['train']|splitsets['val'])&reserved
    rs=[r for r in manifest if r['modality']==modality]
    train_groups={r['split_group'] for r in rs if r['split']=='train'}
    val_groups={r['split_group'] for r in rs if r['split']=='val'}
    assert not train_groups&val_groups
    assert len(rs)==sum(report['sources'][modality][s+'_images'] for s in ['train','val','excluded'])
# Rehash every input independently to verify manifests, not just serialize-time assertions.
for r in manifest+test:
    with Path(r['path']).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
checks['all_input_image_hashes_reverified']=len(manifest)+len(test)
checks['cross_modality_source_duplicate_groups']=sum(len({by_path[p]['modality'] for p in d['paths']})>1 for d in report['internal_source_duplicate_hashes'])
checks['reserved_overlap_retained']=0
checks['all_assertions_passed']=True
(root/'audit/verification.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
