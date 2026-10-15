from pathlib import Path
import json,hashlib,shutil,re,collections,cv2
import grounded_core as c
OUT=Path(__file__).resolve().parent;SRC=Path('C:/lyk/fire/TorchDetection');TEST=Path('C:/lyk/paper_data/paper_data_/paper_data')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
testhash={sha(TEST/folder/line.split()[i]) for line in (TEST/'paper_sample.txt').read_text().splitlines() if line.strip() for folder,i in [('bgr',0),('thermal',1)]};assert len(testhash)==3056
records=json.loads((OUT/'training_dataset_audit.json').read_text())['records'];manifest=[];summary={};excluded=[]
for mode in ['fireOrigin','thermal']:
 pool=c.safe_records([r for r in records if r['mode']==mode],testhash);byhash={};hashgroups=collections.defaultdict(list)
 for r in pool:hashgroups[r['sha256']].append(r)
 for h,rr in hashgroups.items():
  if len({r['label_sha256'] for r in rr})>1:
   excluded.extend({'image':r['image'],'reason':'identical image hash has conflicting annotation files'} for r in rr)
   continue
  byhash[h]=rr[0]
 dataset=OUT/'safe_datasets'/mode;dataset.mkdir(parents=True,exist_ok=True); stats=collections.Counter(); groups={}
 for h,r in sorted(byhash.items()):
  p=Path(r['image']);stem=p.stem; family=re.sub(r'_aug\d+','',stem.split('.rf.')[0]);
  if stem.startswith('2026_'):family='_'.join(stem.split('_')[:5])
  group=hashlib.sha256(family.encode()).hexdigest(); split='val' if int(group[:8],16)%5==0 else 'train'
  image=cv2.imread(str(p));assert image is not None;hh,ww=image.shape[:2]
  vals=[list(map(float,l.split())) for l in Path(r['label']).read_text().splitlines() if l.strip()]
  bounds_ok=all(v[1]-v[3]/2>=-1e-5 and v[1]+v[3]/2<=1+1e-5 and v[2]-v[4]/2>=-1e-5 and v[2]+v[4]/2<=1+1e-5 for v in vals)
  if not bounds_ok:excluded.append({'image':str(p),'reason':'normalized box crosses image boundary'});continue
  dest=dataset/'images'/split/(h+'.jpg');label=dataset/'labels'/split/(h+'.txt');dest.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(p,dest);shutil.copy2(r['label'],label);assert sha(dest)==h and h not in testhash and sha(label)==r['label_sha256']
  manifest.append({**r,'new_split':split,'group':family,'export_image':str(dest),'export_label':str(label),'width':ww,'height':hh});stats[split]+=1;stats['empty_'+split]+=not vals;groups.setdefault(split,set()).add(family)
 assert not groups['train']&groups['val'];summary[mode]=dict(stats)
 yaml='path: '+dataset.as_posix()+'\ntrain: images/train\nval: images/val\nnames:\n  0: fire\n';(dataset/'data.yaml').write_text(yaml,encoding='utf-8')
assert not {r['sha256'] for r in manifest}&testhash
for mode in ['fireOrigin','thermal']:
 tr={r['sha256'] for r in manifest if r['mode']==mode and r['new_split']=='train'};vl={r['sha256'] for r in manifest if r['mode']==mode and r['new_split']=='val'};assert not tr&vl
shutil.copy2(SRC/'yolo26n.pt',OUT/'generic_coco_yolo26n.pt');assert sha(SRC/'yolo26n.pt')==sha(OUT/'generic_coco_yolo26n.pt')
receipt={'summary':summary,'manifest':manifest,'excluded_bounds':excluded,'test_hash_count':len(testhash),'test_overlap_after_export':0,'train_val_exact_overlap':0,'init_model':str(OUT/'generic_coco_yolo26n.pt'),'init_sha256':sha(OUT/'generic_coco_yolo26n.pt'),'initialization':'Generic 80-class COCO checkpoint only, NOT provided fire weights that may inherit test leakage','grouping':'SHA duplicates deduped; .rf augmentation + _aug families grouped; sensor filenames minute-block proxy. Unknown numeric filename scene/video provenance unresolved, near-duplicate leakage not ruled out.'}
(OUT/'safe_export_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps({'summary':summary,'excluded_bounds':len(excluded),'overlap':0},indent=2))
