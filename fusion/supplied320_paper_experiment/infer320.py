from pathlib import Path
import json,hashlib,time,datetime,inspect,csv,collections
import torch,cv2,numpy as np,ultralytics
from ultralytics import YOLO
from ultralytics.utils import nms
from fuse320 import *
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
TEST=Path('C:/lyk/paper_data/paper_data_/paper_data')
ATT=Path('C:/Users/Administrator/AppData/Local/hermes/attachments')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,obj):
 p=OUT/name
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')
def forward(model,paths,ids):
 ims=[cv2.imread(str(p)) for p in paths];assert all(im is not None for im in ims)
 seen=[]
 hook=model.model.register_forward_pre_hook(lambda m,args:seen.append(list(args[0].shape)))
 results=list(model.predict(ims,imgsz=320,rect=False,conf=.05,iou=.7,max_det=300,classes=[0],device=0,batch=32,half=False,augment=False,verbose=False,save=False))
 hook.remove()
 assert len(results)==len(ims);assert seen and all(s[-2:]==[320,320] for s in seen),seen
 rows=[]
 for r,im,p,i in zip(results,ims,paths,ids):
  np.testing.assert_array_equal(r.orig_img,im);assert tuple(r.orig_shape)==im.shape[:2]
  r.path=str(p)
  ds=[]
  for x,y,X,Y,c,k in r.boxes.data.cpu().tolist():
   assert int(k)==0 and c>.05 and X>x and Y>y
   ds.append(dict(bbox=[x,y,X,Y],score=c,class_id=0))
  rows.append(dict(pair_index=i,source_line=i+1,image_name=p.name,image_path=str(p),image_sha256=sha(p),orig_shape=list(r.orig_shape),detections=ds,speed_ms=r.speed,model_input_shapes=seen,device='cuda:0'))
 return rows

def run():
 start=time.perf_counter();started=now()
 frozen=ROOT/'fusion/calibrated_spatial_ds/frozen_selected.json';oldplan=ROOT/'newdirfusion/dataset_grounded_ds/frozen_plan.json';oldcode=ROOT/'newdirfusion/dataset_grounded_ds/run_grounded.py';oldmeta=ROOT/'newdirfusion/dataset_grounded_ds/model_metadata.json'
 cfg=json.loads(frozen.read_text())['parameters'];assert cfg==dict(tR=.2,tT=.1,iou_gate=.01,rmin=1.,decision_threshold=.3)
 geo_path=ROOT/'fusion/supplied_models_affine_iou/candidate_affine_calibration.json';geo=json.loads(geo_path.read_text());A=np.array(geo['affine_matrix_2x3'])
 sources=[frozen,oldplan,oldcode,oldmeta,geo_path,ROOT/'fusion/calibrated_spatial_ds/calibrated_core.py']
 oldhash={str(p):sha(p) for p in sources}
 lines=[s.split() for s in (TEST/'paper_sample.txt').read_text(encoding='utf-8-sig').splitlines() if s.strip()];assert len(lines)==1528
 y=[int(s[3]) for s in lines];assert collections.Counter(y)=={1:1398,0:130}
 paths={m:[TEST/d/s[c] for s in lines] for m,d,c in [('rgb','bgr',0),('thermal','thermal',1)]}
 assert all(len(set(ps))==1528 and all(p.is_file() for p in ps) for ps in paths.values())
 expected={'rgb':'1068cfddcca133ac2f6c66522d7a43f4988daa91023c1c8898bb6fbf4a894e32','thermal':'c068422549d5de50d532431892a940a44ce2300b1cb0fae2e12887749adc12e6'}
 models={};metadata={};native=json.loads(oldmeta.read_text())
 assert torch.cuda.is_available()
 for m,name,key in [('rgb','best.pt','rgb5_native320'),('thermal','best-2.pt','thermal5_native320')]:
  p=ATT/name;h=sha(p);assert h==expected[m]==native[key]['sha256']
  model=YOLO(str(p));assert model.names[0].lower()=='fire' and len(model.names)==1 and model.model.end2end
  models[m]=model;metadata[m]=dict(source=str(p),sha256=h,bytes=p.stat().st_size,names=model.names,architecture=model.model.yaml,train_args=model.ckpt.get('train_args'),checkpoint_version=model.ckpt.get('version'),checkpoint_date=model.ckpt.get('date'),checkpoint_epoch=model.ckpt.get('epoch'),end2end=model.model.end2end,matched_prior_checkpoint=key,prior_checkpoint_sha256=native[key]['sha256'])
 protocol=dict(created_utc=now(),started_utc=started,parameters=cfg,input_size=320,rect=False,full_image_only=True,roi=False,initial_filter='score > 0.05 (Ultralytics actual end2end postprocess)',fusion_filters='RGB score >= .2; thermal score >= .1 BEFORE matching',baseline_filter='score > .5',iou_argument=.7,max_det=300,half=False,device='cuda:0',gpu=torch.cuda.get_device_name(0),torch=torch.__version__,ultralytics=ultralytics.__version__,batch=32,end2end=True,nms='end2end Nx6 confidence filter + max_det truncation; conventional NMS bypassed, iou .7 supplied but inactive',candidate_policy='dummy-enabled Hungarian maximum total eligible IoU, then maximum mF among matched pairs; selection independent of UV',UV_main_input=False,UV_optional='mF * (.5 + .5*UV), strict > .3; circular UV-reference ablation only',prior320_actual_conf=.01,prior320_reproduction_note='Prior actual run conf .01; new run floor .05 requested. Both below fixed operating .1/.2, no floor claim of bit-exact raw detections. Exact same supplied SHA256 models and frozen frame rule.',old_source_sha256=oldhash,geometry=geo,model_sources=metadata)
 save('frozen_protocol.json',protocol);save('freeze_receipt.json',dict(created_utc=now(),protocol_sha256=sha(OUT/'frozen_protocol.json'),before_smoke_and_full_forward=True))
 save('model_metadata.json',metadata)
 text=inspect.getsource(nms.non_max_suppression);assert 'pred[:, 4] > conf_thres' in text
 (OUT/'ultralytics_postprocess_source.txt').write_text(text,encoding='utf-8')
 smoke={m:forward(model,paths[m][:3],list(range(3))) for m,model in models.items()}
 save('smoke_3pairs.json',dict(created_utc=now(),results=smoke,real_forward_images=6));print('SMOKE 3 pairs PASS cuda:0 fixed 320x320',flush=True)
 pred={};timing={}
 for m,model in models.items():
  t=time.perf_counter();rows=[];batches=[]
  for first in range(0,1528,32):
   bt=time.perf_counter();rs=forward(model,paths[m][first:first+32],list(range(first,min(first+32,1528))))
   rows.extend(rs);batches.append(dict(first=first,count=len(rs),seconds=time.perf_counter()-bt,utc=now()))
   if first%320==0:print(m,len(rows),flush=True)
  aligned(rows,[p.name for p in paths[m]]);pred[m]=rows
  timing[m]=dict(seconds=time.perf_counter()-t,batches=batches,processed=1528,unique_ids=1528,missing=0)
  save(m+'_predictions.json',rows)
 hashes={r['image_sha256'] for rows in pred.values() for r in rows};assert len(hashes)==3056
 # Recheck present source split content against actual freshly read test images; not historical training proof.
 SRC=Path('C:/lyk/fire/TorchDetection');audit=[];stats={}
 for mode in ['fireOrigin','thermal']:
  for split in ['train','val']:
   sourcefile=SRC/'datasets'/mode/'bases'/f'{split}.txt';records=[]
   for s in sourcefile.read_text().splitlines():
    if not s.strip():continue
    p=SRC/s.strip();h=sha(p);records.append(dict(path=str(p),sha256=h,test_overlap=h in hashes))
   stats[mode+'_'+split]=dict(images=len(records),test_overlap=sum(r['test_overlap'] for r in records));audit.extend(dict(mode=mode,split=split,**r) for r in records)
 save('source_overlap_audit.json',dict(stats=stats,records=audit,claim='Current source train/val exact hash overlap, not proof of historical checkpoint membership.'))
 rows=[];evidence=[]
 for i in range(1528):
  rb=pred['rgb'][i]['detections'];tb=pred['thermal'][i]['detections']
  ious=np.array([[polygon_iou(corners(r['bbox']),transform(corners(t['bbox']),A)) for t in tb] for r in rb])
  ms=match(rb,tb,ious);best=max(ms,key=lambda x:x['mF'],default=None);mf=best['mF'] if best else 0.
  row=dict(pair_index=i,source_line=i+1,rgb_name=lines[i][0],thermal_name=lines[i][1],UV_reference=y[i],rgb_box_count=len(rb),thermal_box_count=len(tb),rgb_candidates=sum(b['score']>=.2 for b in rb),thermal_candidates=sum(b['score']>=.1 for b in tb),matched_count=len(ms),selected_rgb_index=best['r'] if best else -1,selected_thermal_index=best['t'] if best else -1,selected_iou=best['iou'] if best else 0.,mass_fire=mf,mass_nofire=0.,mass_uncertain=1-mf,conflict=0.,main_fire=int(mf>.3),rgb_baseline=int(any(b['score']>.5 for b in rb)),thermal_baseline=int(any(b['score']>.5 for b in tb)),rgb_operating=int(any(b['score']>=.2 for b in rb)),thermal_operating=int(any(b['score']>=.1 for b in tb)),optional_uv_mass=mf*(.5+.5*y[i]),optional_uv_fire=int(mf*(.5+.5*y[i])>.3))
  rows.append(row);evidence.append(dict(pair_index=i,matches=ms,selected=best))
 with (OUT/'frame_predictions.csv').open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save('matched_evidence.json',evidence)
 methods=['rgb_baseline','thermal_baseline','rgb_operating','thermal_operating','main_fire','optional_uv_fire']
 result={m:metrics(y,[r[m] for r in rows]) for m in methods}
 from scipy.stats import beta
 for m,v in result.items():
  v['FPR_exact95_upper']=float(beta.ppf(.975,v['FP']+1,v['TN'])) if v['TN'] else 1.
  v['FPR_exact95_lower']=float(beta.ppf(.025,v['FP'],v['TN']+1)) if v['FP'] else 0.
 save('metrics.json',dict(reference='UV fourth column; not independent scene GT; NO UV main input',results=result,strict_target_met=result['main_fire']['FPR']<=result['thermal_baseline']['FPR'] and result['main_fire']['FNR']<result['thermal_baseline']['FNR'] and result['main_fire']['FPR']<result['rgb_baseline']['FPR']))
 eligible=[r for r in rows if r['matched_count']];positive=[r for r in rows if r['UV_reference']==1]
 save('coverage_diagnostics.json',dict(rgb_raw_boxes=sum(r['rgb_box_count'] for r in rows),thermal_raw_boxes=sum(r['thermal_box_count'] for r in rows),rgb_operating_boxes=sum(r['rgb_candidates'] for r in rows),thermal_operating_boxes=sum(r['thermal_candidates'] for r in rows),no_rgb_candidates=sum(not r['rgb_candidates'] for r in rows),no_thermal_candidates=sum(not r['thermal_candidates'] for r in rows),no_eligible_match=sum(not r['matched_count'] for r in rows),positive_no_eligible_match=sum(not r['matched_count'] for r in positive),matched_pairs=sum(r['matched_count'] for r in rows),selected_iou_quantiles=dict(zip(['min','q25','median','q75','max'],np.quantile([r['selected_iou'] for r in eligible],[0,.25,.5,.75,1]).tolist()))))
 assert all(sha(p)==h for p,h in oldhash.items())
 save('execution.json',dict(started_utc=started,completed_utc=now(),elapsed_seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name(0),actual_device='cuda:0',full_forward_images=3056,smoke_forward_images=6,unique_test_image_sha256=3056,timing=timing,no_training=True,no_roi=True,old_sources_unchanged=True))
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':run()
