from pathlib import Path
import sys,json,csv,hashlib,time,datetime,collections
import cv2,numpy as np,torch
from ultralytics import YOLO
import grounded_core as gc
ROOT=Path('C:/lyk/fire/qualityDetection'); OUT=Path(__file__).resolve().parent; SRC=Path('C:/lyk/fire/TorchDetection'); TEST=Path('C:/lyk/paper_data/paper_data_/paper_data')
sys.path.insert(0,str(ROOT/'fusion/calibrated_spatial_ds'));import calibrated_core as ds
sys.path.insert(0,str(ROOT/'fusion/crossmodal_roi_ds'));import core as geo
start=time.time(); sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protected={str(p):sha(p) for p in SRC.rglob('*') if p.is_file() and p.suffix.lower() in ['.py','.yaml','.pt','.csv','.txt','.xml']}
(OUT/'source_protected_sha256.json').write_text(json.dumps(protected,indent=2),encoding='utf-8')
lines=[x.split() for x in (TEST/'paper_sample.txt').read_text().splitlines() if x.strip()]; assert len(lines)==1528
allocation=list(csv.DictReader((ROOT/'fusion/calibrated_spatial_ds/split_allocation.csv').open(encoding='utf-8-sig')))
cfg=json.loads((ROOT/'fusion/calibrated_spatial_ds/frozen_selected.json').read_text())['parameters']
A=np.array(json.loads((ROOT/'fusion/supplied_models_affine_iou/candidate_affine_calibration.json').read_text())['affine_matrix_2x3'])
variants=[('rgb5_native320','rgb','fireOrigin_yolo26n-5',320),('rgb5_control640','rgb','fireOrigin_yolo26n-5',640),('rgb6_native320','rgb','fireOrigin_yolo26n-6',320),('thermal5_native320','thermal','thermal_yolo26n-5',320),('thermal5_control640','thermal','thermal_yolo26n-5',640),('thermal3_native320','thermal','thermal_yolo26n-3',320)]
plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prior_cfg_sha256':sha(ROOT/'fusion/calibrated_spatial_ds/frozen_selected.json'),'frozen_parameters':cfg,'variants':variants,'inference_conf':.01,'baseline_conf':.5,'selection':'Primary is supplied RGB5+thermal5 native320, chosen from training imgsz BEFORE inference; 640 control. RGB6 chosen from stored validation mAP95, thermal3 stronger validation control. No selecting winner on full labels. All previously seen; diagnostic exploratory evaluation, not independent.','method':'No UV inference input. Real dual boxes, 4-corner full affine polygon IoU, dummy Hungarian one-to-one. Fire-only DS K=0, rmin1 so spatial is gate only. No ROI synthetic boxes. Prior config tR.2 tT.1 iou.01 strict mF>.3 unchanged.'}
(OUT/'frozen_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8'); planhash=sha(OUT/'frozen_plan.json')
paths={m:[TEST/('bgr' if m=='rgb' else 'thermal')/line[0 if m=='rgb' else 1] for line in lines] for m in ['rgb','thermal']}
hashes={m:[sha(p) for p in ps] for m,ps in paths.items()}; testhashes=set(h for hs in hashes.values() for h in hs);assert len(testhashes)==3056
records=[];auditstats={}
for mode in ['fireOrigin','thermal']:
 for split in ['train','val']:
  pp=SRC/'datasets'/mode/'bases'/f'{split}.txt';sourcepaths=[SRC/x.strip() for x in pp.read_text().splitlines() if x.strip()];stats=collections.Counter()
  for p in sourcepaths:
   assert p.is_file(),p
   h=sha(p); lab=Path(str(p).replace('images','labels')).with_suffix('.txt');assert lab.is_file(),lab
   values=[list(map(float,x.split())) for x in lab.read_text().splitlines() if x.strip()]
   valid=all(len(v)==5 and v[0]==0 and all(0<=s<=1 for s in v[1:]) and v[3]>0 and v[4]>0 for v in values)
   overlap=h in testhashes
   records.append({'mode':mode,'split':split,'image':str(p),'sha256':h,'label':str(lab),'label_sha256':sha(lab),'box_count':len(values),'empty_annotation':not values,'valid':valid,'test_overlap':overlap})
   stats['images']+=1;stats['test_overlap']+=overlap;stats['empty_annotation']+=not values;stats['invalid']+=not valid
  auditstats[mode+'_'+split]=dict(stats)
(OUT/'training_dataset_audit.json').write_text(json.dumps({'stats':auditstats,'records':records,'claim':'Current source data hash audit, not proof of past checkpoint training membership; native saved args point to these configs. Empty provided labels retained as explicit annotations, not fabricated.'},indent=2),encoding='utf-8');print('AUDIT',auditstats,flush=True)
assert torch.cuda.is_available();gpu=torch.cuda.get_device_name(0); preds={}; metadata={}
for name,mode,run,imgsz in variants:
 p=SRC/'runs/detect'/run/'weights/best.pt';model=YOLO(str(p));assert model.names=={0:'fire'} or model.names=={0:'Fire'},model.names
 metadata[name]={'path':str(p),'sha256':sha(p),'names':model.names,'architecture':model.model.yaml,'train_args':model.ckpt['train_args'],'train_metrics':model.ckpt.get('train_metrics'),'imgsz_inference':imgsz,'device':gpu}
 rows=[];t=time.time()
 for first in range(0,1528,32):
  ims=[cv2.imread(str(x)) for x in paths[mode][first:first+32]];assert all(im is not None for im in ims)
  results=model.predict(ims,imgsz=imgsz,conf=.01,iou=.7,device=0,batch=32,rect=False,verbose=False,save=False,half=False,max_det=300,classes=[0]);assert len(results)==len(ims)
  for offset,(r,im) in enumerate(zip(results,ims)):
   np.testing.assert_array_equal(r.orig_img,im);i=first+offset;boxes=[]
   for x1,y1,x2,y2,score,cls in r.boxes.data.cpu().tolist():
    assert int(cls)==0 and score>.01 and x2>x1 and y2>y1
    boxes.append({'bbox':[x1,y1,x2,y2],'score':score,'class_id':0})
   rows.append({'pair_index':i,'image_name':paths[mode][i].name,'image_sha256':hashes[mode][i],'orig_shape':r.orig_shape,'detections':boxes})
  if first%320==0: print(name,len(rows),flush=True)
 assert len(rows)==1528 and len({r['pair_index'] for r in rows})==1528
 preds[name]=rows;metadata[name]['inference_seconds']=time.time()-t
 (OUT/(name+'_predictions.json')).write_text(json.dumps(rows),encoding='utf-8')
(OUT/'model_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
y=[int(x[3]) for x in lines]; held=[i for i,r in enumerate(allocation) if r['split']=='heldout']; subsets={'full':list(range(1528)),'prior202_diagnostic':held}; decisions={}; allrows=[]
combinations=[('primary_native320','rgb5_native320','thermal5_native320'),('control640','rgb5_control640','thermal5_control640'),('validation_selected_RGB6','rgb6_native320','thermal5_native320'),('stronger_thermal_control','rgb5_native320','thermal3_native320')]
for name,rgb,thermal in combinations:
 scores=[]
 for i in range(1528):
  rb=preds[rgb][i]['detections'];tb=preds[thermal][i]['detections'];ious=[[geo.polygon_iou(geo.corners(r['bbox']),geo.transform(geo.corners(t['bbox']),A)) for t in tb] for r in rb]
  score=ds.frame_score(rb,tb,ious,cfg['tR'],cfg['tT'],cfg['iou_gate'],cfg['rmin']);scores.append(score)
  allrows.append({'variant':name,'pair_index':i,'rgb_name':lines[i][0],'thermal_name':lines[i][1],'UV_reference':y[i],'split':allocation[i]['split'],'rgb_candidates_conf01':len(rb),'rgb_candidates_conf02':sum(r['score']>=.2 for r in rb),'thermal_candidates_conf01':len(tb),'mass_fire':score,'primary_fire':int(score>cfg['decision_threshold']),'rgb_baseline_fire':int(any(r['score']>.5 for r in rb)),'thermal_baseline_fire':int(any(t['score']>.5 for t in tb)),'max_polygon_iou':max([v for row in ious for v in row],default=0.)})
 decisions[name]={}
 for subset,ids in subsets.items():
  methods={}
  for method,p in [('fusion',[int(s>cfg['decision_threshold']) for s in scores]),('rgb_baseline',[int(any(b['score']>.5 for b in r['detections'])) for r in preds[rgb]]),('thermal_baseline',[int(any(b['score']>.5 for b in r['detections'])) for r in preds[thermal]])]:methods[method]=gc.metrics([y[i] for i in ids],[p[i] for i in ids])
  methods['target_met']=gc.target(methods['fusion'],methods['thermal_baseline'],methods['rgb_baseline']);decisions[name][subset]=methods
with (OUT/'frame_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
metrics={'reference':'UV reference only; UV is NOT inference input; previous entire set seen, diagnostic not independent.','results':decisions,'primary_target_full_and_prior202':all(decisions['primary_native320'][s]['target_met'] for s in subsets),'frozen_plan_sha256':planhash,'training_performed':False}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
assert sha(OUT/'frozen_plan.json')==planhash;assert all(sha(p)==h for p,h in protected.items())
(OUT/'execution.json').write_text(json.dumps({'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.time()-start,'actual_device':gpu,'forward_images':1528*len(variants),'unique_input_images':3056,'source_files_unchanged':len(protected),'protected_unchanged':True,'audit':auditstats},indent=2),encoding='utf-8')
print(json.dumps(metrics,indent=2),flush=True)
