import json,time,hashlib,csv,collections,sys,datetime
from pathlib import Path
import cv2,numpy as np,torch,ultralytics
from ultralytics import YOLO
import core
ROOT=Path('C:/lyk/fire/qualityDetection'); OUT=ROOT/'fusion/crossmodal_roi_ds'; OLD=ROOT/'fusion/supplied_models_uv_reference'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def run(limit=None,tag='full'):
 start=time.perf_counter(); config=json.loads((OUT/'frozen_config.json').read_text()); affine=json.loads((ROOT/'fusion/supplied_models_affine_iou/candidate_affine_calibration.json').read_text()); a=np.array(affine['affine_matrix_2x3']); inv=core.inverse(a)
 source=Path('C:/lyk/paper_data/paper_data_/paper_data/paper_sample.txt'); lines=[s.split() for s in source.read_text().splitlines() if s.strip()]
 cached={m:json.loads((OLD/(m+'_predictions.json')).read_text())['images'] for m in ['rgb','thermal']}
 assert len(lines)==1528 and all(len(cached[m])==1528 for m in cached)
 weights={'rgb':Path('C:/Users/Administrator/AppData/Local/hermes/attachments/best.pt'),'thermal':Path('C:/Users/Administrator/AppData/Local/hermes/attachments/best-2.pt')}
 protected={str(p):sha(p) for p in [source,ROOT/'fusion/supplied_models_affine_iou/candidate_affine_calibration.json',OLD/'rgb_predictions.json',OLD/'thermal_predictions.json',*weights.values()]}
 save(tag+'_receipt.json',{'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'config_sha256':sha(OUT/'frozen_config.json'),'protected_hashes':protected})
 models={m:YOLO(str(p)) for m,p in weights.items()}; device=0 if torch.cuda.is_available() else 'cpu'; fallbacks=[]; calls=collections.Counter(); runtime=[]
 for m,model in models.items():
  assert all(str(n).lower()=='fire' for n in model.names.values())
  assert model.model.yaml['yaml_file'].endswith('yolo26n.yaml')
 save(tag+'_model_metadata.json',{m:{'path':str(weights[m]),'sha256':sha(weights[m]),'names':model.names,'yaml':model.model.yaml,'parameters':sum(p.numel() for p in model.model.parameters()),'end2end':model.model.end2end} for m,model in models.items()})
 def predict(m,images):
  nonlocal device
  calls[m]+=len(images)
  kwargs=dict(conf=config['candidate_conf'],imgsz=config['imgsz'],max_det=config['max_det'],iou=config['inference_iou'],verbose=False,save=False,half=False,rect=False)
  try: results=models[m].predict(source=images,device=device,**kwargs)
  except RuntimeError as e:
   if device=='cpu':raise
   fallbacks.append({'modality':m,'error':str(e)}); device='cpu'; results=models[m].predict(source=images,device=device,**kwargs)
  assert len(results)==len(images)
  outputs=[]
  for im,res in zip(images,results):
   assert tuple(res.orig_shape)==im.shape[:2]
   runtime.append({'modality':m,'device':str(next(models[m].model.parameters()).device),'speed_ms':res.speed})
   boxes=[]
   for b,c,cls in zip(res.boxes.xyxy.cpu().numpy(),res.boxes.conf.cpu().numpy(),res.boxes.cls.cpu().numpy()):
    assert int(cls)==0
    boxes.append({'bbox':b.astype(float).tolist(),'score':float(c),'class_id':0,'provenance':'full_frame'})
   outputs.append(boxes)
  return outputs
 n=len(lines) if limit is None else limit; rows=[]; audited=[]
 for begin in range(0,n,config['batch']):
  batch=[]
  for i in range(begin,min(begin+config['batch'],n)):
   names=lines[i]; imgs={}; paths={}
   for m,j in [('rgb',0),('thermal',1)]:
    record=cached[m][i]; assert record['pair_index']==i and record['image_name']==names[j]
    path=Path(record['image_path']); digest=sha(path); assert digest==record['image_sha256']; image=cv2.imread(str(path)); assert image is not None and image.shape[:2]==(record['height'],record['width'])
    imgs[m]=image; paths[m]=str(path); audited.append({'pair_index':i,'modality':m,'path':str(path),'sha256':digest,'shape':list(image.shape)})
   batch.append((i,names,imgs,paths))
  initial={m:predict(m,[b[2][m] for b in batch]) for m in models}
  for k,(i,names,imgs,paths) in enumerate(batch):
   boxes={m:initial[m][k] for m in models}; before=core.match(boxes['rgb'],boxes['thermal'],a,config['polygon_iou_gate'])
   # Fair no-ROI ablation applies the same deterministic same-model duplicate suppression as the final branch.
   preboxes={}
   for m in models:
    keep=[]
    for b in sorted(boxes[m],key=lambda b:-b['score']):
     if not any(core.polygon_iou(core.corners(b['bbox']),core.corners(q['bbox']))>.7 for q in keep):keep.append(b)
    preboxes[m]=keep
   prepairs=core.match(preboxes['rgb'],preboxes['thermal'],a,config['polygon_iou_gate'])
   previsual,_,prebest,_=core.decide(preboxes['rgb'],preboxes['thermal'],prepairs,int(names[3]))
   matched={'rgb':{p[0] for p in before},'thermal':{p[1] for p in before}}; attempts=[]; additions={'rgb':[],'thermal':[]}
   for m,other,matrix in [('rgb','thermal',inv),('thermal','rgb',a)]:
    triggers=[(idx,b) for idx,b in enumerate(boxes[m]) if idx not in matched[m] and b['score']>=config['roi_trigger_conf']]
    triggers.sort(key=lambda x:(-x[1]['score'],x[0])); triggers=triggers[:config['roi_top_per_modality']]
    valid=[]; crops=[]
    for idx,b in triggers:
     bounds=core.crop_bounds(core.transform(core.corners(b['bbox']),matrix),imgs[other].shape,config['roi_padding'],config['registration_margin_px'])
     x1,y1,x2,y2=bounds; attempt={'trigger_modality':m,'trigger_index':idx,'trigger_bbox':b['bbox'],'trigger_score':b['score'],'review_modality':other,'crop_xyxy_original':bounds,'detections':[],'status':'invalid_crop'}; attempts.append(attempt)
     if x2>x1 and y2>y1:
      attempt['status']='inferred'; valid.append(attempt); crops.append(imgs[other][y1:y2,x1:x2].copy())
    if crops:
     detections=predict(other,crops)
     for attempt,det in zip(valid,detections):
      for d in det:
       mapped={**d,'roi_bbox':d['bbox'],'bbox':core.offset_box(d['bbox'],attempt['crop_xyxy_original']),'provenance':'roi_actual_detection','roi_attempt_index':attempts.index(attempt)}
       additions[other].append(mapped); attempt['detections'].append(mapped)
   for m in boxes:
    boxes[m]+=additions[m]
    # Same-model duplicate reviews are not cumulatively fused: deterministic native-coordinate duplicate suppression.
    keep=[]
    for b in sorted(boxes[m],key=lambda b:-b['score']):
     if not any(core.polygon_iou(core.corners(b['bbox']),core.corners(q['bbox']))>.7 for q in keep):keep.append(b)
    boxes[m]=keep
   pairs=core.match(boxes['rgb'],boxes['thermal'],a,config['polygon_iou_gate']); visual,uvpred,best,evidence=core.decide(boxes['rgb'],boxes['thermal'],pairs,int(names[3])); mf=best['mass'][0] if best else 0.
   if visual:reason='fire'
   elif not boxes['rgb'] and not boxes['thermal']:reason='both_no_candidates'
   elif not boxes['rgb'] or not boxes['thermal']:
    reason='roi_no_response' if any(t['status']=='inferred' for t in attempts) else 'candidate_weak_or_missing'
   elif not pairs:reason='iou_failed'
   else:reason='ds_low_score'
   rows.append({'pair_index':i,'source_line':i+1,'rgb_name':names[0],'thermal_name':names[1],'rgb_path':paths['rgb'],'thermal_path':paths['thermal'],'temperature_raw':names[2],'uv_reference':int(names[3]),'rgb_detections':boxes['rgb'],'thermal_detections':boxes['thermal'],'roi_attempts':attempts,'matched_pairs':pairs,'pair_evidence':evidence,'best':best,'mf':mf,'uv_gated_mf':core.uv_score(mf,int(names[3])),'visual_fire':visual,'uv_fire':uvpred,'pre_roi_visual_fire':previsual,'reason':reason})
  print(f'{tag} {len(rows)}/{n} elapsed={time.perf_counter()-start:.1f}s ROI={sum(len(r["roi_attempts"]) for r in rows)}',flush=True)
 assert len(rows)==n and len({r['pair_index'] for r in rows})==n and len(audited)==2*n
 assert all(sha(p)==h for p,h in protected.items())
 save(tag+'_results.json',rows); save(tag+'_image_audit.json',audited)
 with (OUT/(tag+'_results.csv')).open('w',encoding='utf-8-sig',newline='') as f:
  fields=['pair_index','source_line','rgb_name','thermal_name','temperature_raw','uv_reference','mf','uv_gated_mf','visual_fire','uv_fire','pre_roi_visual_fire','reason']; w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 truth=[r['uv_reference'] for r in rows]; methods={key:core.metrics(truth,[r[field] for r in rows]) for key,field in [('primary_vision_only','visual_fire'),('optional_uv_gate','uv_fire'),('before_roi_vision','pre_roi_visual_fire')]}
 if n==1528:
  for m in ['rgb','thermal']:
   methods[m+'_historical_baseline']=core.metrics(truth,[bool(x['detections']) for x in cached[m]])
  for m in models: methods[m+'_new_conf01']=core.metrics(truth,[any(b['score']>.1 and b['provenance']=='full_frame' for b in r[m+'_detections']) for r in rows])
 save(tag+'_metrics.json',{'N':n,'reference':'UV fourth column; optional UV metrics circular, vision-only UV-reference agreement not independently reviewed truth','methods':methods,'miss_diagnostics':dict(collections.Counter(r['reason'] for r in rows if r['uv_reference'] and not r['visual_fire']))})
 save(tag+'_execution.json',{'N':n,'unique_pairs':n,'verified_source_images':len(audited),'full_frame_inferences':2*n,'model_input_counts':dict(calls),'roi_attempts':sum(len(r['roi_attempts']) for r in rows),'roi_actual_forward_images':sum(t['status']=='inferred' for r in rows for t in r['roi_attempts']),'roi_real_detections':sum(len(t['detections']) for r in rows for t in r['roi_attempts']),'actual_devices':sorted({r['device'] for r in runtime}),'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'torch':torch.__version__,'ultralytics':ultralytics.__version__,'fallbacks':fallbacks,'elapsed_seconds':time.perf_counter()-start,'protected_unchanged':True,'config_sha256':sha(OUT/'frozen_config.json'),'timings':runtime})
 return rows
if __name__=='__main__':run()
