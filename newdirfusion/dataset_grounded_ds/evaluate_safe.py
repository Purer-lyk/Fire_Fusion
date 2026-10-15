from pathlib import Path
import sys,json,csv,hashlib,time,datetime,cv2,torch,numpy as np
from ultralytics import YOLO
import grounded_core as gc
ROOT=Path('C:/lyk/fire/qualityDetection');OUT=Path(__file__).resolve().parent;TEST=Path('C:/lyk/paper_data/paper_data_/paper_data')
sys.path.insert(0,str(ROOT/'fusion/calibrated_spatial_ds'));import calibrated_core as ds
sys.path.insert(0,str(ROOT/'fusion/crossmodal_roi_ds'));import core as geo
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.time()
assert json.loads((OUT/'safe_training_complete.json').read_text())['success']
export=json.loads((OUT/'safe_export_receipt.json').read_text()); assert export['test_overlap_after_export']==0
sourceprotected=json.loads((OUT/'source_protected_sha256.json').read_text());assert all(sha(p)==h for p,h in sourceprotected.items())
lines=[l.split() for l in (TEST/'paper_sample.txt').read_text().splitlines() if l.strip()];assert len(lines)==1528
allocation=list(csv.DictReader((ROOT/'fusion/calibrated_spatial_ds/split_allocation.csv').open(encoding='utf-8-sig'))); held=[int(r['pair_index']) for r in allocation if r['split']=='heldout'];assert len(held)==202
cfg=json.loads((ROOT/'fusion/calibrated_spatial_ds/frozen_selected.json').read_text())['parameters'];A=np.array(json.loads((ROOT/'fusion/supplied_models_affine_iou/candidate_affine_calibration.json').read_text())['affine_matrix_2x3'])
plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prior_cfg_sha256':sha(ROOT/'fusion/calibrated_spatial_ds/frozen_selected.json'),'parameters':cfg,'conf':.01,'imgsz':320,'baseline_conf':.5,'batches':32,'model_selection':'Best validation fitness; no full-test selection','variants':['fresh_safe_RGB_and_thermal_primary'],'historical_baseline':'Supplied thermal640 TP732 FP0 FN666 TN130 reference kept unchanged for stronger FPR comparison. Same fresh native320 thermal baseline primary.','labels':'UV reference only, no UV inference input; prior202 diagnostic already seen, FNR strict less impossible if thermal baseline FNR0.','model_hashes':{m:sha(OUT/'safe_training'/m/'weights/best.pt') for m in ['fireOrigin','thermal']}}
(OUT/'safe_evaluation_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8');planhash=sha(OUT/'safe_evaluation_plan.json')
preds={}; metadata={};audit=[]
for mode,run,folder,col in [('rgb','fireOrigin','bgr',0),('thermal','thermal','thermal',1)]:
 p=OUT/'safe_training'/run/'weights/best.pt';model=YOLO(str(p));assert model.names=={0:'fire'};rows=[];t=time.time()
 metadata[mode]={'checkpoint':str(p),'checkpoint_sha256':sha(p),'train_args':model.ckpt['train_args'],'best_validation':model.ckpt['train_metrics'],'architecture':model.model.yaml,'class_names':model.names,'end2end':model.model.end2end,'parameters':sum(x.numel() for x in model.model.parameters())}
 for first in range(0,1528,32):
  paths=[TEST/folder/l[col] for l in lines[first:first+32]];ims=[cv2.imread(str(p)) for p in paths];assert all(x is not None for x in ims)
  rs=model.predict(ims,imgsz=320,conf=.01,iou=.7,device=0,batch=32,rect=False,verbose=False,save=False,half=False,max_det=300,classes=[0]);assert len(rs)==len(ims)
  for offset,(r,im,p) in enumerate(zip(rs,ims,paths)):
   np.testing.assert_array_equal(r.orig_img,im);i=first+offset;h=sha(p);audit.append({'mode':mode,'pair_index':i,'path':str(p),'sha256':h});boxes=[]
   for x1,y1,x2,y2,score,cls in r.boxes.data.cpu().tolist():
    assert int(cls)==0 and score>.01 and x2>x1 and y2>y1
    boxes.append({'bbox':[x1,y1,x2,y2],'score':score,'class_id':0})
   rows.append({'pair_index':i,'image_name':p.name,'sha256':h,'orig_shape':r.orig_shape,'detections':boxes})
  if first%320==0:print('SAFE_FORWARD',mode,len(rows),flush=True)
 assert len(rows)==1528 and len({r['pair_index'] for r in rows})==1528
 preds[mode]=rows; metadata[mode]['forward_seconds']=time.time()-t
 (OUT/('safe_'+mode+'_predictions.json')).write_text(json.dumps(rows),encoding='utf-8')
assert len({r['sha256'] for r in audit})==3056
assert not {r['sha256'] for r in export['manifest']}&{r['sha256'] for r in audit}
(OUT/'safe_model_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');(OUT/'safe_test_image_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
rows=[];selected=[];yy=[int(l[3]) for l in lines]
for i,line in enumerate(lines):
 rgb=preds['rgb'][i]['detections'];thermal=preds['thermal'][i]['detections'];rr=[b for b in rgb if b['score']>=cfg['tR']];tt=[b for b in thermal if b['score']>=cfg['tT']]
 pairs=geo.match(rr,tt,A,cfg['iou_gate']);evidence=[{'rgb_box':rr[r],'thermal_box':tt[t],'polygon_iou':s,'mass_fire':ds.fire_mass(rr[r]['score'],tt[t]['score'],s,cfg['rmin'])} for r,t,s in pairs];evidence.sort(key=lambda e:-e['mass_fire']);chosen=evidence[0] if evidence else None;mf=chosen['mass_fire'] if chosen else 0.
 selected.append({'pair_index':i,'one_to_one_pair_count':len(pairs),'chosen':chosen,'matched_evidence':evidence})
 rows.append({'pair_index':i,'rgb_name':line[0],'thermal_name':line[1],'UV_reference':yy[i],'split':allocation[i]['split'],'rgb_boxes_conf01':len(rgb),'thermal_boxes_conf01':len(thermal),'rgb_boxes_conf02':len(rr),'thermal_boxes_conf1':len(tt),'one_to_one_matches':len(pairs),'selected_iou':chosen['polygon_iou'] if chosen else 0.,'mass_fire':mf,'fusion_fire':int(mf>cfg['decision_threshold']),'rgb_baseline_fire':int(any(b['score']>.5 for b in rgb)),'thermal_baseline_fire':int(any(b['score']>.5 for b in thermal))})
with (OUT/'safe_frame_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'safe_matched_evidence.json').write_text(json.dumps(selected),encoding='utf-8')
results={}
for subset,ids in [('full',list(range(1528))),('prior202_diagnostic',held)]:
 methods={method:gc.metrics([yy[i] for i in ids],[rows[i][field] for i in ids]) for method,field in [('fusion','fusion_fire'),('rgb_baseline','rgb_baseline_fire'),('thermal_baseline','thermal_baseline_fire')]}
 methods['target_met_same_models']=gc.target(methods['fusion'],methods['thermal_baseline'],methods['rgb_baseline']);results[subset]=methods
historical={'N':1528,'TP':732,'FP':0,'FN':666,'TN':130,'FPR':0.,'FNR':666/1398}
results['full']['target_met_vs_historical_thermal']=gc.target(results['full']['fusion'],historical,results['full']['rgb_baseline'])
metrics={'reference':'UV-reference agreement; UV NOT fusion input. Entire set previously seen, prior202 diagnostic not fresh independent. Exact test hashes fully excluded from both networks fresh training AND validation. Near-duplicate/source-video provenance unresolved.','results':results,'source_model_diagnostic':json.loads((OUT/'metrics.json').read_text())['results'],'training_performed':True,'generic_COCO_init_no_inherited_fire_test_leak':True,'config_unchanged':cfg,'primary_full_target_met':results['full']['target_met_same_models'],'full_and_prior202_strict_target_met':all(v['target_met_same_models'] for v in results.values()),'prior202_limitation':'A thermal FNR0 cannot be strictly improved below0; equality is not the requested strict less.'}
(OUT/'safe_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
# Display actual source images and real detector boxes, not projected detections.
from PIL import Image,ImageDraw
montage=Image.new('RGB',(1000,840),'white');d=ImageDraw.Draw(montage)
for k,i in enumerate([1348,1438,1527]):
 for j,mode,folder,col in [(0,'rgb','bgr',0),(1,'thermal','thermal',1)]:
  im=cv2.imread(str(TEST/folder/lines[i][col]));h,w=im.shape[:2]
  for b in preds[mode][i]['detections']:
   if b['score']<(.2 if mode=='rgb' else .1):continue
   x1,y1,x2,y2=map(int,b['bbox']);cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),1);cv2.putText(im,f"{b['score']:.3f}",(x1,max(12,y1-2)),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,0),1)
  montage.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((480,240)),(j*500,k*280+30));d.text((j*500,k*280),f"pair{i} {mode} mass={rows[i]['mass_fire']:.3f} fire={rows[i]['fusion_fire']}",fill='black')
montage.save(OUT/'safe_failed_pair_detections.png')
assert sha(OUT/'safe_evaluation_plan.json')==planhash and all(sha(p)==h for p,h in sourceprotected.items())
(OUT/'safe_execution.json').write_text(json.dumps({'success':True,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'forward_images':3056,'unique_input_images':3056,'elapsed_seconds':time.time()-start,'device':torch.cuda.get_device_name(0),'protected_source_files_unchanged':len(sourceprotected),'test_train_exact_overlap':0,'freeze_sha256':planhash},indent=2),encoding='utf-8')
print(json.dumps({'results':results,'target':metrics['primary_full_target_met']},indent=2),flush=True)
