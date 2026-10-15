from pathlib import Path
import sys,json,csv,hashlib,datetime,time,collections,zipfile
import cv2
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent.parent;PREV=OUT.parent/'dual_baseline_target_ds'
sys.path.insert(0,str(PREV))
import run_dual as prior
import adaptive_core as c
load=prior.load;sha=prior.sha;now=prior.now;P=prior.P;A=prior.A

def save(name,obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvsave(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def prepare():
 if (OUT/'protocol.json').exists():
  assert sha(OUT/'protocol.json')==load(OUT/'protocol_receipt.json')['sha256'];return
 rows=prior.old.parse();sources={m:load(PREV/(m+'_full_predictions.json')) for m in ['rgb','thermal']}
 for m,v in P['models'].items():assert sha(v['source'])==v['sha256']
 protected={str(p):sha(p) for p in PREV.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
 save('protected_previous.json',protected)
 proto=dict(utc=now(),source_sha256=sha(prior.old.SOURCE),models={m:{'source':v['source'],'sha256':v['sha256']} for m,v in P['models'].items()},N=len(rows),groups=P['all_groups'],original_shapes={m:dict(collections.Counter(str(r['orig_shape']) for r in s)) for m,s in sources.items()},detector=dict(views=['full320','full640','tile00_320','tile01_320','tile10_320','tile11_320'],tile_size_fraction='ceil(4/7 * original width/height)',tile_overlap='.25 per crop (rounding may vary)',floor=.005,half=False,rect=False,device='cuda:0',cross_view_nms=.8,trigger='every image; no labels, boxes, timestamp or source ID'),registration=dict(variants=['gradient_sift','intensity_sift','gradient_orb'],sample_indices=sorted(set(np.linspace(0,len(rows)-1,24,dtype=int).tolist()+[60,80,817])),mask='fixed 24px top/12px remaining borders plus existing detector boxes expanded8px; also exclude detected red/yellow burned-in annotations',matching='mutual Lowe .75; full affine RANSAC3px; spatially ordered third heldout',promotion='Require holdout p90<=5px, median<=3px, >=8 train inliers, >=4 holdout, >=.1 hull both views, >=4/9 RGB cells AND independent neutral landmark verification. Otherwise retain nominal candidate-unvalidated geometry.'),uncertainty=dict(seed=741,samples=32,shrinkage=.5,loo_max_px=15,singular_values=[.3,1.6],semantics='mean actual polygon IoU across fixed seven-point bootstrap distribution, not nominal IoU, never per-frame maximum; q10 recorded; internal model-uncertainty diagnostic only, cannot certify scene systematic offset'),calibration=dict(features=['logit_detector_confidence','log_bbox_area_fraction','log_aspect_ratio'],learner='LogisticRegression C=1 max_iter1000 seed741; each frame total weight1',label='weak frame UV reference only on fold training rows; runtime no UV, IDs or filenames',bpa='m=(.9*p,.9*(1-p),.1); thermal spatial r=.6+.4*min(s/.3,1); symmetric reliability discount thermal',combination='classic DS, fixed Yager switch K>=.95',thresholds=[.2,.3,.4,.5,.6,.7,.8,.9],trials=32,branches=['nominal/raw','nominal/calibrated','expected/raw','expected/calibrated'],selection='FP ascending then TP descending first deterministic threshold, training only'),primary_gate='two actual detector boxes, actual nominal mapped polygon IoU>=.001; feature transform only if independently validated',target='FP=0 FN<=25; also FP<=min(updated baselines) FN<min(updated baselines)',independent_test=False,exposure='all1528 repeatedly viewed, temporal-proxy six-group OOF internal exploration only; known content overlap risk')
 save('protocol.json',proto);save('protocol_receipt.json',dict(utc=now(),sha256=sha(OUT/'protocol.json'),before_new_forward=True))
 print('PREPARED',proto['original_shapes'],flush=True)
def dedup(boxes):
 out=[]
 for b in sorted(boxes,key=lambda b:-b['score']):
  if all(c.piou(c.corners(b['bbox']),c.corners(v['bbox']))<.8 for v in out):out.append(b)
 return out
def fresh(limit=None,suffix='all'):
 import torch
 from ultralytics import YOLO
 prepare();assert torch.cuda.is_available();start=time.perf_counter();started=now();sources={m:load(PREV/(m+'_full_predictions.json'))[:limit] for m in ['rgb','thermal']};shapes=[];receipts=[];actual=0
 for m in ['rgb','thermal']:
  assert sha(P['models'][m]['source'])==P['models'][m]['sha256'];model=YOLO(P['models'][m]['source']);rows=sources[m];out=[dict(row,detections=[]) for row in rows];full320=[dict(row,detections=[]) for row in rows]
  def forward(jobs,size):
   nonlocal actual
   seen=[];hook=model.model.register_forward_pre_hook(lambda mm,args:seen.append(list(args[0].shape)))
   ims=[j['pixels'] for j in jobs]
   z=list(model.predict(ims,imgsz=size,rect=False,conf=.005,device=0,half=False,batch=32,classes=[0],max_det=300,iou=.7,augment=False,verbose=False,save=False));hook.remove()
   assert len(z)==len(jobs) and all(s[-2:]==[size,size] for s in seen);shapes.extend(seen);actual+=len(z)
   for result,job in zip(z,jobs):
    np.testing.assert_array_equal(result.orig_img,job['pixels']);i=job['index'];bs=[]
    for x,y,X,Y,score,cls in result.boxes.data.cpu().tolist():
     assert int(cls)==0 and score>.005 and X>x and Y>y
     b=dict(bbox=c.map_box([x,y,X,Y],job['crop']),score=score,class_id=0,view=job['view'],crop=job['crop'],input_size=size,actual_detector=True);bs.append(b)
    out[i]['detections'].extend(bs)
    if job['view']=='full320':full320[i]['detections']=bs
    receipts.append(dict(modality=m,pair_index=rows[i]['pair_index'],view=job['view'],crop=job['crop'],input_size=size,detections=len(bs)))
  for size in [320,640]:
   jobs=[]
   for i,row in enumerate(rows):
    im=cv2.imread(row['image_path']);assert sha(row['image_path'])==row['image_sha256'] and list(im.shape[:2])==row['orig_shape']
    h,w=im.shape[:2];views=[('full'+str(size),[0,0,w,h])]
    if size==320:views += [('tile'+str(j)+'_320',crop) for j,crop in enumerate(c.tiles(im.shape))]
    for view,crop in views:
     x,y,X,Y=crop;jobs.append(dict(index=i,view=view,crop=crop,pixels=im[y:Y,x:X].copy()))
     if len(jobs)==32:forward(jobs,size);jobs=[]
   if jobs:forward(jobs,size)
  for r in out:r['detections']=dedup(r['detections'])
  save(m+'_'+suffix+'_predictions.json',out);save(m+'_'+suffix+'_full320.json',full320);print('FORWARD',suffix,m,len(rows),actual,flush=True)
 receipt=dict(utc_start=started,utc_end=now(),seconds=time.perf_counter()-start,actual_detector_views=actual,original_images=sum(len(s) for s in sources.values()),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,input_shapes=shapes,all_image_hashes_verified=True,weights={m:sha(P['models'][m]['source']) for m in sources},source_sha256=sha(prior.old.SOURCE),floor=.005,synthetic_data=False,views_sha256=None)
 save('views_'+suffix+'.json',receipts);receipt['views_sha256']=sha(OUT/('views_'+suffix+'.json'));save('forward_'+suffix+'_receipt.json',receipt);return receipt

def geometry():
 prepare();sources={m:load(PREV/(m+'_full_predictions.json')) for m in ['rgb','thermal']};g=P['geometry'];mats,receipt=c.bootstrap_affines(g['thermal_points'],g['rgb_points'],A,32);save('bootstrap_geometry.json',receipt)
 proto=load(OUT/'protocol.json');audits=[]
 for i in proto['registration']['sample_indices']:
  r=sources['rgb'][i];t=sources['thermal'][i];im=cv2.imread(r['image_path']);th=cv2.imread(t['image_path'])
  for variant in proto['registration']['variants']:
   z=c.feature_proposal(im,th,r['detections'],t['detections'],variant);z['pair_index']=i;audits.append(z)
 save('registration_audit.json',dict(attempts=audits,holdout_passes=sum(v['accepted'] for v in audits),promoted=0,reason='No independent full-field neutral correspondences verified; no fire-box/UV-fitted correction used'))
 # Choose actual zero-overlap cases by geometry only (not UV), ensure three with real R/T.
 zeros=[]
 for i,(r,t) in enumerate(zip(sources['rgb'],sources['thermal'])):
  if r['detections'] and t['detections'] and max(c.piou(c.corners(rb['bbox']),c.project(c.corners(tb['bbox']),A)) for rb in r['detections'] for tb in t['detections'])==0:zeros.append(i)
 selected=list(dict.fromkeys([zeros[0],zeros[len(zeros)//2],817]))
 save('selected_geometry_cases.json',dict(all_zero_indices=zeros,selected=selected))
 for i in selected:
  r=sources['rgb'][i];t=sources['thermal'][i];im=cv2.imread(r['image_path']);th=cv2.imread(t['image_path']);ov=im.copy();thov=th.copy()
  for b in r['detections']:cv2.rectangle(ov,tuple(np.int32(b['bbox'][:2])),tuple(np.int32(b['bbox'][2:])),(0,255,0),1)
  for b in t['detections']:
   cv2.rectangle(thov,tuple(np.int32(b['bbox'][:2])),tuple(np.int32(b['bbox'][2:])),(255,255,0),1);cv2.polylines(ov,[np.int32(c.project(c.corners(b['bbox']),A))],True,(0,0,255),1)
  cv2.putText(ov,f'pair{i} original coordinates',(3,14),0,.38,(255,255,255),1)
  cv2.imwrite(str(OUT/f'geometry_case_{i}.png'),prior.montage([im,th,thov,ov]))
 print('REGISTRATION',len(audits),'holdout passes',sum(v['accepted'] for v in audits),'zero cases',selected,flush=True)

def spatial_matrices(rgb,thermal,B):
 n,m=len(rgb),len(thermal);s=np.zeros((n,m))
 if not n or not m:return s
 rc=np.array([c.corners(b['bbox']) for b in rgb]);tc=np.array([c.project(c.corners(b['bbox']),B) for b in thermal]);rl=rc.min(1);rh=rc.max(1);tl=tc.min(1);th=tc.max(1)
 possible=np.all(np.minimum(rh[:,None,:],th[None,:,:])>np.maximum(rl[:,None,:],tl[None,:,:]),axis=2)
 for i,j in zip(*np.nonzero(possible)):s[i,j]=c.piou(rc[i],tc[j])
 return s

def features(boxes,shape):
 h,w=shape
 out=[]
 for b in boxes:
  x,y,X,Y=b['bbox'];score=np.clip(b['score'],1e-6,1-1e-6);bw=max(X-x,1e-6);bh=max(Y-y,1e-6)
  out.append([float(np.log(score/(1-score))),float(np.log(bw*bh/(w*h))),float(np.log(bw/bh))])
 return np.array(out).reshape(-1,3)

def fit_calibrator(rows,y,train):
 from sklearn.linear_model import LogisticRegression
 fs=[];ys=[];weights=[]
 for i in train:
  x=features(rows[i]['detections'],rows[i]['orig_shape'])
  if len(x):fs.extend(x.tolist());ys.extend([int(y[i])]*len(x));weights.extend([1/len(x)]*len(x))
 x=np.array(fs);yy=np.array(ys);mu=x.mean(0);sd=np.maximum(x.std(0),1e-6)
 model=LogisticRegression(C=1,max_iter=1000,random_state=741).fit((x-mu)/sd,yy,sample_weight=weights)
 record=dict(mean=mu.tolist(),std=sd.tolist(),coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),train_indices=[int(v) for v in train],features=['logit_score','log_area_fraction','log_aspect'],supervision='weak frame labels, not verified box correctness')
 return record

def apply_calibrator(rows,record):
 mu=np.array(record['mean']);sd=np.array(record['std']);co=np.array(record['coef']);inter=record['intercept'];out=[]
 for r in rows:
  z=(features(r['detections'],r['orig_shape'])-mu)/sd@co+inter;out.append((1/(1+np.exp(-np.clip(z,-60,60)))).tolist())
 return out

def evaluate():
 prepare();proto=load(OUT/'protocol.json');assert sha(OUT/'protocol.json')==load(OUT/'protocol_receipt.json')['sha256']
 rows=prior.old.parse();y=np.array([r['UV_reference'] for r in rows]);groups=np.array(proto['groups']);rgb=load(OUT/'rgb_all_predictions.json');thermal=load(OUT/'thermal_all_predictions.json');mats=[np.array(v) for v in load(OUT/'bootstrap_geometry.json')['matrix_distribution']]
 shapes=[(len(r['detections']),len(t['detections'])) for r,t in zip(rgb,thermal)];offsets=np.cumsum([0]+[n*m for n,m in shapes]);cache=OUT/'geometry_cache.npz'
 if cache.exists():
  z=np.load(cache);nominal=[z['nominal'][offsets[i]:offsets[i+1]].reshape(sh) for i,sh in enumerate(shapes)];expected=[z['expected'][offsets[i]:offsets[i+1]].reshape(sh) for i,sh in enumerate(shapes)];q10=[z['q10'][offsets[i]:offsets[i+1]].reshape(sh) for i,sh in enumerate(shapes)]
 else:
  nominal=[];expected=[];q10=[]
  for i,(r,t) in enumerate(zip(rgb,thermal)):
   rr=r['detections'];tt=t['detections'];s=spatial_matrices(rr,tt,A);samples=np.array([spatial_matrices(rr,tt,B) for B in mats]);nominal.append(s);expected.append(samples.mean(0));q10.append(np.quantile(samples,.1,axis=0))
   if i%200==0:print('GEOMETRY',i,flush=True)
  np.savez_compressed(cache,nominal=np.concatenate([s.ravel() for s in nominal]),expected=np.concatenate([s.ravel() for s in expected]),q10=np.concatenate([s.ravel() for s in q10]),offsets=offsets,shapes=np.array(shapes))
 bounds={};diag=[]
 for name,sp in [('nominal',nominal),('expected',expected),('q10',q10)]:
  maxima=np.array([s.max() if s.size else 0 for s in sp]);covered=maxima>=.001;reasons=collections.Counter()
  for i in np.flatnonzero((y==1)&~covered):reasons['no_RGB' if not rgb[i]['detections'] else 'no_thermal' if not thermal[i]['detections'] else 'zero_overlap' if maxima[i]==0 else 'tiny_overlap']+=1
  bounds[name]=dict(positive_covered=int(sum(covered&(y==1))),positive_uncovered=int(sum(~covered&(y==1))),negative_covered=int(sum(covered&(y==0))),uncovered_positive_reasons=dict(reasons),target_possible=bool(sum(covered&(y==1))>=1263))
 for i in range(len(rows)):
  z=dict(pair_index=i,reference=int(y[i]),group=int(groups[i]),rgb_count=len(rgb[i]['detections']),thermal_count=len(thermal[i]['detections']))
  for k,sp in [('nominal',nominal),('expected',expected),('q10',q10)]:z[k+'_max_iou']=float(sp[i].max()) if sp[i].size else 0.
  diag.append(z)
 save('candidate_bounds_before_calibration.json',dict(utc=now(),bounds=bounds,trial_limit=32,exhaustive_search=False,reason='Fixed bounded diagnostic; no threshold can exceed real paired geometry coverage'));csvsave('coverage_by_frame.csv',diag)
 print('BOUNDS',bounds,flush=True)
 baselines={}
 for m,v in [('rgb',rgb),('thermal',thermal)]:
  for pool,s in [('multiview',v),('original320',load(OUT/(m+'_all_full320.json')))]:baselines[m+'_'+pool]=prior.old.metrics(y.tolist(),[int(any(b['score']>.5 for b in r['detections'])) for r in s])
 thresholds=proto['calibration']['thresholds'];entries={};all_evidence=[dict(pair_index=i,reference=int(y[i]),group=int(groups[i]),full={},oof={}) for i in range(len(y))];trials=[]
 for geometry,sp in [('nominal',nominal),('expected',expected)]:
  for mode in ['raw','calibrated']:
   branch=geometry+'_'+mode;oof=np.zeros(len(y),int);folds=[];full_result=None
   for fold in [-1]+np.unique(groups).tolist():
    train=np.arange(len(y)) if fold==-1 else np.flatnonzero(groups!=fold);test=np.arange(len(y)) if fold==-1 else np.flatnonzero(groups==fold);scope='full' if fold==-1 else 'fold_'+str(fold)
    calibrators={};post=None
    if mode=='calibrated':
     calibrators={m:fit_calibrator(v,y,train) for m,v in [('rgb',rgb),('thermal',thermal)]};post={m:apply_calibrator(v,calibrators[m]) for m,v in [('rgb',rgb),('thermal',thermal)]}
    outputs=[]
    for i in range(len(y)):
     po=(post['rgb'][i],post['thermal'][i]) if post else None;o=c.fuse(rgb[i]['detections'],thermal[i]['detections'],A,posteriors=po,spatial=sp[i]);outputs.append(o)
    scores=np.array([v['masses'][0] for v in outputs]);present=np.array([v['selected'] is not None for v in outputs]);preds=np.array([present&(scores>t) for t in thresholds]);fp=np.sum(preds[:,train][:,y[train]==0],axis=1);tp=np.sum(preds[:,train][:,y[train]==1],axis=1);k=int(np.lexsort((np.arange(len(thresholds)),-tp,fp))[0]);threshold=thresholds[k]
    freeze=dict(utc=now(),branch=branch,scope=scope,threshold=threshold,index=k,train_indices=train.tolist(),heldout_indices=[] if fold==-1 else test.tolist(),calibrators=calibrators,selected_on_training_only=True,geometry_semantics='actual nominal polygon IoU' if geometry=='nominal' else 'expected IoU over fixed unvalidated bootstrap transform distribution, not nominal IoU')
    save(branch+'_'+scope+'_freeze.json',freeze)
    for j,t in enumerate(thresholds):trials.append(dict(branch=branch,scope=scope,threshold=t,**prior.old.metrics(y[train].tolist(),preds[j,train].astype(int).tolist())))
    predictions=preds[k].astype(int);metric=prior.old.metrics(y[test].tolist(),predictions[test].tolist())
    if fold==-1:full_result=metric
    else:oof[test]=predictions[test];folds.append(dict(fold=fold,threshold=threshold,heldout=metric))
    for i in test:
     o=outputs[i];o['fire']=int(predictions[i]);o['threshold']=threshold;o['geometry_semantics']=geometry;o['bpa_mode']=mode
     if o['selected']:
      ii=o['selected']['rgb_index'];jj=o['selected']['thermal_index'];o['nominal_iou']=float(nominal[i][ii,jj]);o['expected_iou']=float(expected[i][ii,jj]);o['iou_q10']=float(q10[i][ii,jj])
     all_evidence[i]['full' if fold==-1 else 'oof'][branch]=o
   entries[branch]=dict(full=full_result,oof=prior.old.metrics(y.tolist(),oof.tolist()),folds=folds,primary=geometry=='nominal')
   print('RESULT',branch,entries[branch]['full'],entries[branch]['oof'],flush=True)
 save('frame_evidence.json',all_evidence);csvsave('all_32_trials_train_only.csv',trials)
 details=[]
 for i,e in enumerate(all_evidence):
  z=dict(pair_index=i,reference=int(y[i]),group=int(groups[i]))
  for scope in ['full','oof']:
   for branch,o in e[scope].items():
    for k,v in dict(fire=o['fire'],mF=o['masses'][0],mN=o['masses'][1],mU=o['masses'][2],K=o['K'],spatial=o['iou']).items():z[scope+'_'+branch+'_'+k]=v
  details.append(z)
 csvsave('frame_predictions.csv',details)
 targets={}
 for branch,v in entries.items():
  targets[branch]={}
  for scope in ['full','oof']:
   mm=v[scope];targets[branch][scope]=dict(original_target=bool(mm['FP']==0 and mm['FN']<=25),fair_updated=bool(mm['FP']<=min(baselines['rgb_multiview']['FP'],baselines['thermal_multiview']['FP']) and mm['FN']<min(baselines['rgb_multiview']['FN'],baselines['thermal_multiview']['FN'])))
 save('metrics.json',dict(source_sha256=sha(prior.old.SOURCE),N=len(y),positive=int(sum(y)),negative=int(sum(y==0)),UV_runtime=False,independent_test=False,trials=32,baselines=baselines,results=entries,targets=targets,bounds=bounds,registration_promoted=False))

def anchor_evaluate():
 import itertools
 prepare();rgb=load(OUT/'rgb_all_full320.json');thermal=load(OUT/'thermal_all_predictions.json');rows=prior.old.parse();y=np.array([v['UV_reference'] for v in rows]);groups=np.array(P['all_groups']);grid=[dict(rT=rt,threshold=th,soft=soft) for rt,th,soft in itertools.product([.05,.1,.2,.3],[.5,.55,.6,.65],[False,True])]
 receipt=OUT/'anchor_protocol.json'
 if not receipt.exists():save(receipt.name,dict(utc=now(),adaptive_followup=True,reason='Symmetric multiview creates RGB false alarms; preserve original real RGB anchors, thermal multiscale repairs geometric coverage',prior_full_label_exposure=True,not_independent_selection=True,fixed=dict(rT=.1,threshold=.5,soft=False),grid=grid,trials=len(grid),same_candidate_baselines=['rgb_original320','thermal_multiview'],all_multiview_RGB_control_not_dominated=True,gate='actual nominal polygon IoU>=.001',UV_runtime=False))
 pred=[];outputs=[]
 for cfg in grid:
  z=[c.anchor_fuse(r['detections'],t['detections'],A,**cfg) for r,t in zip(rgb,thermal)];outputs.append(z);pred.append([o['fire'] for o in z])
 pred=np.array(pred);oof=np.zeros(len(y),int);assigned=np.zeros(len(y),int);folds=[]
 def choose(ix):
  fp=np.sum(pred[:,ix][:,y[ix]==0],axis=1);tp=np.sum(pred[:,ix][:,y[ix]==1],axis=1);return int(np.lexsort((np.arange(len(grid)),-tp,fp))[0])
 full=choose(np.arange(len(y)));fixed=next(k for k,cfg in enumerate(grid) if cfg==dict(rT=.1,threshold=.5,soft=False))
 for g in np.unique(groups):
  train=np.flatnonzero(groups!=g);test=np.flatnonzero(groups==g);k=choose(train);save('anchor_fold_'+str(g)+'_freeze.json',dict(utc=now(),config=grid[k],index=k,train_indices=train.tolist(),test_indices=test.tolist()));oof[test]=pred[k,test];assigned[test]=k;folds.append(dict(fold=int(g),config=grid[k],test=prior.old.metrics(y[test].tolist(),oof[test].tolist())))
 evidence=[];details=[]
 for i in range(len(y)):
  e=dict(pair_index=i,reference=int(y[i]),group=int(groups[i]),fixed=outputs[fixed][i],full=outputs[full][i],oof=outputs[assigned[i]][i]);evidence.append(e)
  z=dict(pair_index=i,**rows[i],group=int(groups[i]))
  for scope in ['fixed','full','oof']:
   o=e[scope];z.update({scope+'_'+k:v for k,v in dict(fire=o['fire'],mF=o['masses'][0],mN=o['masses'][1],mU=o['masses'][2],K=o['K'],actual_nominal_iou=o['iou']).items()})
  details.append(z)
 csvsave('anchor_frame_predictions.csv',details);save('anchor_evidence.json',evidence);csvsave('anchor_all_32_trials.csv',[dict(index=k,**cfg,**prior.old.metrics(y.tolist(),pred[k].tolist())) for k,cfg in enumerate(grid)])
 metrics={scope:prior.old.metrics(y.tolist(),pp.tolist()) for scope,pp in [('fixed_exposed',pred[fixed]),('full_descriptive',pred[full]),('group_oof_exploratory',oof)]};baselines={m:prior.old.metrics(y.tolist(),[int(any(b['score']>.5 for b in row['detections'])) for row in v]) for m,v in [('rgb_native320',rgb),('thermal_multiview',thermal)]}
 covered=np.array([o['selected'] is not None for o in outputs[fixed]]);result=dict(metrics=metrics,baselines=baselines,fixed=grid[fixed],full_config=grid[full],folds=folds,coverage=dict(positive_covered=int(sum(covered&(y==1))),positive_uncovered=int(sum(~covered&(y==1))),negative_covered=int(sum(covered&(y==0)))),UV_runtime=False,independent_test=False,prior_label_exposure=True,targets={scope:dict(original_target=bool(v['FP']==0 and v['FN']<=25),fair_same_selected_views=bool(v['FP']<=min(baselines[m]['FP'] for m in baselines) and v['FN']<min(baselines[m]['FN'] for m in baselines))) for scope,v in metrics.items()})
 save('anchor_metrics.json',result)
 # Make actual selected-box overlays, not proposal rectangles.
 selected=load(OUT/'selected_geometry_cases.json')['selected']
 for i in selected:
  r=rgb[i];t=thermal[i];im=cv2.imread(r['image_path']);th=cv2.imread(t['image_path']);ov=im.copy();q=outputs[fixed][i]['selected']
  if q:
   b=q['rgb']['bbox'];tb=q['thermal']['bbox'];cv2.rectangle(ov,tuple(np.int32(b[:2])),tuple(np.int32(b[2:])),(0,255,0),1);cv2.rectangle(th,tuple(np.int32(tb[:2])),tuple(np.int32(tb[2:])),(255,255,0),1);cv2.polylines(ov,[np.int32(c.project(c.corners(tb),A))],True,(0,0,255),1)
  cv2.imwrite(str(OUT/f'anchor_selected_{i}.png'),prior.montage([cv2.imread(r['image_path']),th,ov]))
 print('ANCHOR',json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
 action=sys.argv[1];globals()[action]()
