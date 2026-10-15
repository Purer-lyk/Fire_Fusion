"""Audited internal, group-blocked exploratory calibration; no independent test claim.
Run once. Allocations and finite grid written before search; heldout read for evaluation only after freeze.
"""
from pathlib import Path
import json,csv,hashlib,datetime,time,sys,collections,itertools
import numpy as np
import calibrated_core as c
ROOT=Path('C:/lyk/fire/qualityDetection'); OUT=ROOT/'fusion/calibrated_spatial_ds'; OLD=ROOT/'fusion/crossmodal_roi_ds'
sys.path.insert(0,str(OLD));import core as geometry

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):
 (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def csvsave(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def timestamp(name):
 parts=name.split('_');return datetime.datetime.strptime('_'.join(parts[:6]),'%Y_%m_%d_%H_%M_%S')+datetime.timedelta(seconds=int(parts[6])/10000)
def metrics(y,p):
 m=geometry.metrics(y,p);m.pop('meets_all_target');return m

def run():
 started=time.time();assert not (OUT/'frozen_selected.json').exists(),'Completed experiment exists; do not retune heldout.'
 rows=json.loads((OLD/'full_results.json').read_text());audit=json.loads((OLD/'full_image_audit.json').read_text());assert len(rows)==1528 and len(audit)==3056
 baseline={m:json.loads((ROOT/'fusion/supplied_models_uv_reference'/f'{m}_predictions.json').read_text())['images'] for m in ['rgb','thermal']}
 source=Path('C:/lyk/paper_data/paper_data_/paper_data/paper_sample.txt');lines=[s.split() for s in source.read_text().splitlines() if s.strip()]; assert len(lines)==1528
 protected_files=[source,*OLD.glob('*.*'),*(ROOT/'reports').glob('*.docx'),ROOT/'fusion/supplied_models_affine_iou/candidate_affine_calibration.json',ROOT/'fusion/supplied_models_uv_reference/rgb_predictions.json',ROOT/'fusion/supplied_models_uv_reference/thermal_predictions.json',Path('C:/Users/Administrator/AppData/Local/hermes/attachments/best.pt'),Path('C:/Users/Administrator/AppData/Local/hermes/attachments/best-2.pt')]
 protected={str(p):sha(p) for p in protected_files if p.is_file()}; actual=[]
 for item in audit:
  h=sha(item['path']);assert h==item['sha256'];actual.append(h)
 assert len(set(actual))==3056
 for i,r in enumerate(rows):
  assert r['pair_index']==i and [r['rgb_name'],r['thermal_name'],r['temperature_raw'],str(r['uv_reference'])]==lines[i]
  for m in baseline:assert baseline[m][i]['pair_index']==i and baseline[m][i]['image_name']==r[m+'_name']
 save('source_audit.json',{'images':audit,'verified_images':3056,'unique_hashes':3056,'source_sha256':sha(source),'protected_hashes':protected,'all_data_previously_seen':True,'models':json.loads((OLD/'full_model_metadata.json').read_text()),'reused_inference':json.loads((OLD/'full_execution.json').read_text()),'cache_sha256':sha(OLD/'full_results.json')})
 ts=[timestamp(r['rgb_name']) for r in rows];cuts=[0]+[i for i in range(1,len(rows)) if (ts[i]-ts[i-1]).total_seconds()>10 or ts[i]<ts[i-1] or rows[i]['uv_reference']!=rows[i-1]['uv_reference']]+[len(rows)]
 groups=[]
 for k,(a,b) in enumerate(zip(cuts,cuts[1:])):
  labels=collections.Counter(r['uv_reference'] for r in rows[a:b]);assert len(labels)==1
  groups.append({'group_id':f'temporal_gap_proxy_{k:02d}','start':a,'stop_exclusive':b,'label':rows[a]['uv_reference'],'count':b-a,'start_timestamp':ts[a].isoformat(),'end_timestamp':ts[b-1].isoformat()})
 allocation={}
 for label in [0,1]:
  gs=[g for g in groups if g['label']==label];assert len(gs)>=3
  total=sum(g['count'] for g in gs);cumulative=np.cumsum([g['count'] for g in gs]);k=min(range(1,len(gs)-1),key=lambda k:abs(cumulative[k-1]-.6*total));j=min(range(k+1,len(gs)),key=lambda j:abs(cumulative[j-1]-.8*total))
  for z,g in enumerate(gs):
   split='train' if z<k else 'calibration' if z<j else 'heldout'
   g['split']=split
   for i in range(g['start'],g['stop_exclusive']):allocation[i]=(split,g['group_id'])
 manifest=[{'pair_index':i,'split':allocation[i][0],'group_id':allocation[i][1],'rgb_name':r['rgb_name'],'thermal_name':r['thermal_name'],'uv_reference':r['uv_reference'],'rgb_sha256':audit[2*i]['sha256'],'thermal_sha256':audit[2*i+1]['sha256']} for i,r in enumerate(rows)]
 csvsave('split_allocation.csv',manifest)
 ids={s:[x['pair_index'] for x in manifest if x['split']==s] for s in ['train','calibration','heldout']};ids['full']=list(range(len(rows)))
 assert all(set(rows[i]['uv_reference'] for i in ids[s])=={0,1} for s in ['train','calibration','heldout'])
 splitmeta={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'allocation_sha256':sha(OUT/'split_allocation.csv'),'gap_seconds':10,'groups':groups,'distribution':{s:dict(collections.Counter(rows[i]['uv_reference'] for i in ix)) for s,ix in ids.items()},'disclosure':'Entire 1528 already seen. Internal retained exploratory evaluation only, NOT independent validation. Filename timestamp 4 digits interpreted as fractional seconds; gap groups are temporal proxies, NOT verified scenes. Class-stratified chronologically ordered WHOLE groups; cannot globally chronological-split and have both classes because negatives are May and positives August. Closest feasible 60/20/20 frame allocation within each class; unequal group sizes prevent exact ratio. No random-row splitting, no scene fabrication. Near duplicate/source-video leakage remains unverified. No supervised score fitting; train portion used only for descriptive evaluation.'}
 save('split_metadata.json',splitmeta)
 pairs=[(.1,.1),(.15,.15),(.2,.2),(.25,.25),(.3,.3),(.4,.4),(.5,.5),(.1,.2),(.2,.1),(.1,.3),(.3,.1),(.2,.3)]
 grid=[dict(tR=a,tT=b,iou_gate=g,rmin=r,decision_threshold=d) for (a,b),g,r,d in itertools.product(pairs,[.01,.05,.1,.2,.3],[.5,1.],[.3,.4,.5,.6])]
 assert len(grid)==480
 save('search_plan.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'grid':grid,'trial_count':len(grid),'selection':'Calibration ONLY: lexicographic minimum FP, then minimum FN, then deterministic grid index. Primary WITHOUT UV. No heldout retuning.','BPA':'(r*c,0,1-r*c), r=q_A*(rmin+(1-rmin)*IoU), q_A=1 nominal; no calibrated posterior claim. K=0, DS fire-only OR reinforcement. Absence unknown but final missing dual-modality gate ->0. Max over one-to-one matched supported candidates; no cumulative correlated-box fusion.','cache':'Real .1 full frame and .1 ROI detections reused, original imgsz640; only thresholds >=.1 permitted. Original ROI trigger .25 top3 each modality; no new below-cache threshold or fabricated proposal boxes.','uv':'Secondary only mF_prime=mF*[lambda+(1-lambda)*gUV], lambda=.5; UV never forces fire, circular UV-reference metrics.','registration':'Previously fitted affine candidate; NOT independently validated geometry.'})
 a=np.array(json.loads((ROOT/'fusion/supplied_models_affine_iou/candidate_affine_calibration.json').read_text())['affine_matrix_2x3']);ious={}
 def prepare(ix):
  for i in ix:
   r=rows[i];ious[i]=[[geometry.polygon_iou(geometry.corners(rb['bbox']),geometry.transform(geometry.corners(tb['bbox']),a)) for tb in r['thermal_detections']] for rb in r['rgb_detections']]
 def scores(ix,cfg):
  return [c.frame_score(rows[i]['rgb_detections'],rows[i]['thermal_detections'],ious[i],cfg['tR'],cfg['tT'],cfg['iou_gate'],cfg['rmin']) for i in ix]
 prepare(ids['calibration']);trials=[]
 truth=[rows[i]['uv_reference'] for i in ids['calibration']]
 for k,cfg in enumerate(grid):
  sc=scores(ids['calibration'],cfg);m=metrics(truth,[v>cfg['decision_threshold'] for v in sc]);trials.append({'trial_id':k,**cfg,**m})
 csvsave('calibration_trials.csv',trials);winner=min(trials,key=lambda t:(t['FP'],t['FN'],t['trial_id']));cfg=grid[winner['trial_id']]
 frozen={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selected_trial':winner,'parameters':cfg,'allocation_sha256':sha(OUT/'split_allocation.csv'),'search_plan_sha256':sha(OUT/'search_plan.json'),'trial_csv_sha256':sha(OUT/'calibration_trials.csv'),'heldout_evaluations_before_freeze':0,'formula':'fire-only nominal BPA, r=q_A*(rmin+(1-rmin)*IoU), q_A=1; mF=1-(1-r*cR)*(1-r*cT); decision strict >; dual real box gate; UV lambda .5 secondary only'}
 save('frozen_selected.json',frozen);freezehash=sha(OUT/'frozen_selected.json');print('FROZEN',json.dumps(frozen),flush=True)
 prepare([i for i in ids['full'] if i not in ious]);sc=scores(ids['full'],cfg);pred=[v>cfg['decision_threshold'] for v in sc];uv=[c.uv_score(v,r['uv_reference'],.5)>cfg['decision_threshold'] for v,r in zip(sc,rows)]
 outputs=[]
 for i,r in enumerate(rows):outputs.append({'pair_index':i,'split':allocation[i][0],'group_id':allocation[i][1],'uv_reference':r['uv_reference'],'mF':sc[i],'primary_vision_fire':int(pred[i]),'uv_mF':c.uv_score(sc[i],r['uv_reference'],.5),'secondary_uv_fire':int(uv[i]),'rgb_baseline_fire':int(bool(baseline['rgb'][i]['detections'])),'thermal_baseline_fire':int(bool(baseline['thermal'][i]['detections']))})
 csvsave('frame_predictions.csv',outputs);results={}
 for split,ix in ids.items():
  y=[rows[i]['uv_reference'] for i in ix];methods={}
  for name,pp in [('primary_vision',pred),('secondary_uv',uv),('rgb_baseline',[bool(x['detections']) for x in baseline['rgb']]),('thermal_baseline',[bool(x['detections']) for x in baseline['thermal']])]:methods[name]=metrics(y,[pp[i] for i in ix])
  for name in ['primary_vision','secondary_uv']:methods[name]['meets_target_same_subset']=c.target(methods[name],methods['thermal_baseline'],methods['rgb_baseline'])
  results[split]={'N':len(ix),'positive':sum(y),'negative':len(y)-sum(y),'methods':methods}
 # Label-dependent optimistic candidate coverage is diagnostic only, never a classifier or selection signal.
 bound={}
 for g in [.01,.05,.1,.2,.3]:
  covered=[any(s>=g for rr in ious[i] for s in rr) for i in ids['full']]
  bound[str(g)]={'positive_real_pair_coverage':sum(bool(r['uv_reference']) and v for r,v in zip(rows,covered)),'negative_real_pair_coverage':sum(not r['uv_reference'] and v for r,v in zip(rows,covered)),'meaning':'Optimistic spatial candidate upper bound ONLY, not selected predictions or achievable label-free performance.'}
 save('candidate_coverage_diagnostic.json',bound)
 save('metrics.json',{'reference':'UV-reference agreement; secondary UV input/reference circular; all 1528 previously seen; internal retained exploration, NOT independent validation','results':results,'selected_parameters':cfg,'freeze_sha256':freezehash,'heldout_evaluations':1,'target_full_and_heldout_primary':results['full']['methods']['primary_vision']['meets_target_same_subset'] and results['heldout']['methods']['primary_vision']['meets_target_same_subset'],'calibration_score_model':'NONE; nominal confidence supports and quality. Calibration is threshold search only.'})
 assert sha(OUT/'frozen_selected.json')==freezehash and all(sha(p)==h for p,h in protected.items())
 save('execution.json',{'elapsed_seconds':time.time()-started,'real_cached_rows':1528,'verified_source_images':3056,'unique_image_hashes':3056,'trials':480,'selection_labels':'calibration only','heldout_evaluations_after_freeze':1,'protected_unchanged':True,'frozen_config_sha256':freezehash,'new_detector_forwards':0,'reuse':'Real prior .1 full frame/ROI YOLO26n outputs; no new inference needed for thresholds >=.1','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 print(json.dumps({'selected':cfg,'results':results,'coverage':bound},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':run()
