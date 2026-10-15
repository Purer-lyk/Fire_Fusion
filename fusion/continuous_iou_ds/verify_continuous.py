"""Independent all-row polygon/Hungarian/mass/CSV/metrics verification; no continuous_core calls."""
from pathlib import Path
import sys,json,csv,hashlib,zipfile,inspect,collections
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from docx import Document
OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'adaptive_spatial_ds'
ROOT=OUT.parent.parent
sys.dont_write_bytecode=True
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def polygon(rb,tb,A):
 x,y,X,Y=rb;rp=np.float32([[x,y],[X,y],[X,Y],[x,Y]])
 x,y,X,Y=tb;corners=np.float32([[x,y],[X,y],[X,Y],[x,Y]]);tp=np.float64(corners)@np.asarray(A)[:,:2].T+np.asarray(A)[:,2];tp=np.float32(tp)
 area1=abs(cv2.contourArea(rp));area2=abs(cv2.contourArea(tp));inter=float(cv2.intersectConvexConvex(rp,tp)[0]);union=area1+area2-inter
 return max(0.,min(1.,inter/union)) if union else 0.

def counts(y,p):
 return dict(TP=sum(a==1 and b==1 for a,b in zip(y,p)),FP=sum(a==0 and b==1 for a,b in zip(y,p)),FN=sum(a==1 and b==0 for a,b in zip(y,p)),TN=sum(a==0 and b==0 for a,b in zip(y,p)))

def verify():
 protocol=load(OUT/'protocol.json');freeze=load(OUT/'freeze_receipt.json');assert sha(OUT/'protocol.json')==freeze['protocol_sha256'];assert sha(OUT/'protected_sources.json')==freeze['protected_sha256']
 protected=load(OUT/'protected_sources.json')
 for path,h in protected.items():assert sha(path)==h,('protected source changed',path)
 R=load(OLD/'rgb_all_full320.json');T=load(OLD/'thermal_all_predictions.json');E=load(OUT/'frame_evidence.json');metrics=load(OUT/'metrics.json');A=protocol['geometry'];reference=Path(protocol['reference_source']).read_text(encoding='utf-8-sig').splitlines();reference=[v.split() for v in reference if v.strip() and not v.lstrip().startswith('#')];assert len(reference)==1528
 y=[int(r[3]) for r in reference];assert collections.Counter(y)=={1:1288,0:240}
 csvrows=list(csv.DictReader((OUT/'frame_predictions.csv').open(encoding='utf-8-sig')));assert len(E)==len(csvrows)==1528
 allpool=list(csv.DictReader((OUT/'all_matched_candidates.csv').open(encoding='utf-8-sig')))
 allpoolmap={(int(q['pair_index']),float(q['hard_gate']),int(q['rgb_index']),int(q['thermal_index'])):q for q in allpool};assert len(allpoolmap)==len(allpool)
 preds={k:[] for k in protocol['policies']};verified_matches=0;mass_changes=0
 for i,(r,t,e,row) in enumerate(zip(R,T,E,csvrows)):
  assert i==int(row['pair_index'])==e['pair_index'];assert reference[i][0]==e['rgb_name']==Path(r['image_path']).name;assert reference[i][1]==e['thermal_name']==Path(t['image_path']).name;assert e['UV_reference']==y[i]==int(row['UV_reference'])
  rb=r['detections'];tb=t['detections'];n=len(rb);nt=len(tb);s=np.array([[polygon(a['bbox'],b['bbox'],A) for b in tb] for a in rb]).reshape(n,nt)
  matches={}
  for gate in [.001,.01,.3]:
   w=np.zeros((n+nt,n+nt));w[:n,:nt]=np.where(s>=gate,s,-1.);rr,tt=linear_sum_assignment(-w);pairs=[(int(a),int(b),float(s[a,b])) for a,b in zip(rr,tt) if a<n and b<nt and s[a,b]>=gate];matches[gate]=pairs
   assert [(a,b) for a,b,_ in pairs]==[(q['rgb_index'],q['thermal_index']) for q in e['matches_by_gate'][str(gate)]]
   for a,b,spatial in pairs:
    q=allpoolmap[(i,gate,a,b)];g=max(0.,min(1.,(spatial-.1)/.2));mf=1-(1-rb[a]['score'])*(1-.1*g*tb[b]['score']);assert abs(float(q['s'])-spatial)<1e-6;assert abs(float(q['g'])-g)<1e-5;assert abs(float(q['mF_continuous'])-mf)<1e-7;verified_matches+=1
  for key,(gate,continuous) in protocol['policies'].items():
   candidates=[]
   for a,b,spatial in matches[gate]:
    g=max(0.,min(1.,(spatial-.1)/.2)) if continuous else 1.;mf=1-(1-rb[a]['score'])*(1-.1*g*tb[b]['score']);candidates.append((mf,a,b,spatial))
   best=max(candidates,key=lambda x:x[0]) if candidates else (0.,None,None,0.);o=e['outputs'][key];assert (o['rgb_index'],o['thermal_index'])==best[1:3];assert abs(o['mF']-best[0])<1e-7;assert abs(o['s']-best[3])<1e-6;assert abs(sum(o[k] for k in ['mF','mN','mU'])-1)<1e-12;assert min(o[k] for k in ['mF','mN','mU'])>=0;assert o['mN']==0 and o['K']==0
   assert o['fire']==int(best[0]>.5)==int(row[key+'_fire']);assert abs(float(row[key+'_mF'])-best[0])<1e-7;preds[key].append(o['fire'])
   for field in ['cR','cT','s','g','actual_rT','mF','mN','mU']:assert abs(float(row[key+'_'+field])-o[field])<1e-12
  fixed=e['outputs']['fixed_source_gate001'];new=e['outputs']['continuous_primary_source_gate001'];assert new['mF']<=fixed['mF']+1e-12;mass_changes+=abs(new['mF']-fixed['mF'])>1e-12
 for key,p in preds.items():
  cc=counts(y,p);assert cc=={k:metrics['results'][key][k] for k in cc};v=metrics['results'][key];assert abs(v['FPR']-cc['FP']/240)<1e-14;assert abs(v['FNR']-cc['FN']/1288)<1e-14
  expected=dict(precision=cc['TP']/(cc['TP']+cc['FP']),recall=cc['TP']/1288,F1=2*cc['TP']/(2*cc['TP']+cc['FP']+cc['FN']),accuracy=(cc['TP']+cc['TN'])/1528)
  for name,value in expected.items():assert abs(v[name]-value)<1e-14
 assert preds['fixed_source_gate001']==preds['continuous_primary_source_gate001']
 assert preds['fixed_requested_gate01']==preds['continuous_requested_gate01']
 for key,p in preds.items():
  if key=='fixed_source_gate001':continue
  oldp=preds['fixed_source_gate001'];expected_lost=[i for i in range(1528) if y[i]==1 and oldp[i]==1 and p[i]==0];assert expected_lost==metrics['transitions'][key]['oldTP_newFN']
 assert mass_changes==metrics['affected']['selected_g_less_1']==81
 for key,path in [('rgb_original320','rgb_all_full320.json'),('thermal_multiview','thermal_all_predictions.json'),('rgb_multiview_auxiliary','rgb_all_predictions.json')]:
  p=[int(any(b['score']>.5 for b in r['detections'])) for r in load(OLD/path)];cc=counts(y,p);assert cc=={k:metrics['baselines'][key][k] for k in cc}
 image_audit=load(OUT/'image_hash_verification.json');assert len(image_audit)==3056
 for q in image_audit:assert sha(q['image_path'])==q['sha256']
 for model in protocol['models'].values():assert sha(model['source'])==model['sha256']
 # Deterministic dense continuity/normalization/monotonicity/boundary tests; labels absent from API.
 import continuous_core as c
 assert list(inspect.signature(c.mass).parameters)==['cR','cT','s','continuous'];assert 'uv' not in str(inspect.signature(c.select)).lower()
 for cr in [0.,.49,.5,.8,1.]:
  for ct in [0.,.3,1.]:
   masses=[c.mass(cr,ct,float(s)) for s in np.linspace(0,1,1001)];assert all(abs(sum(v)-1)<1e-12 and min(v)>=0 for v in masses);assert all(a[0]<=b[0]+1e-12 for a,b in zip(masses,masses[1:]));np.testing.assert_allclose(c.mass(cr,ct,.3),c.mass(cr,ct,.3,False),atol=1e-12)
 assert c.gain(0)==c.gain(.1)==0 and c.gain(.3)==c.gain(1)==1
 for boundary in [.1,.3]:assert abs(c.gain(boundary-1e-9)-c.gain(boundary+1e-9))<2e-8
 docpath=ROOT/'reports/连续IoU可靠性_DS复验报告.docx'
 with zipfile.ZipFile(docpath) as z:assert z.testzip() is None
 doc=Document(docpath);text='\n'.join(p.text for p in doc.paragraphs)+'\n'+'\n'.join('|'.join(c.text for c in row.cells) for t in doc.tables for row in t.rows)
 for value in ['1265','1261','1190','0.001','0.01','0.3','5.77','独立','0.449','727','747','879']:assert value in text
 (OUT/'report_readback.txt').write_text(text,encoding='utf-8')
 receipt=dict(verified=True,all_frames=1528,policies=5,frame_policy_checks=1528*5,independent_polygon_hungarian=True,all_match_rows=verified_matches,source_primary_match_rows=sum(float(q['hard_gate'])==.001 for q in allpool),images_sha256_verified=len(image_audit),model_hashes_verified=2,protected_sources_unchanged=len(protected),mass_changed_frames=int(mass_changes),UV_runtime=False,fresh_forward=False,protocol_sha256=sha(OUT/'protocol.json'),dense_continuity_normalization_monotonicity_tests=True,docx=dict(path=str(docpath),sha256=sha(docpath),crc_ok=True,tables=len(doc.tables)),artifacts={name:sha(OUT/name) for name in ['continuous_core.py','run_continuous.py','verify_continuous.py','frame_predictions.csv','all_matched_candidates.csv','frame_evidence.json','metrics.json','delta_cases.json']})
 (OUT/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':verify()
