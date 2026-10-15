import numpy as np

def bpa(score,reliability,u0=0):
 """Positive-only simple support. Raw detector score is NOT probability calibration."""
 if not (0<=score<=1 and 0<=reliability<=1 and 0<=u0<=1): raise ValueError('BPA range')
 f=(1-u0)*score*reliability
 return np.array([f,0.,1-f])

def combine(a,b):
 af,an,au=a;bf,bn,bu=b
 k=af*bn+an*bf
 if k>=1: return np.array([0.,0.,1.]),float(k)
 return np.array([(af*bf+af*bu+au*bf)/(1-k),(an*bn+an*bu+au*bn)/(1-k),au*bu/(1-k)]),float(k)

def pair_mass(cr,ct,s,cfg):
 g=min(1.,max(0.,s/.3)) if cfg['soft'] else 1.
 rr=cfg['rR'];rt=cfg['rT']*g
 mr=bpa(cr,rr,cfg['u0']);mt=bpa(ct,rt,cfg['u0']);m,k=combine(mr,mt)
 return dict(mF=float(m[0]),mN=float(m[1]),mU=float(m[2]),K=k,rR_effective=rr,rT_effective=rt,soft_iou=g,bpa_rgb=mr.tolist(),bpa_thermal=mt.tolist())

def frame(rgb,thermal,ious,cfg):
 from scipy.optimize import linear_sum_assignment
 ri=[i for i,b in enumerate(rgb) if b['score']>=cfg['tR']];ti=[i for i,b in enumerate(thermal) if b['score']>=cfg['tT']]
 matches=[]
 if ri and ti:
  s=np.asarray(ious)[np.ix_(ri,ti)];n,m=s.shape
  w=np.zeros((n+m,n+m));w[:n,:m]=np.where(s>=cfg['gate'],s,-1)
  rr,cc=linear_sum_assignment(-w)
  for i,j in zip(rr,cc):
   if i<n and j<m and s[i,j]>=cfg['gate']:
    r,t=ri[i],ti[j]
    matches.append(dict(r=r,t=t,iou=float(s[i,j]),rgb_score=rgb[r]['score'],thermal_score=thermal[t]['score'],rgb_bbox=rgb[r]['bbox'],thermal_bbox=thermal[t]['bbox'],**pair_mass(rgb[r]['score'],thermal[t]['score'],float(s[i,j]),cfg)))
 best=max(matches,key=lambda x:x['mF'],default=None)
 out=dict(fire=int(best is not None and best['mF']>cfg['threshold']),matches=matches,selected=best)
 out.update({k:best[k] if best else v for k,v in [('mF',0.),('mN',0.),('mU',1.),('K',0.)]})
 return out

def select(predictions,labels,train_indices):
 """Lexicographic minimum calibration FP, then maximum TP; deterministic grid tie order."""
 pp=predictions[:,train_indices]; yy=labels[train_indices]
 fp=np.sum(pp[:,yy==0],axis=1);tp=np.sum(pp[:,yy==1],axis=1)
 return int(np.lexsort((np.arange(len(pp)),-tp,fp))[0])
