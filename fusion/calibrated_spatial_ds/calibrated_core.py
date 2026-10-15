import numpy as np
from scipy.optimize import linear_sum_assignment

def fire_mass(cr,ct,s,rmin):
 """Nominal fire-only discounted simple supports; K=0, not calibrated posteriors."""
 r=rmin+(1-rmin)*s
 return float(1-(1-r*cr)*(1-r*ct))
def frame_score(rgb,thermal,ious,tr,tt,gate,rmin):
 ri=[i for i,b in enumerate(rgb) if b['score']>=tr]
 ti=[i for i,b in enumerate(thermal) if b['score']>=tt]
 if not ri or not ti:return 0.
 s=np.asarray(ious)[np.ix_(ri,ti)]; n,m=s.shape
 w=np.zeros((n+m,n+m));w[:n,:m]=np.where(s>=gate,s,-1)
 rr,cc=linear_sum_assignment(-w)
 values=[fire_mass(rgb[ri[i]]['score'],thermal[ti[j]]['score'],s[i,j],rmin) for i,j in zip(rr,cc) if i<n and j<m and s[i,j]>=gate]
 return max(values,default=0.)
def uv_score(mf,uv,lam):
 return float(mf*(lam+(1-lam)*int(uv)))
def target(m,thermal,rgb):
 return bool(m['FP']==0 and m['FNR']<thermal['FNR'] and rgb['FPR']>0)
