import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

def corners(b):
 x,y,X,Y=b
 return np.array([[x,y],[X,y],[X,Y],[x,Y]],dtype=np.float32)
def transform(p,A):return (np.c_[p,np.ones(len(p))]@np.asarray(A).T).astype(np.float32)
def polygon_iou(p,q):
 p=np.asarray(p,dtype=np.float32);q=np.asarray(q,dtype=np.float32)
 a=abs(cv2.contourArea(p));b=abs(cv2.contourArea(q));inter,_=cv2.intersectConvexConvex(p,q)
 return float(inter/(a+b-inter)) if a+b-inter>0 else 0.
def masses(cr,ct):
 u=(1-cr)*(1-ct)
 return 1-u,0.,u
def match(rgb,thermal,ious,tr=.2,tt=.1,gate=.01):
 ri=[i for i,b in enumerate(rgb) if b['score']>=tr];ti=[j for j,b in enumerate(thermal) if b['score']>=tt]
 if not ri or not ti:return []
 s=np.asarray(ious)[np.ix_(ri,ti)];n,m=s.shape
 w=np.zeros((n+m,n+m));w[:n,:m]=np.where(s>=gate,s,-1)
 rr,cc=linear_sum_assignment(-w)
 return [dict(r=ri[i],t=ti[j],iou=float(s[i,j]),mF=masses(rgb[ri[i]]['score'],thermal[ti[j]]['score'])[0]) for i,j in zip(rr,cc) if i<n and j<m and s[i,j]>=gate]
def aligned(rows,names):
 if len(rows)!=len(names) or len({r['pair_index'] for r in rows})!=len(names):raise ValueError('Missing or duplicate IDs')
 for i,(r,n) in enumerate(zip(rows,names)):
  if r['pair_index']!=i or r['image_name']!=n:raise ValueError('Order/name mismatch')
def metrics(y,p):
 assert len(y)==len(p) and len(y)>0
 tp=sum(a==b==1 for a,b in zip(y,p));tn=sum(a==b==0 for a,b in zip(y,p));fp=sum(a==0 and b==1 for a,b in zip(y,p));fn=sum(a==1 and b==0 for a,b in zip(y,p))
 div=lambda a,b:a/b if b else None
 return dict(N=len(y),TP=tp,FP=fp,FN=fn,TN=tn,accuracy=div(tp+tn,len(y)),precision=div(tp,tp+fp),recall=div(tp,tp+fn),F1=div(2*tp,2*tp+fp+fn),FPR=div(fp,fp+tn),FNR=div(fn,fn+tp))
