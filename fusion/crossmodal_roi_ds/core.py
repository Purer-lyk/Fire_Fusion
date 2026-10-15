import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

def polygon_iou(p,q):
 p=np.asarray(p,dtype=np.float32); q=np.asarray(q,dtype=np.float32)
 inter,_=cv2.intersectConvexConvex(p,q)
 union=abs(cv2.contourArea(p))+abs(cv2.contourArea(q))-inter
 return float(inter/union) if union>0 else 0.
def match(rgb,thermal,a,gate=.3):
 n,m=len(rgb),len(thermal)
 if not n or not m: return []
 s=np.array([[polygon_iou(corners(r['bbox']),transform(corners(t['bbox']),a)) for t in thermal] for r in rgb])
 weights=np.zeros((n+m,n+m)); weights[:n,:m]=np.where(s>=gate,s,-1)
 rows,cols=linear_sum_assignment(-weights)
 return [(int(i),int(j),float(s[i,j])) for i,j in zip(rows,cols) if i<n and j<m and s[i,j]>=gate]
def calibrate(c,calibrator=None):
 return float(c if calibrator is None else calibrator(c))
def combine(cr,ct,s,uncertainty=.1):
 reliability=.5+.5*s
 def bpa(c):
  c=calibrate(c); return np.array([reliability*(1-uncertainty)*c,reliability*(1-uncertainty)*(1-c),1-reliability*(1-uncertainty)])
 f,n,u=bpa(cr); g,h,v=bpa(ct); conflict=f*h+n*g
 if conflict>=1-1e-9: return [0.,0.,1.],float(conflict)
 return [(f*g+f*v+u*g)/(1-conflict),(n*h+n*v+u*h)/(1-conflict),u*v/(1-conflict)],float(conflict)
def uv_score(mf,uv,lam=.5): return mf*((1-lam)+lam*int(uv))
def decide(rgb,thermal,pairs,uv,threshold=.5):
 evidence=[]
 for i,j,s in pairs:
  mass,k=combine(rgb[i]['score'],thermal[j]['score'],s)
  evidence.append({'rgb_index':i,'thermal_index':j,'iou':s,'mass':mass,'conflict':k})
 evidence.sort(key=lambda e:(-e['mass'][0],e['rgb_index'],e['thermal_index']))
 best=evidence[0] if evidence else None
 mf=best['mass'][0] if best else 0.
 return bool(best and mf>threshold),bool(best and uv_score(mf,uv)>threshold),best,evidence
def metrics(truth,pred):
 tp=sum(bool(y) and bool(p) for y,p in zip(truth,pred)); fp=sum(not y and bool(p) for y,p in zip(truth,pred)); fn=sum(bool(y) and not p for y,p in zip(truth,pred)); tn=sum(not y and not p for y,p in zip(truth,pred))
 div=lambda x,y:x/y if y else 0.
 precision=div(tp,tp+fp); recall=div(tp,tp+fn); fpr=div(fp,fp+tn); fnr=div(fn,fn+tp)
 return dict(TP=tp,FP=fp,FN=fn,TN=tn,FPR=fpr,FNR=fnr,precision=precision,recall=recall,F1=div(2*precision*recall,precision+recall),balanced_accuracy=(recall+1-fpr)/2,meets_all_target=bool(fpr<=0 and fnr<666/1398 and fpr<126/130))

def corners(box):
 x1,y1,x2,y2=box
 return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],dtype=np.float32)
def transform(points,a):
 return np.asarray(points)@np.asarray(a)[:,:2].T+np.asarray(a)[:,2]
def inverse(a):
 return cv2.invertAffineTransform(np.asarray(a,dtype=float))
def offset_box(box,bounds):
 return [float(box[0]+bounds[0]),float(box[1]+bounds[1]),float(box[2]+bounds[0]),float(box[3]+bounds[1])]
def crop_bounds(points,shape,padding=.25,margin=10):
 low=np.min(points,axis=0); high=np.max(points,axis=0); pad=(high-low)*padding+margin
 h,w=shape[:2]
 return [max(0,int(np.floor(low[0]-pad[0]))),max(0,int(np.floor(low[1]-pad[1]))),min(w,int(np.ceil(high[0]+pad[0]))),min(h,int(np.ceil(high[1]+pad[1])))]
