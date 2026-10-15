import numpy as np
import cv2

def tiles(shape):
 """Fixed 2x2 views, 4/7 of each original dimension, 25% overlap per crop."""
 h,w=shape[:2];ch=int(np.ceil(h*4/7));cw=int(np.ceil(w*4/7))
 return [[x,y,x+cw,y+ch] for y in [0,h-ch] for x in [0,w-cw]]

def map_box(box,crop):
 return [float(v+crop[k%2]) for k,v in enumerate(box)]

def corners(b):
 x,y,X,Y=b;return np.array([[x,y],[X,y],[X,Y],[x,Y]],np.float32)

def project(p,A):return np.asarray(p)@np.asarray(A)[:,:2].T+np.asarray(A)[:,2]

def piou(p,q):
 p=np.asarray(p,np.float32);q=np.asarray(q,np.float32)
 a=abs(cv2.contourArea(p));b=abs(cv2.contourArea(q))
 if a<=0 or b<=0:return 0.
 inter=float(cv2.intersectConvexConvex(p,q)[0]);return max(0.,min(1.,inter/(a+b-inter)))

def expected_iou(r,t,matrices):
 vals=np.array([piou(corners(r),project(corners(t),a)) for a in matrices])
 return dict(mean=float(vals.mean()),q10=float(np.quantile(vals,.1)),std=float(vals.std()))

def combine(a,b):
 af,an,au=a;bf,bn,bu=b;k=af*bn+an*bf
 if k>=.95:
  return [af*bf+af*bu+au*bf,an*bn+an*bu+au*bn,au*bu+k],float(k)
 return [(af*bf+af*bu+au*bf)/(1-k),(an*bn+an*bu+au*bn)/(1-k),au*bu/(1-k)],float(k)

def fuse(rgb,thermal,A,threshold=.5,gate=.001,posteriors=None,spatial=None):
 """Real dual boxes required. No UV parameter exists. spatial optionally expected IoU, disclosed separately."""
 from scipy.optimize import linear_sum_assignment
 empty=dict(fire=0,masses=[0.,0.,1.],K=0.,iou=0.,selected=None)
 if not rgb or not thermal:return empty
 s=np.array([[piou(corners(r['bbox']),project(corners(t['bbox']),A)) for t in thermal] for r in rgb]) if spatial is None else np.asarray(spatial)
 n,m=s.shape;w=np.zeros((n+m,n+m));w[:n,:m]=np.where(s>=gate,s,-1.)
 ri,ti=linear_sum_assignment(-w);best=None
 for i,j in zip(ri,ti):
  if i>=n or j>=m or s[i,j]<gate:continue
  rel=.6+.4*min(1.,s[i,j]/.3)
  if posteriors is None:pr=rgb[i]['score'];pt=thermal[j]['score'];a=[pr,0,1-pr];b=[rel*pt,0,1-rel*pt]
  else:
   pr=float(posteriors[0][i]);pt=float(posteriors[1][j]);a=[.9*pr,.9*(1-pr),.1];b=[rel*.9*pt,rel*.9*(1-pt),1-rel*.9]
  masses,k=combine(a,b)
  o=dict(fire=int(masses[0]>threshold),masses=masses,K=k,iou=float(s[i,j]),selected=dict(rgb_index=int(i),thermal_index=int(j),rgb=rgb[i],thermal=thermal[j],bpa_rgb=a,bpa_thermal=b,thermal_reliability=rel,posterior_rgb=pr,posterior_thermal=pt))
  if best is None or o['masses'][0]>best['masses'][0]:best=o
 return best if best is not None else empty

def anchor_fuse(rgb,thermal,A,rT=.1,threshold=.5,soft=False):
 """Native RGB anchors, genuine multiview thermal. Simple support K=0, no class posterior claim."""
 from scipy.optimize import linear_sum_assignment
 empty=dict(fire=0,masses=[0.,0.,1.],K=0.,iou=0.,selected=None)
 if not rgb or not thermal:return empty
 s=np.array([[piou(corners(r['bbox']),project(corners(t['bbox']),A)) for t in thermal] for r in rgb]);n,m=s.shape;w=np.zeros((n+m,n+m));w[:n,:m]=np.where(s>=.001,s,-1);ri,ti=linear_sum_assignment(-w);out=[]
 for i,j in zip(ri,ti):
  if i>=n or j>=m or s[i,j]<.001:continue
  r=rgb[i];t=thermal[j];reliability=rT*(min(1.,s[i,j]/.3) if soft else 1.);a=[r['score'],0.,1-r['score']];b=[reliability*t['score'],0.,1-reliability*t['score']];mass,k=combine(a,b)
  out.append(dict(fire=int(mass[0]>threshold),masses=mass,K=k,iou=float(s[i,j]),selected=dict(rgb_index=int(i),thermal_index=int(j),rgb=r,thermal=t,bpa_rgb=a,bpa_thermal=b,thermal_reliability=reliability)))
 return max(out,key=lambda z:z['masses'][0],default=empty)

def background_mask(shape,boxes):
 h,w=shape[:2];m=np.full((h,w),255,np.uint8);m[:24]=0;m[-12:]=0;m[:,:12]=0;m[:,-12:]=0
 for b in boxes:
  x,y,X,Y=np.array(b['bbox'],int);x=max(0,x-8);y=max(0,y-8);X=min(w,X+8);Y=min(h,Y+8);m[y:Y,x:X]=0
 return m

def overlay_mask(im):
 if im.ndim==2:return np.full(im.shape,255,np.uint8)
 hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);h,s,v=cv2.split(hsv)
 bad=((((h<10)|(h>170))|((h>20)&(h<40)))&(s>140)&(v>140)).astype(np.uint8)*255
 return 255-cv2.dilate(bad,np.ones((7,7),np.uint8))

def feature_proposal(rgb,thermal,rb,tb,variant):
 """Mutual Lowe-filtered background features; deterministic holdout, no labels."""
 def gray(x):return cv2.cvtColor(x,cv2.COLOR_BGR2GRAY) if x.ndim==3 else x
 r=gray(rgb);t=gray(thermal)
 if variant.startswith('gradient'):
  def grad(x):
   a=x.astype(np.float32);z=cv2.magnitude(cv2.Sobel(a,cv2.CV_32F,1,0),cv2.Sobel(a,cv2.CV_32F,0,1));return cv2.normalize(z,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
  r=grad(r);t=grad(t)
 orb=variant.endswith('orb');det=cv2.ORB_create(nfeatures=2000,edgeThreshold=12) if orb else cv2.SIFT_create(nfeatures=2000,contrastThreshold=.01,edgeThreshold=12)
 kr,dr=det.detectAndCompute(r,background_mask(r.shape,rb)&overlay_mask(rgb));kt,dt=det.detectAndCompute(t,background_mask(t.shape,tb)&overlay_mask(thermal))
 o=dict(variant=variant,accepted=False,rgb_features=len(kr),thermal_features=len(kt),mutual_matches=0,reason='insufficient_features')
 if dr is None or dt is None or min(len(dr),len(dt))<2:return o
 bf=cv2.BFMatcher(cv2.NORM_HAMMING if orb else cv2.NORM_L2)
 def ratio(a,b):return {v[0].queryIdx:v[0].trainIdx for v in bf.knnMatch(a,b,k=2) if len(v)==2 and v[0].distance<.75*v[1].distance}
 rt=ratio(dr,dt);tr=ratio(dt,dr);matches=[(i,j) for i,j in rt.items() if tr.get(j)==i];o['mutual_matches']=len(matches);o['reason']='insufficient_mutual_matches'
 if len(matches)<12:return o
 dst=np.array([kr[i].pt for i,j in matches]);src=np.array([kt[j].pt for i,j in matches]);order=np.lexsort((src[:,0],src[:,1]));val=order[::3];train=np.array([i for i in order if i not in set(val)])
 cv2.setRNGSeed(741);A,mask=cv2.estimateAffine2D(src[train],dst[train],method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=5000,confidence=.999,refineIters=10)
 if A is None:o['reason']='ransac_failed';return o
 inl=mask.ravel().astype(bool);err=np.linalg.norm(project(src[val],A)-dst[val],axis=1);sv=np.linalg.svd(A[:,:2],compute_uv=False)
 srcgood=src[train][inl];dstgood=dst[train][inl]
 def hull_fraction(p,shape):return float(cv2.contourArea(cv2.convexHull(np.float32(p)))/(shape[0]*shape[1])) if len(p)>=3 else 0.
 hs=hull_fraction(srcgood,t.shape);hd=hull_fraction(dstgood,r.shape)
 cells=len(set(tuple(v) for v in np.floor(dstgood/np.array([r.shape[1]/3,r.shape[0]/3])).astype(int)))
 accepted=sum(inl)>=8 and len(val)>=4 and np.quantile(err,.9)<=5 and np.median(err)<=3 and hs>=.1 and hd>=.1 and cells>=4 and np.linalg.det(A[:,:2])>0 and min(sv)>.3 and max(sv)<3
 o.update(accepted=bool(accepted),reason='holdout_pass_candidate_requires_independent_landmarks' if accepted else 'holdout_or_field_coverage_failed',matrix=A.tolist(),train_inliers=int(sum(inl)),validation_errors=err.tolist(),thermal_hull=hs,rgb_hull=hd,spatial_cells=cells,matched_src=src.tolist(),matched_dst=dst.tolist(),holdout_indices=val.tolist(),independently_validated=False)
 return o

def bootstrap_affines(src,dst,nominal,n=32):
 """Seeded resampling, shrink to nominal, reject ill-conditioning and poor LOO."""
 src=np.asarray(src,float);dst=np.asarray(dst,float);nominal=np.asarray(nominal,float);X=np.c_[src,np.ones(len(src))];loo=[]
 for i in range(len(src)):
  ix=np.arange(len(src))!=i;B=np.linalg.lstsq(X[ix],dst[ix],rcond=None)[0].T;loo.append(float(np.linalg.norm(project(src[i:i+1],B)[0]-dst[i])))
 rng=np.random.default_rng(741);mats=[];trials=0;scale=max(1.,float(np.max(np.linalg.norm(dst-project(src,nominal),axis=1))));ridge=np.diag([1e-3,1e-3,1e-6])
 while len(mats)<n and trials<4096:
  trials+=1;ix=rng.integers(0,len(src),len(src))
  if len(set(ix))<4 or np.linalg.cond(X[ix])>1e6:continue
  B=np.linalg.solve(X[ix].T@X[ix]+ridge,X[ix].T@dst[ix]+ridge@nominal.T).T;B=.5*nominal+.5*B;sv=np.linalg.svd(B[:,:2],compute_uv=False)
  if min(sv)<.3 or max(sv)>1.6 or np.linalg.det(B[:,:2])<=0:continue
  if max(np.linalg.norm(project(src,B)-dst,axis=1))>max(12.,2*scale):continue
  le=[]
  for i in range(len(src)):
   mask=np.arange(len(src))!=i;Z=X[mask];C=np.linalg.solve(Z.T@Z+ridge,Z.T@dst[mask]+ridge@B.T).T;C=.5*B+.5*C;le.append(np.linalg.norm(project(src[i:i+1],C)[0]-dst[i]))
  if max(le)>15:continue
  mats.append(B)
 if not mats:mats=[nominal.copy() for _ in range(n)]
 return mats,dict(seed=741,requested=n,accepted=len(mats),attempts=trials,loo_errors=loo,loo_rmse=float(np.sqrt(np.mean(np.square(loo)))),matrix_distribution=[a.tolist() for a in mats],validation='seven-point self/LOO only; NOT out-of-hull systematic scene error validation',shrinkage=.5,loo_max_limit=15)
