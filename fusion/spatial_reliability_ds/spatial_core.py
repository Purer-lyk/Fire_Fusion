"""Fixed spatial reliability discount and classic Dempster combination.
Mass ordering F,N,U. No labels or temporal state enter spatial evidence.
"""
import math
import numpy as np

def base(confidence, ignorance=.2):
    if confidence is None: return np.array([0.,0.,1.])
    if not 0<=confidence<=1 or not 0<=ignorance<=1: raise ValueError('probability range')
    return np.array([(1-ignorance)*confidence,(1-ignorance)*(1-confidence),ignorance])

def discount(mass,reliability):
    m=np.asarray(mass,float)
    if not 0<=reliability<=1 or not np.isfinite(m).all() or np.min(m)<0 or not np.isclose(m.sum(),1): raise ValueError('invalid BPA/reliability')
    return np.array([reliability*m[0],reliability*m[1],1-reliability+reliability*m[2]])

def ds(a,b):
    af,an,au=a; bf,bn,bu=b
    conflict=float(af*bn+an*bf)
    support=np.array([af*bf+af*bu+au*bf,an*bn+an*bu+au*bn,au*bu],float)
    if support.sum()<=1e-12: return np.array([0.,0.,1.]),conflict,True
    return support/support.sum(),conflict,False

def polygon(box):
    x1,y1,x2,y2=map(float,box)
    if not np.isfinite(box).all() or x2<=x1 or y2<=y1: raise ValueError('invalid rectangle')
    return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],np.float32)

def affine(box,matrix):
    a=np.asarray(matrix,float)
    if a.shape!=(2,3) or not np.isfinite(a).all() or abs(np.linalg.det(a[:,:2]))<1e-12: raise ValueError('invalid affine')
    return (polygon(box)@a[:,:2].T+a[:,2]).astype(np.float32)

def iou(a,b):
    import cv2
    aa=abs(cv2.contourArea(a)); bb=abs(cv2.contourArea(b))
    intersection,_=cv2.intersectConvexConvex(a,b)
    union=aa+bb-intersection
    return float(np.clip(intersection/union if union>0 else 0,0,1))

def match(rgb,thermal,matrix,gate=.01):
    from scipy.optimize import linear_sum_assignment
    nr,nt=len(rgb),len(thermal)
    if nr+nt==0: return []
    scores=np.zeros((nr+nt,nr+nt),float)
    for r,rd in enumerate(rgb):
        for t,td in enumerate(thermal):
            value=iou(polygon(rd['bbox_xyxy']),affine(td['bbox_xyxy'],matrix))
            scores[r,t]=value if value>=gate else 0
    rs,ts=linear_sum_assignment(-scores)
    pairs=[dict(rgb_index=int(r),thermal_index=int(t),iou=float(scores[r,t]),status='matched') for r,t in zip(rs,ts) if r<nr and t<nt and scores[r,t]>=gate]
    pairs.sort(key=lambda x:(x['rgb_index'],x['thermal_index']))
    usedr={x['rgb_index'] for x in pairs}; usedt={x['thermal_index'] for x in pairs}
    pairs += [dict(rgb_index=r,thermal_index=None,iou=0.,status='unmatched_rgb') for r in range(nr) if r not in usedr]
    pairs += [dict(rgb_index=None,thermal_index=t,iou=0.,status='unmatched_thermal') for t in range(nt) if t not in usedt]
    assert len({x['rgb_index'] for x in pairs if x['rgb_index'] is not None})==nr
    assert len({x['thermal_index'] for x in pairs if x['thermal_index'] is not None})==nt
    return pairs

def binary(m): return int(float(m[0]+.5*m[2])>.5+1e-12)
def decision(m,threshold=.8): return 'fire' if m[0]>=threshold else ('nofire' if m[1]>=threshold else 'uncertain')
