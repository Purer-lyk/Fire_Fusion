"""Pure binary-frame BPA and original C++ fusion rules. No inference or truth creation."""
from dataclasses import dataclass
import math

EPS = 1e-6  # original denominator guard; now a flagged vacuous fallback

def probability(x, name='probability'):
    if not isinstance(x, (int,float)) or not math.isfinite(x) or not 0 <= x <= 1:
        raise ValueError(f'{name} must be finite in [0,1]')
    return float(x)

@dataclass(frozen=True)
class Mass:
    fire: float
    nofire: float
    uncertain: float

    def __post_init__(self):
        for x in self.values(): probability(x, 'mass')
        if not math.isclose(sum(self.values()),1,abs_tol=1e-9):
            raise ValueError('Mass must sum to one')

    def values(self):
        return self.fire, self.nofire, self.uncertain

    @classmethod
    def normalized(cls, fire, nofire, uncertain):
        values=(fire,nofire,uncertain)
        if any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError('Invalid mass weights')
        total=sum(values)
        if total <= 0: return cls(0,0,1)
        return cls(*(x/total for x in values))

@dataclass(frozen=True)
class FusionResult:
    mass: Mass
    conflict: float
    degenerate: bool = False
    reason: str = ''

def combine(a, b, method='ds', spatial_consistency=None):
    """Port ds_fusion.cpp: DS, PCR5, or fire-scaled spatial formula.

    Spatial input must be caller-supplied, never fabricated from a missing H.
    K reported by spatial is (1-s)*raw_K, exactly as in the original.
    """
    if not isinstance(a,Mass) or not isinstance(b,Mass): raise ValueError('Mass inputs required')
    raw_k=a.fire*b.nofire+a.nofire*b.fire
    f=a.fire*b.fire+a.fire*b.uncertain+a.uncertain*b.fire
    n=a.nofire*b.nofire+a.nofire*b.uncertain+a.uncertain*b.nofire
    u=a.uncertain*b.uncertain
    k=raw_k
    if method=='pcr':
        c1,c2=a.fire*b.nofire,a.nofire*b.fire
        d1,d2=a.fire+b.nofire,a.nofire+b.fire
        if d1>0:
            f+=c1*a.fire/d1; n+=c1*b.nofire/d1
        if d2>0:
            n+=c2*a.nofire/d2; f+=c2*b.fire/d2
    elif method in ('ds','spatial'):
        if method=='spatial':
            s=probability(spatial_consistency,'spatial_consistency')
            k=(1-s)*raw_k
            f*=s
        if 1-k < EPS or f+n+u <= 0:
            return FusionResult(Mass(0,0,1),k,True,'total_or_near_conflict_or_zero_support')
        denom=1-k
        f/=denom; n/=denom; u/=denom
    else: raise ValueError('Unknown fusion rule')
    return FusionResult(Mass.normalized(f,n,u),k)

def fuse3(a,b,c,method='ds',spatial_consistency=None):
    """Left-to-right order RGB, thermal, UV. Spatial s repeats as legacy fuse3.

    This legacy mode is NOT a claim that UV has geometry. Recommended pipeline
    uses spatial only on RGB/thermal and ordinary DS or PCR for scalar UV.
    """
    ab=combine(a,b,method,spatial_consistency)
    abc=combine(ab.mass,c,method,spatial_consistency)
    return FusionResult(abc.mass,ab.conflict,ab.degenerate or abc.degenerate,
                        ab.reason or abc.reason)

def uv_gate(vis_fire,uv_fire,lambda_factor=.2):
    """Original unused scalar gate. Not applied automatically to normalized mass."""
    v=probability(vis_fire); u=probability(uv_fire); l=probability(lambda_factor)
    return v*(l+(1-l)*u)

@dataclass(frozen=True)
class Sample:
    rgb: str
    thermal: str
    temperature: float
    uv: int

def parse_sample_line(line):
    cols=line.split()
    if len(cols)!=4: raise ValueError('Expected RGB thermal Tmax UV')
    if any(any(c in name for c in '/\\:') or name in ('.','..') for name in cols[:2]):
        raise ValueError('Expected plain filenames')
    t=float(cols[2]); uv=float(cols[3])
    if not math.isfinite(t) or uv not in (0,1): raise ValueError('Invalid temperature/UV')
    return Sample(cols[0],cols[1],t,int(uv))

@dataclass(frozen=True)
class SensorState:
    rgb_previous: tuple = (.5,.3,.2)
    thermal_previous: tuple = (.5,.3,.2)
    uv_previous: tuple = (.1,.7,.05)  # preserve original unnormalized startup
    rgb_history: tuple = ()
    thermal_history: tuple = ()
    uv_confidence: float = .001

@dataclass(frozen=True)
class SensorResult:
    rgb: Mass
    thermal: Mass
    uv: Mass
    state: SensorState

def sensor_step(state,row,rgb_confidence,thermal_confidence,reliability=.3,
                window=10,time_lambda=.7,t0=50,alpha=.01,beta=.1):
    """Pure state transition. Temperature=column 3; UV=column 4.

    Caller MUST reset state at independently established sequence boundaries.
    No model output or UV is promoted to ground truth.
    """
    conf=probability(rgb_confidence); tc=probability(thermal_confidence)
    r=probability(reliability); l=probability(time_lambda)
    probability(alpha,'alpha'); probability(beta,'beta')
    if not isinstance(window,int) or window<1: raise ValueError('Positive window required')
    if not math.isfinite(t0) or not math.isfinite(row.temperature) or row.uv not in (0,1):
        raise ValueError('Invalid source evidence')
    rh=(state.rgb_history+(conf,))[-window:]
    th=(state.thermal_history+(tc,))[-window:]
    def stability(history):
        avg=sum(history)/len(history)
        return math.exp(-math.sqrt(sum((x-avg)**2 for x in history)/len(history)))
    def smooth(previous,current):
        return Mass.normalized(*(l*p+(1-l)*q for p,q in zip(previous,current)))
    s=stability(rh)
    f=conf*s*r; n=(1-conf)*s
    rgb=smooth(state.rgb_previous,(f,n,max(0,1-f-n)))
    pc=-math.expm1(-10*tc)
    pt=-math.expm1(-.1*max(0,row.temperature-t0))  # clamp before exp avoids cold overflow
    s=stability(th)
    f=pc*pt*s; n=(1-tc)*(1-pt)*s
    thermal=smooth(state.thermal_previous,(f,n,max(0,1-f-n)))
    q=probability(state.uv_confidence)
    q=max(0,min(1,q+(alpha if row.uv else -beta)*q*(1-q)))
    uv=smooth(state.uv_previous,(q,.8*(1-q),.2*(1-q)))
    new=SensorState(rgb.values(),thermal.values(),uv.values(),rh,th,q)
    return SensorResult(rgb,thermal,uv,new)


def box_polygon(box):
    import numpy as np
    if len(box)!=4 or any(not math.isfinite(float(x)) for x in box):
        raise ValueError('Finite xyxy box required')
    x1,y1,x2,y2=map(float,box)
    if x2<=x1 or y2<=y1: raise ValueError('Positive box area required')
    return np.array(((x1,y1),(x2,y1),(x2,y2),(x1,y2)),dtype=np.float64)


def project_polygon(box,h):
    import numpy as np
    if h is None: raise ValueError('Explicit homography required')
    h=np.asarray(h,dtype=float)
    if h.shape!=(3,3) or not np.isfinite(h).all(): raise ValueError('Finite 3x3 H required')
    p=box_polygon(box)
    q=np.column_stack((p,np.ones(4)))@h.T
    z=q[:,2]
    if np.min(np.abs(z))<1e-9 or np.min(z)*np.max(z)<=0:
        raise ValueError('Homography horizon intersects box')
    return q[:,:2]/z[:,None]


def project_box(box,h):
    p=project_polygon(box,h)
    return tuple(map(float,(*p.min(axis=0),*p.max(axis=0))))


def polygon_iou(a,b):
    import cv2
    import numpy as np
    a=np.asarray(a,dtype=np.float32); b=np.asarray(b,dtype=np.float32)
    aa=abs(cv2.contourArea(a)); bb=abs(cv2.contourArea(b))
    if aa<=0 or bb<=0: return 0.0
    inter,_=cv2.intersectConvexConvex(a,b)
    return max(0.,min(1.,float(inter)/(aa+bb-float(inter))))


def box_iou(a,b):
    box_polygon(a); box_polygon(b)
    x=max(0,min(a[2],b[2])-max(a[0],b[0]))
    y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=x*y
    return area/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def fuse_rgb_thermal_uv(a,b,u,method='ds',spatial_consistency=None):
    ab=combine(a,b,method,spatial_consistency)
    abc=combine(ab.mass,u,'ds')
    return FusionResult(abc.mass,ab.conflict,ab.degenerate or abc.degenerate,ab.reason or abc.reason)


def decide(mass,threshold=.95):
    probability(threshold)
    if mass.fire>=threshold: return 'fire'
    if mass.nofire>=threshold: return 'nofire'
    return 'uncertain'
