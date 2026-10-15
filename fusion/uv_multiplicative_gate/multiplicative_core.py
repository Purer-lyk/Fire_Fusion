"""Literal user formula; UV is not a third D-S source."""
import math

def apply(mass,uv,lam=.5,tau=.5,hard_gate=False,matched=False,iou=0.,tau_h=.3):
    if uv not in (0,1) or not 0<=lam<=1 or not 0<=tau<=1 or not 0<=iou<=1:
        raise ValueError('invalid binary UV, lambda, threshold or IoU')
    f,n,u=map(float,mass)
    if not all(math.isfinite(x) and x>=0 for x in (f,n,u)) or abs(f+n+u-1)>1e-9:
        raise ValueError('normalized nonnegative visual-only mass required')
    q=lam+(1-lam)*uv
    after=(q*f,n,u+(1-q)*f)
    eligible=not hard_gate or (matched and iou>=tau_h)
    base=int(eligible and f>tau)
    fire=int(eligible and after[0]>tau)
    reason=('spatial_gate_blocked' if not eligible else 'fire' if fire else
            'UV_suppressed' if base else 'visual_mF_not_strictly_above_threshold')
    return dict(q=q,mass=after,base_fire=base,fire=fire,eligible=int(eligible),reason=reason)
