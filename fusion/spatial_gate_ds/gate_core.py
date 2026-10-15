"""Explicit user-authorized consensus gate; not a claim of physical absence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'spatial_reliability_ds'))
import spatial_core as s
import numpy as np

def frame(rgb,thermal,matrix,uv,tau_l=.1,tau_h=.3,ignorance=.2,uv_reliability=.9):
    if uv not in (0,1): raise ValueError('UV must be binary')
    if not 0<=tau_l<tau_h<=1: raise ValueError('invalid thresholds')
    # Tiny positive matching floor excludes true zero-overlap edges; no hard gate here.
    pairs=s.match(rgb,thermal,matrix,gate=1e-12)
    if not pairs: pairs=[dict(rgb_index=None,thermal_index=None,iou=0.,status='missing_both')]
    um=np.array([uv_reliability*uv,uv_reliability*(1-uv),1-uv_reliability])
    details=[]
    for cand in pairs:
        ri,ti=cand['rgb_index'],cand['thermal_index']
        rc=rgb[ri]['score'] if ri is not None else None
        tc=thermal[ti]['score'] if ti is not None else None
        rm,tm=s.base(rc,ignorance),s.base(tc,ignorance)
        ordinary,k,_=s.ds(rm,tm)
        g=float(np.clip((cand['iou']-tau_l)/(tau_h-tau_l),0,1)) if cand['status']=='matched' else 0.
        rd,td=s.discount(rm,g),s.discount(tm,g)
        visual,ks,deg=s.ds(rd,td)
        visual_uv,ku,du=s.ds(visual,um)
        vd=decide(cand['iou'],cand['status']=='matched',visual,tau_l,tau_h)
        ud=decide(cand['iou'],cand['status']=='matched',visual_uv,tau_l,tau_h)
        details.append(dict(**cand,rgb_confidence=rc,thermal_confidence=tc,g=g,reliability=g,ordinary_mass=ordinary.tolist(),rgb_base=rm.tolist(),thermal_base=tm.tolist(),rgb_discounted=rd.tolist(),thermal_discounted=td.tolist(),visual_mass=visual.tolist(),visual_uv_mass=visual_uv.tolist(),ordinary_conflict=k,visual_conflict=ks,uv_conflict=ku,degenerate=deg or du,state=vd['state'],gate_pass=vd['gate_pass'],visual_betp=vd['betp'],visual_uv_betp=ud['betp'],final_visual=vd['label'],final_visual_uv=ud['label']))
    # Spatial/visual-only selection, never UV/reference labels. No correlated multi-box DS sum.
    chosen=max(range(len(details)),key=lambda j:(details[j]['iou'],details[j]['visual_betp'],min([x for x in [details[j]['rgb_confidence'],details[j]['thermal_confidence']] if x is not None],default=0),-j))
    result=dict(details[chosen]); result.update(selected_candidate=chosen,candidates=details,instant_uv_mass=um.tolist())
    return result

def decide(iou,matched,mass,tau_l=.1,tau_h=.3):
    if not 0<=tau_l<tau_h<=1 or not 0<=iou<=1: raise ValueError('invalid thresholds/IoU')
    state='NoFire' if not matched or iou<=tau_l else ('Transition' if iou<tau_h else 'FireCandidate')
    betp=float(mass[0]+.5*mass[2])
    return dict(label=int(state=='FireCandidate' and betp>.5),state=state,gate_pass=state=='FireCandidate',betp=betp)
