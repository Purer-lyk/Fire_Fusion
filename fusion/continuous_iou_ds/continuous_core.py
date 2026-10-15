"""Positive-only continuous-IoU simple-support DS. No UV, labels or identities as inputs."""
def gain(s,sL=.1,sH=.3):
 if not sH>sL:raise ValueError('sH must exceed sL')
 return max(0.,min(1.,(s-sL)/(sH-sL)))

def mass(cR,cT,s,continuous=True):
 rt=.1*(gain(s) if continuous else 1.)
 u=(1-cR)*(1-rt*cT)
 return [1-u,0.,u]

def select(pairs,continuous=True,gate=.001):
 out=[]
 for p in pairs:
  if p['s']<gate:continue
  m=mass(p['cR'],p['cT'],p['s'],continuous)
  out.append(dict(p,g=gain(p['s']),actual_rT=.1*(gain(p['s']) if continuous else 1.),mF=m[0],mN=m[1],mU=m[2],K=0.,fire=int(m[0]>.5)))
 return max(out,key=lambda x:x['mF'],default=dict(rgb_index=None,thermal_index=None,cR=0.,cT=0.,s=0.,g=0.,actual_rT=0.,mF=0.,mN=0.,mU=1.,K=0.,fire=0))
