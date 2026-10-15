def metrics(truth,pred):
 if not truth or len(truth)!=len(pred) or any(x not in (0,1) for x in list(truth)+list(pred)):
  raise ValueError('Nonempty aligned binary inputs required')
 tp=sum(y==1 and p==1 for y,p in zip(truth,pred)); fp=sum(y==0 and p==1 for y,p in zip(truth,pred)); fn=sum(y==1 and p==0 for y,p in zip(truth,pred)); tn=sum(y==0 and p==0 for y,p in zip(truth,pred))
 div=lambda a,b:a/b if b else None
 return dict(N=len(truth),TP=tp,FP=fp,FN=fn,TN=tn,FPR=div(fp,fp+tn),FNR=div(fn,fn+tp),precision=div(tp,tp+fp),recall=div(tp,tp+fn),F1=div(2*tp,2*tp+fp+fn))
def safe_records(records,testhashes):
 return [r for r in records if r['valid'] and r['sha256'] not in testhashes]
def target(m,thermal,rgb):
 if not m['N']==thermal['N']==rgb['N']:raise ValueError('Different denominator')
 return bool(m['FPR']<=thermal['FPR'] and m['FNR']<thermal['FNR'] and m['FPR']<rgb['FPR'])
