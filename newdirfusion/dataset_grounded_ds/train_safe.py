from pathlib import Path
import json,hashlib,time,datetime,torch
from ultralytics import YOLO
OUT=Path(__file__).resolve().parent
if __name__=='__main__':
 assert torch.cuda.is_available()
 receipt=json.loads((OUT/'safe_export_receipt.json').read_text());assert receipt['test_overlap_after_export']==0 and receipt['train_val_exact_overlap']==0
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 for r in receipt['manifest']:
  assert sha(r['export_image'])==r['sha256'] and sha(r['export_label'])==r['label_sha256']
 model=YOLO(str(OUT/'generic_coco_yolo26n.pt'));assert len(model.names)==80 and model.ckpt['train_args']['data'].endswith('coco.yaml')
 plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'epochs_each':30,'imgsz':320,'batch':32,'workers':0,'seed':42,'amp':False,'device':torch.cuda.get_device_name(0),'initialization_sha256':sha(OUT/'generic_coco_yolo26n.pt'),'init':'generic COCO only, no leaked fire checkpoint fine-tune','export_sha256':sha(OUT/'safe_export_receipt.json'),'test_in_network_training':False,'selection':'Validation fitness best.pt only. No network training on 1528; fixed prior DS cfg unchanged. Planned fresh same-model .5 baselines and .01 candidate forward. All 1528 already seen; not independent fusion validation.'}
 (OUT/'safe_training_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
 status={};t=time.time()
 for mode in ['fireOrigin','thermal']:
  print('START_MODALITY',mode,flush=True)
  model=YOLO(str(OUT/'generic_coco_yolo26n.pt'))
  kwargs={'data':str(OUT/'safe_datasets'/mode/'data.yaml'),'epochs':30,'imgsz':320,'batch':32,'device':0,'workers':0,'seed':42,'amp':False,'project':str(OUT/'safe_training'),'name':mode,'exist_ok':False,'cache':False,'plots':True,'patience':30,'resume':False,'save':True,'val':True,'close_mosaic':10,'deterministic':True,'optimizer':'AdamW','lr0':.001,'lrf':.01}
  if mode=='thermal':kwargs.update(hsv_h=0.,hsv_s=0.)
  started=time.time();result=model.train(**kwargs)
  save=Path(model.trainer.save_dir);best=save/'weights/best.pt';last=save/'weights/last.pt';assert best.is_file() and last.is_file()
  checkpoint=YOLO(str(best));assert len(checkpoint.names)==1
  status[mode]={'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seconds':time.time()-started,'best':str(best),'last':str(last),'best_sha256':sha(best),'val_metrics':result.results_dict,'best_saved_metrics':checkpoint.ckpt.get('train_metrics'),'train_args':checkpoint.ckpt['train_args'],'successful':True}
  (OUT/'safe_training_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
  print('COMPLETED_MODALITY',mode,flush=True)
 (OUT/'safe_training_complete.json').write_text(json.dumps({'status':status,'elapsed_seconds':time.time()-t,'success':True},indent=2),encoding='utf-8')
