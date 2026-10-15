"""Standalone no-UV runtime: RGB native320 + thermal full320/full640/fixed2x2 tiles320."""
from pathlib import Path
import argparse,json,hashlib,time
import cv2,numpy as np,torch
from ultralytics import YOLO
from adaptive_core import tiles,map_box,piou,corners,anchor_fuse
ROOT=Path(__file__).resolve().parent

def infer(rgb_path,thermal_path):
 cfg=json.loads((ROOT/'deployment.json').read_text(encoding='utf-8'));start=time.perf_counter();receipt=[];boxes={};shapes=[]
 for modality,path in [('rgb',rgb_path),('thermal',thermal_path)]:
  checkpoint=Path(cfg['models'][modality]['source']);assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==cfg['models'][modality]['sha256'];model=YOLO(str(checkpoint));im=cv2.imread(str(path));assert im is not None;h,w=im.shape[:2];bs=[]
  jobs=[('full320',[0,0,w,h],320)]
  if modality=='thermal':jobs += [('tile'+str(i)+'_320',q,320) for i,q in enumerate(tiles(im.shape))]+[('full640',[0,0,w,h],640)]
  for view,crop,size in jobs:
   x,y,X,Y=crop;pixels=im[y:Y,x:X].copy();hook=model.model.register_forward_pre_hook(lambda mm,args:shapes.append(list(args[0].shape)))
   z=model.predict(pixels,imgsz=size,rect=False,conf=.005,half=False,device=0,classes=[0],max_det=300,verbose=False,save=False)[0];hook.remove();np.testing.assert_array_equal(z.orig_img,pixels)
   for xx,yy,XX,YY,score,cls in z.boxes.data.cpu().tolist():bs.append(dict(bbox=map_box([xx,yy,XX,YY],crop),score=score,class_id=int(cls),view=view,crop=crop,input_size=size,actual_detector=True))
   receipt.append(dict(modality=modality,view=view,crop=crop,input_size=size))
  out=[]
  for b in sorted(bs,key=lambda b:-b['score']):
   if all(piou(corners(b['bbox']),corners(v['bbox']))<.8 for v in out):out.append(b)
  boxes[modality]=bs if modality=='rgb' else out
 result=anchor_fuse(boxes['rgb'],boxes['thermal'],np.array(cfg['affine']),**cfg['fusion'])
 return dict(result=result,actual_detector_views=len(receipt),views=receipt,input_shapes=shapes,UV_runtime=False,geometry_validated=False,warning=cfg['warning'],image_sha256={m:hashlib.sha256(Path(p).read_bytes()).hexdigest() for m,p in [('rgb',rgb_path),('thermal',thermal_path)]},gpu=torch.cuda.get_device_name(0),seconds=time.perf_counter()-start)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--rgb',required=True);p.add_argument('--thermal',required=True);p.add_argument('--output',required=True);args=p.parse_args();result=infer(args.rgb,args.thermal);Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
