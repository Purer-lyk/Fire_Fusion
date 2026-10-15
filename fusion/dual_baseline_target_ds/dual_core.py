import numpy as np
import cv2

def montage(images):
 h=max(x.shape[0] for x in images)
 return np.hstack([cv2.copyMakeBorder(x,0,h-x.shape[0],0,0,cv2.BORDER_CONSTANT,value=0) for x in images])

def project(points,A):
 return np.asarray(points)@np.asarray(A)[:,:2].T+np.asarray(A)[:,2]

def inverse(A):
 return cv2.invertAffineTransform(np.asarray(A,dtype=float))

def offset(box,crop):
 return [float(v+crop[k%2]) for k,v in enumerate(box)]
