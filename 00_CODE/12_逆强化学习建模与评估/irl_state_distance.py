"""Batch-independent nearest state assignment for quantized VR features."""
import numpy as np

def stable_nearest(x,centers):
    x=np.asarray(x,np.float64);centers=np.asarray(centers,np.float64)
    labels=[];distance=[]
    # Accumulate actual differences, avoiding cancellation in norm^2 - 2 dot.
    # Identical ties always select the first center regardless of BLAS/batch size.
    for start in range(0,len(x),4096):
        part=x[start:start+4096];d2=np.zeros((len(part),len(centers)),np.float64)
        for j in range(part.shape[1]):d2+=(part[:,j,None]-centers[None,:,j])**2
        k=d2.argmin(1);labels.append(k);distance.append(np.sqrt(d2[np.arange(len(k)),k]))
    return np.concatenate(labels) if labels else np.empty(0,int),np.concatenate(distance) if distance else np.empty(0,float)
