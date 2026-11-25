"""Display-only normalized Gaussian smoothing of a signed scalar field.

This is NOT point-density estimation. No training or query code imports it.
Nearby cell centers contribute only if the straight segment remains walkable.
"""
import numpy as np
import shapely
from scipy.ndimage import map_coordinates


def smooth_display(values, x_centers, z_centers, walkable, sigma_m=.6, cell_m=.4):
    if not np.isfinite(sigma_m) or sigma_m <= 0: raise ValueError('sigma_m must be positive')
    raw=np.asarray(values,float)
    if raw.shape != (len(z_centers),len(x_centers)): raise ValueError('Grid dimensions disagree')
    xx,zz=np.meshgrid(x_centers,z_centers)
    valid=np.isfinite(raw) & shapely.intersects_xy(walkable,xx,zz)
    numerator=np.zeros_like(raw); denominator=np.zeros_like(raw)
    # Finite Gaussian support, with boundary normalization to avoid attenuating edges.
    radius=int(np.ceil(3*sigma_m/cell_m))
    nz,nx=raw.shape
    for dz in range(-min(radius,nz-1),min(radius,nz-1)+1):
        for dx in range(-min(radius,nx-1),min(radius,nx-1)+1):
            weight=np.exp(-.5*((dx*cell_m)**2+(dz*cell_m)**2)/sigma_m**2)
            rz,rx=np.nonzero(valid)
            sz,sx=rz+dz,rx+dx
            inside=(sz>=0)&(sz<nz)&(sx>=0)&(sx<nx)
            rz,rx,sz,sx=rz[inside],rx[inside],sz[inside],sx[inside]
            ok=valid[sz,sx]
            rz,rx,sz,sx=rz[ok],rx[ok],sz[ok],sx[ok]
            if not len(rz): continue
            if dx or dz:
                coords=np.stack([np.column_stack([xx[rz,rx],zz[rz,rx]]),
                                 np.column_stack([xx[sz,sx],zz[sz,sx]])],axis=1)
                visible=shapely.covers(walkable,shapely.linestrings(coords))
                rz,rx,sz,sx=rz[visible],rx[visible],sz[visible],sx[visible]
            numerator[rz,rx]+=weight*raw[sz,sx]
            denominator[rz,rx]+=weight
    result=np.full_like(raw,np.nan)
    np.divide(numerator,denominator,out=result,where=valid&(denominator>0))
    return result


def continuous_display(values, xe, ze, walkable, upsample=4):
    """Bilinear display resampling only; mask missing cells and exact geometry."""
    if not isinstance(upsample,int) or not 1<=upsample<=16: raise ValueError('upsample must be 1..16')
    nz,nx=values.shape
    fine_x=np.linspace(xe[0],xe[-1],nx*upsample+1)
    fine_z=np.linspace(ze[0],ze[-1],nz*upsample+1)
    xx,zz=np.meshgrid((fine_x[:-1]+fine_x[1:])/2,(fine_z[:-1]+fine_z[1:])/2)
    iy=(zz-ze[0])/(ze[1]-ze[0])-.5
    ix=(xx-xe[0])/(xe[1]-xe[0])-.5
    valid=np.isfinite(values)
    visible=map_coordinates(valid.astype(float),[iy,ix],order=0,mode='nearest')>.5
    visible &= shapely.intersects_xy(walkable,xx,zz)
    # Normalized bilinear weights, excluding corners across walls as well.
    # Otherwise upsampling could reintroduce mixing after wall-aware smoothing.
    py=np.clip(iy,0,nz-1);px=np.clip(ix,0,nx-1)
    lo_y=np.floor(py).astype(int);lo_x=np.floor(px).astype(int)
    frac_y=py-lo_y;frac_x=px-lo_x
    num=np.zeros(xx.shape);den=np.zeros(xx.shape)
    for dy in [0,1]:
        for dx in [0,1]:
            sy=np.minimum(lo_y+dy,nz-1);sx=np.minimum(lo_x+dx,nx-1)
            weight=(frac_y if dy else 1-frac_y)*(frac_x if dx else 1-frac_x)
            rows,cols=np.nonzero(visible & valid[sy,sx] & (weight>0))
            if not len(rows):continue
            cx=(xe[sx[rows,cols]]+xe[sx[rows,cols]+1])/2
            cz=(ze[sy[rows,cols]]+ze[sy[rows,cols]+1])/2
            points=np.column_stack([xx[rows,cols],zz[rows,cols]])
            centers=np.column_stack([cx,cz])
            same=np.all(points==centers,axis=1)
            good=same | shapely.covers(walkable,shapely.linestrings(np.stack([points,centers],axis=1)))
            rows,cols=rows[good],cols[good]
            num[rows,cols]+=weight[rows,cols]*values[sy[rows,cols],sx[rows,cols]]
            den[rows,cols]+=weight[rows,cols]
    result=np.full(xx.shape,np.nan)
    np.divide(num,den,out=result,where=visible&(den>0))
    return result,fine_x,fine_z
