"""Trace only the native blue/cyan floor arrows in the new high-smoke assets.

Background pixels are never modified. Connected colour silhouettes are traced
in native pixel coordinates; the Full mini-map pointer is outside the floor ROI.
Open silhouettes at the bottom image border remain open.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_fill_holes, find_objects, binary_closing
from shapely.geometry import LineString

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'GAZE_TOPO/Results_GazePlaneHeatmap'
SPECS={'ArrowRibbon-HighSmoke':('Arrow-High.png',(2350,1680,3700,4000)),
       'Full-HighSmoke':('Multimodal-High.png',(2250,2200,3700,4000))}


def trace(path,roi):
    image=np.asarray(Image.open(path).convert('RGB')).astype(np.int16)
    x0,y0,x1,y1=roi
    region=image[y0:y1,x0:x1]
    mask=(region[:,:,2]-region[:,:,0]>12)&(region[:,:,2]>=region[:,:,1]-5)
    components,count=label(mask,np.ones((3,3),int))
    paths=[]
    for index,slices in enumerate(find_objects(components),1):
        if slices is None:
            continue
        chosen=components[slices]==index
        area=int(chosen.sum())
        if area<6:
            continue
        # Pad internal components so contouring sees the complete silhouette;
        # do not pad outside the screenshot at its lower clipping boundary.
        ys,xs=slices
        top=max(ys.start-1,0); bottom=min(ys.stop+1,mask.shape[0])
        left=max(xs.start-1,0); right=min(xs.stop+1,mask.shape[1])
        component=components[top:bottom,left:right]==index
        saturation=region[top:bottom,left:right,2]-region[top:bottom,left:right,0]
        threshold=max(12,float(np.percentile(saturation[component],98))*.28)
        core=component&(saturation>threshold)
        # Preserve the image-border clipping; closing is only used away from it.
        solid=binary_fill_holes(core if bottom==mask.shape[0] else binary_closing(core))
        if min(solid.shape)<2:
            continue
        fig,ax=plt.subplots()
        contours=ax.contour(solid.astype(float),levels=[.5])
        segments=[p for p in contours.allsegs[0] if len(p)>=3]
        for points in sorted(segments,key=lambda p:np.linalg.norm(np.diff(p,axis=0),axis=1).sum(),reverse=True)[:1]:
            if len(points)<3:
                continue
            points=points+np.array([x0+left,y0+top])
            closed=bool(np.allclose(points[0],points[-1]))
            # Native contours retain detail while avoiding sub-pixel stair steps.
            points=np.asarray(LineString(points).simplify(max(1,min(5,np.sqrt(area)*.006))).coords)
            if len(points)<3:
                continue
            paths.append(dict(closed=closed,method='native_colour_silhouette',
                              area_px=area,points=np.round(points,2).tolist()))
        plt.close(fig)
    paths.sort(key=lambda p:np.mean(np.asarray(p['points'])[:,1]))
    if len(paths)<8:
        raise ValueError('Too few arrow components; review the ROI/colour threshold')
    return paths


def main():
    target=OUTPUT/'background_alignment/arrow_contours.json'
    stored=json.loads(target.read_text(encoding='utf-8'))
    for key,(name,roi) in SPECS.items():
        path=OUTPUT/'背景'/name
        paths=trace(path,roi)
        with Image.open(path) as im:
            size=list(im.size)
        stored['groups'][key]=dict(image_path=str(path.relative_to(OUTPUT)),
            image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            image_size_px=size,coordinate_origin='upper_left',
            trace_roi_px=list(roi),trace_rule='B-R > 12; B >= G-5; components >=6 px; core threshold max(12, P98(B-R)*0.28); holes filled; longest silhouette per component',
            paths=paths)
        print(key,len(paths),'paths',flush=True)
    for key,name in [('ArrowRibbon','Arrow-Low.png'),('Full','Multimodal-Low.png')]:
        path=OUTPUT/'背景'/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=stored['groups'][key]['image_sha256']:
            raise ValueError('Low-smoke asset changed unexpectedly')
        stored['groups'][key]['image_path']=str(path.relative_to(OUTPUT))
    target.write_text(json.dumps(stored,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
