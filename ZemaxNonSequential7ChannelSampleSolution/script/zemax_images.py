"""Extract Zemax image regions, apply explicit orientation, and supply nominal search centers.
Actual content-based registration is implemented separately in registration.py.
"""
from pathlib import Path
import argparse, json, math
import numpy as np
from PIL import Image, ImageDraw, ImageOps

CHANNELS = 'ABCDEFG'

def centers(separation=3.0):
    q = math.sqrt(3) * separation / 2
    return dict(zip(CHANNELS, [(0,separation),(q,separation/2),(q,-separation/2),
        (0,-separation),(-q,-separation/2),(-q,separation/2),(0,0)]))

def runs(a):
    edges = np.diff(np.r_[False, a, False].astype(np.int8))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))

def detect_plot(image):
    """Find a large dark square surrounded by the white Zemax export frame.
    Coordinates are left, top, right, bottom; right/bottom are exclusive.
    Reject ambiguous/unsupported layouts rather than using a hard-coded crop.
    """
    a = np.asarray(image.convert('RGB'))
    h,w = a.shape[:2]
    dark = a.max(axis=2) < 160
    candidates = []
    for y0,y1 in runs(dark.mean(axis=1) > 0.25):
        if y1-y0 < min(w,h)*0.20: continue
        for x0,x1 in runs(dark[y0:y1].mean(axis=0) > 0.80):
            bw,bh=x1-x0,y1-y0
            if min(bw,bh) < min(w,h)*0.20 or not 0.9 < bw/bh < 1.1: continue
            if x0 < 4 or x1 > w-4 or y0 < 4 or y1 > h-4: continue
            # Exclude plot frames and footer text: the square needs white margins.
            sides=[a[y0:y1,max(0,x0-4):x0-2],a[y0:y1,x1+2:x1+4],a[max(0,y0-4):y0-2,x0:x1],a[y1+2:y1+4,x0:x1]]
            if min(float((s.min(axis=2)>210).mean()) for s in sides) < .80: continue
            candidates.append((bw*bh,(int(x0),int(y0),int(x1),int(y1))))
    if len(candidates)!=1:
        raise ValueError(f'Expected one dark square with white margins; found {len(candidates)}. '
            'For an unframed output bitmap use --input-kind bitmap. Otherwise inspect export format.')
    return candidates[0][1]

def orient(im, name):
    ops={'identity':lambda x:x,'rot180':lambda x:x.transpose(Image.Transpose.ROTATE_180),
         'flip-x':ImageOps.mirror,'flip-y':ImageOps.flip,
         'rot90':lambda x:x.transpose(Image.Transpose.ROTATE_90),
         'rot270':lambda x:x.transpose(Image.Transpose.ROTATE_270)}
    if name not in ops: raise ValueError(f'Unknown orientation: {name}')
    return ops[name](im)
