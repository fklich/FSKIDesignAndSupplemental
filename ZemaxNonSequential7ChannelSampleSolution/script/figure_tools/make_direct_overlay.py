from pathlib import Path
import json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont

root=Path(__file__).resolve().parents[1]
import os
results=Path(os.environ.get('TELESCOPE_RESULTS_DIR',str(root/'registered_output')))
figures=Path(os.environ.get('TELESCOPE_FIGURE_DIR',str(root/'comparison_figures')))
project=Path(os.environ.get('TELESCOPE_PROJECT_DIR',str(root.parent)))
out=figures;out.mkdir(parents=True,exist_ok=True)
meta=json.loads((results/'manifest.json').read_text(encoding='utf-8'))
registered=Image.open(results/'mosaic.png').convert('RGB')
width,height=registered.size;size=meta['registration_size'];ppd=size/3.5
G=np.array(meta['channels']['G']['image_to_mosaic'])
origin=(G@np.array([(size-1)/2,(size-1)/2,1]))[:2]
canvas=Image.new('RGB',(width,height),'black');placements={}
yy,xx=np.mgrid[:size,:size]
mask=np.uint8(np.hypot(xx-(size-1)/2,yy-(size-1)/2)<size*.494)*255
mask[:3]=0;mask[-3:]=0;mask[:,:3]=0;mask[:,-3:]=0
mask=Image.fromarray(mask)
for name in 'ABCDEFG':
    im=Image.open(results/f'crops/Channel_{name}.png').convert('RGB').resize((size,size))
    dx,dy=meta['channels'][name]['nominal_center_deg']
    cx,cy=origin+np.array([dx*ppd,-dy*ppd])
    left=round(cx-(size-1)/2);top=round(cy-(size-1)/2)
    canvas.paste(im,(left,top),mask)
    placements[name]={'left':left,'top':top,'center':[float(cx),float(cy)]}
canvas.save(out/'Direct_Overlay_Clean.png')
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',27)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
label=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',32)
annotated=canvas.copy();draw=ImageDraw.Draw(annotated)
for name,loc in placements.items():
    cx,cy=loc['center'];r=size*.494
    color='#e6b956' if name=='G' else '#79a9bf'
    draw.ellipse((round(cx-r),round(cy-r),round(cx+r),round(cy+r)),outline=color,width=2)
    x=round(cx-r*.58);y=round(cy-r*.58)
    draw.rounded_rectangle((x-7,y-4,x+31,y+36),radius=4,fill='#080c10')
    draw.text((x,y),name,font=label,fill=color)
header=110
schematic=Image.new('RGB',(width,height+header),'#080b10');schematic.paste(annotated,(0,header))
draw=ImageDraw.Draw(schematic)
draw.text((30,18),'Direct overlay of seven channel images',font=font,fill='white')
draw.text((30,60),'Nominal positions   |   No image alignment or seam blending',font=small,fill='#b5c0cc')
schematic.save(out/'Direct_Overlay_Schematic.png',dpi=(200,200))
# Equal size and display scale: both panels share the same reference origin and canvas.
thumbw=1000;thumbh=round(height*thumbw/width);gap=30;top=100;bottom=62
comparison=Image.new('RGB',(thumbw*2+gap,top+thumbh+bottom),'#080b10');draw=ImageDraw.Draw(comparison)
comparison.paste(annotated.resize((thumbw,thumbh),Image.Resampling.LANCZOS),(0,top))
comparison.paste(registered.resize((thumbw,thumbh),Image.Resampling.LANCZOS),(thumbw+gap,top))
draw.text((24,18),'Direct overlay',font=font,fill='white')
draw.text((24,59),'Nominal positions; channel boundaries shown',font=small,fill='#b5c0cc')
draw.text((thumbw+gap+24,18),'Aligned mosaic',font=font,fill='white')
draw.text((thumbw+gap+24,59),'Overlap alignment and edge blending',font=small,fill='#b5c0cc')
draw.text((24,top+thumbh+18),'Same seven channel images and display scale. No artificial shifts were added.',font=small,fill='#b5c0cc')
comparison.save(out/'Direct_Overlay_vs_Registered_Mosaic.png',dpi=(200,200))
(out/'overlay_notes.json').write_text(json.dumps({'method':'Circular hard masks with opaque pasting at nominal positions; no image registration or feather blending','paste_order':list('ABCDEFG'),'same_dataset_as':'registered_output/manifest.json','artificial_perturbations':False,'placements':placements},indent=2),encoding='utf-8')
print('Saved',out)
print('Schematic',schematic.size,'Comparison',comparison.size)
