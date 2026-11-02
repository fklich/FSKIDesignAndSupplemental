from pathlib import Path
import json,zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont
root=Path(__file__).resolve().parents[1]
import os
results=Path(os.environ.get('TELESCOPE_RESULTS_DIR',str(root/'registered_output')))
figures=Path(os.environ.get('TELESCOPE_FIGURE_DIR',str(root/'comparison_figures')))
project=Path(os.environ.get('TELESCOPE_PROJECT_DIR',str(root.parent)))
out=figures/'local_details'
info=json.loads((out/'candidate_regions.json').read_text());by_pair={r['a']+r['b']:r for r in info['regions']}
keys=['AB','DE','EG','FG'];regions=[by_pair[k] for k in keys]
original=Image.open(out/'Original_Reference_Aligned.png').convert('RGB')
overlap=Image.open(out/'Direct_Overlap_Equal_Weight.png').convert('RGB')
mosaic=Image.open(results/'mosaic.png').convert('RGB')
hard=Image.open(figures/'Direct_Overlay_Clean.png').convert('RGB')
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',29)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21)
rowfont=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',23)
side=384;roi=48;zoom=8;pad=28;gap=20;header=112;rowhead=44;rowgap=18;footer=84
w=2*pad+3*side+2*gap;h=header+4*(rowhead+side+rowgap)+footer

def stretch(im):
 a=np.asarray(im,dtype=float)/255
 return Image.fromarray(np.uint8(np.clip(np.sqrt(a)*255,0,255)))

def build(direct,direct_title,direct_sub,path):
 sheet=Image.new('RGB',(w,h),'white');draw=ImageDraw.Draw(sheet)
 titles=['Original sky reference',direct_title,'Mosaic']
 subs=['Corresponding input region',direct_sub,'Aligned and blended']
 for j in range(3):
  x=pad+j*(side+gap)
  draw.text((x,20),titles[j],font=font,fill='black')
  draw.text((x,62),subs[j],font=small,fill='#444444')
 for i,r in enumerate(regions):
  x,y=r['x'],r['y'];box=(x-roi//2,y-roi//2,x+roi//2,y+roi//2)
  top=header+i*(rowhead+side+rowgap)
  draw.text((pad,top+8),f'Region {i+1}   Channels {r["a"]} and {r["b"]}',font=rowfont,fill='black')
  for j,im in enumerate([original,direct,mosaic]):
   crop=im.crop(box)
   display=stretch(crop).resize((side,side),Image.Resampling.NEAREST)
   px=pad+j*(side+gap);py=top+rowhead
   sheet.paste(display,(px,py));draw.rectangle((px,py,px+side-1,py+side-1),outline='#aaaaaa',width=1)
   # Save the unmodified pixel crop separately as evidence.
   label=['Original','DirectOverlap' if direct_title=='Direct overlap' else 'DirectOverwrite','Mosaic'][j]
   crop.save(out/f'Region_{i+1}_{label}_Native.png')
  single=sheet.crop((0,top,w,top+rowhead+side))
  single.save(out/f'Region_{i+1}_{"Overlap" if direct_title=="Direct overlap" else "Overwrite"}_Comparison.png',dpi=(200,200))
 draw.text((pad,h-66),'48 x 48 pixel regions enlarged 8x. Identical square root display stretch in every panel.',font=small,fill='#444444')
 draw.text((pad,h-34),'Same input images. No artificial displacement, sharpening or extra blur was added.',font=small,fill='#444444')
 sheet.save(out/path,dpi=(250,250));return sheet

build(overlap,'Direct overlap','Equal weights; no alignment','Local_Original_Overlap_Mosaic.png')
build(hard,'Direct overwrite','Later image replaces earlier image','Local_Original_Overwrite_Mosaic.png')
# Show where every inspected region lies, on the original sky reference in the common frame.
overview=original.copy();draw=ImageDraw.Draw(overview)
for i,r in enumerate(regions):
 x,y=r['x'],r['y'];draw.rectangle((x-24,y-24,x+24,y+24),outline='#ffcc55',width=3)
 draw.text((x+30,y-24),str(i+1),font=font,fill='#ffcc55',stroke_width=2,stroke_fill='black')
overview.save(out/'Local_Region_Locations.png')
notes={'roi_size_pixels':roi,'zoom':zoom,'display':'identical sqrt mapping; no independent normalization',
       'direct_overlap':'Equal-weight average of valid images at nominal positions, without registration',
       'direct_overwrite':'Matches the previous direct-overlay figure; later images replace earlier ones',
       'reference':'Original CDS sky image, mapped to the common frame by one central-channel affine fit',
       'artificial_misalignment':False,'regions':regions,
       'interpretation':'The supplied fields have small initial misalignment. Local differences mainly concern mild broadening, seams and resampling; pronounced double-image ghosts are not established by these examples. Smoother appearance does not establish increased resolution.'}
(out/'local_comparison_notes.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
readme='''局部对比说明\n\nLocal_Original_Overlap_Mosaic.png 为三列对比：原始星图、未配准等权叠加、配准融合后的 mosaic。\nLocal_Original_Overwrite_Mosaic.png 的中列采用此前示意图的直接覆盖方式，后图覆盖前图。\n两种直接叠加方式应区分：直接覆盖可能出现接缝，但通常不会同时保留两份星点；等权叠加在位置有误差时可能使星点展宽或重影。\n\n选取了 A–B、D–E、E–G、F–G 四处真实重叠区域。每格为共同坐标下 48×48 像素，最近邻放大8倍，所有面板使用同一个平方根亮度显示拉伸。Native 文件保存未经亮度拉伸、未经显示放大的对应像素。原始参考图仅用中央G支路拟合一个统一变换，以便显示相应天空区域。\n\n没有人为增加错位、模糊或锐化。当前这组图初始位置本来就较接近，因此明显的双影不强；不能据此声称 mosaic 提升了光学分辨率。截图压缩和重采样也会影响局部外观。\n\nLocal_Region_Locations.png 标出四处区域位置；JSON 文件记录选择位置和处理方式。\n'''
(out/'README_local_zh.txt').write_text(readme,encoding='utf-8')
include=[out/'Local_Original_Overlap_Mosaic.png',out/'Local_Original_Overwrite_Mosaic.png',out/'Local_Region_Locations.png',out/'local_comparison_notes.json',out/'README_local_zh.txt']
include+=sorted(out.glob('Region_*.png'))
zpath=figures/'Local_Detail_Comparisons.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
 for f in include:z.write(f,f.name)
print('Created',w,h,'Archive',zpath.stat().st_size)
