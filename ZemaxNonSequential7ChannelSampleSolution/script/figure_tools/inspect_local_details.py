from pathlib import Path
import sys,json,math
root=Path(__file__).resolve().parents[1]
import os
results=Path(os.environ.get('TELESCOPE_RESULTS_DIR',str(root/'registered_output')))
figures=Path(os.environ.get('TELESCOPE_FIGURE_DIR',str(root/'comparison_figures')))
project=Path(os.environ.get('TELESCOPE_PROJECT_DIR',str(root.parent)))

import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
cv2.setNumThreads(1)
out=figures/'local_details';out.mkdir(exist_ok=True)
meta=json.loads((results/'manifest.json').read_text());notes=json.loads((figures/'overlay_notes.json').read_text())
mosaic=np.asarray(Image.open(results/'mosaic.png').convert('RGB'));height,width=mosaic.shape[:2];size=652
T={n:np.array(v['image_to_mosaic']) for n,v in meta['channels'].items()}
yy,xx=np.mgrid[:size,:size];mask=np.float32(np.hypot(xx-325.5,yy-325.5)<size*.494)
mask[:3]=0;mask[-3:]=0;mask[:,:3]=0;mask[:,-3:]=0
layers={};masks={};sums=np.zeros_like(mosaic,dtype=np.float32);count=np.zeros((height,width),np.float32)
for n in 'ABCDEFG':
 im=np.asarray(Image.open(results/f'crops/Channel_{n}.png').convert('RGB').resize((size,size)))
 loc=notes['placements'][n];M=np.array([[1.,0,loc['left']],[0,1.,loc['top']]])
 layer=cv2.warpAffine(im,M,(width,height));valid=cv2.warpAffine(mask,M,(width,height),flags=cv2.INTER_NEAREST)
 layers[n]=layer;masks[n]=valid;sums+=layer*valid[...,None];count+=valid
average=np.zeros_like(sums);np.divide(sums,count[...,None],out=average,where=count[...,None]>0)
average=np.uint8(np.clip(np.rint(average),0,255));Image.fromarray(average).save(out/'Direct_Overlap_Equal_Weight.png')
hard=np.asarray(Image.open(figures/'Direct_Overlay_Clean.png').convert('RGB'))
raw=np.asarray(Image.open(project/'cutout-CDS_P_DSS2_color.jpg').convert('RGB'))
# Align the original sky reference to the central channel only, then use that one mapping everywhere.
crop=round(raw.shape[1]*3.5/11);left=round(raw.shape[1]/2-crop/2);top=round(raw.shape[0]/2-crop/2)
rawg=raw[top:top+crop,left:left+crop];rawg=cv2.resize(rawg,(size,size),interpolation=cv2.INTER_CUBIC)
g=np.asarray(Image.open(results/'crops/Channel_G.png').resize((size,size)))
def sig(a):
 f=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
 return cv2.GaussianBlur(f,(0,0),.8)-cv2.GaussianBlur(f,(0,0),4)
cc,W=cv2.findTransformECC(sig(g),sig(rawg),np.eye(2,3,dtype=np.float32),cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,150,1e-6),inputMask=np.uint8(mask)*255,gaussFiltSize=3)
H=np.eye(3);H[:2]=W
C=np.array([[size/crop,0,-left*size/crop],[0,size/crop,-top*size/crop],[0,0,1.]])
reference_T=T['G']@np.linalg.inv(H)@C
reference=cv2.warpAffine(raw,reference_T[:2],(width,height),flags=cv2.INTER_CUBIC)
Image.fromarray(reference).save(out/'Original_Reference_Aligned.png')
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
# Locate stars or structure inside each real overlap. No artificial misalignment is introduced.
refgray=cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY).astype(np.float32)
hi=np.maximum(0,cv2.GaussianBlur(refgray,(0,0),.7)-cv2.GaussianBlur(refgray,(0,0),3))
blur=cv2.GaussianBlur(hi,(0,0),3)
selection=[]
for pair in meta['pairs']:
 a,b=pair['a'],pair['b'];valid=(masks[a]>0)&(masks[b]>0)
 core=cv2.erode(valid.astype(np.uint8),np.ones((15,15),np.uint8))
 score=blur.copy();score[core==0]=-1
 py,px=np.unravel_index(np.argmax(score),score.shape)
 selection.append({'a':a,'b':b,'x':int(px),'y':int(py),'signal':float(score[py,px]),'accepted':pair['accepted']})
thumb=300;boxsize=64;rows=len(selection);sheet=Image.new('RGB',(thumb*4,rows*(thumb+42)+45),'#0a0c10');draw=ImageDraw.Draw(sheet)
for j,text in enumerate(['Original sky reference','Direct overlap 50/50','Direct overwrite','Aligned mosaic']):draw.text((j*thumb+8,10),text,font=small,fill='white')
for i,item in enumerate(selection):
 x,y=item['x'],item['y'];box=(x-boxsize//2,y-boxsize//2,x+boxsize//2,y+boxsize//2);row=45+i*(thumb+42)
 for j,im in enumerate([reference,average,hard,mosaic]):
  patch=np.asarray(Image.fromarray(im).crop(box)).astype(float)
  patch=np.uint8(np.clip(np.sqrt(patch/255)*255,0,255))
  pic=Image.fromarray(patch).resize((thumb,thumb),Image.Resampling.NEAREST)
  sheet.paste(pic,(j*thumb,row+30))
 draw.text((8,row+5),f'{item["a"]}-{item["b"]}: x={x} y={y}',font=font,fill='white')
sheet.save(out/'candidate_contact.jpg')
(out/'candidate_regions.json').write_text(json.dumps({'reference_alignment_ecc':float(cc),'reference_mapping':reference_T.tolist(),'regions':selection},indent=2),encoding='utf-8')
print('Saved contact',sheet.size,'Reference ECC',cc)
for i in selection:print(i)
