"""Read seven Zemax exports, register their overlaps, and blend a mosaic."""
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from zemax_images import CHANNELS,centers,detect_plot,orient
from registration import register,compose,save_overlap_diagnostics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('registered_output'))
    p.add_argument('--input-kind',choices=['framed','bitmap'],default='framed')
    p.add_argument('--orientation',choices=['identity','rot180','flip-x','flip-y','rot90','rot270'],default='identity')
    p.add_argument('--config',type=Path)
    p.add_argument('--size',type=int,default=652)
    p.add_argument('--channel-fov',type=float,default=3.5)
    p.add_argument('--separation',type=float,default=3.)
    p.add_argument('--search',type=float,default=24,help='Expected center error in registration pixels')
    p.add_argument('--min-ncc',type=float,default=.45)
    args=p.parse_args()
    if args.size<128 or args.channel_fov<=0 or args.separation<=0 or args.search<=0 or not 0<args.min_ncc<1:p.error('Invalid numerical parameter')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.json').exists() or (args.output/'mosaic.png').exists():p.error('Use a new output folder to avoid stale results.')
    cfg=json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    if set(cfg)-{'centers_deg','orientation'}:p.error('Unsupported config keys')
    pos=centers(args.separation);pos.update(cfg.get('centers_deg',{}))
    if set(pos)!=set(CHANNELS) or set(cfg.get('orientation',{}))-set(CHANNELS):p.error('Channels must be A through G')
    for name,xy in pos.items():
        if len(xy)!=2 or not all(math.isfinite(float(v)) for v in xy):p.error(f'Invalid center {name}')
    patches={};meta={};audit=Image.new('RGB',(4*420,2*355),'white')
    for i,name in enumerate(CHANNELS):
        found=[f for f in args.input.iterdir() if f.stem.lower()==f'imagesimulation{name}'.lower() and f.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}]
        if len(found)!=1:raise ValueError(f'Expected one ImageSimulation{name}; found {len(found)}')
        with Image.open(found[0]) as src:im=src.convert('RGB')
        box=detect_plot(im) if args.input_kind=='framed' else (0,0,*im.size)
        patch=im.crop(box)
        if abs(patch.width/patch.height-1)>.02:raise ValueError(f'{name}: expected square field')
        o=cfg.get('orientation',{}).get(name,args.orientation);patches[name]=orient(patch,o)
        meta[name]={'source':str(found[0].resolve()),'crop_box':box,'orientation':o,'nominal_center_deg':pos[name]}
        marked=im.copy();ImageDraw.Draw(marked).rectangle(box,outline='red',width=3);marked.thumbnail((410,320))
        x=(i%4)*420;y=(i//4)*355;audit.paste(marked,(x,y+20));ImageDraw.Draw(audit).text((x+5,y+3),name,fill='black')
    audit.save(args.output/'crop_detection.jpg');(args.output/'crops').mkdir(exist_ok=True)
    for n,im in patches.items():im.save(args.output/'crops'/f'Channel_{n}.png')
    try:images,transforms,pairs=register(patches,pos,args.size,args.channel_fov,args.search,args.min_ncc)
    except Exception as exc:
        (args.output/'FAILED.txt').write_text(str(exc),encoding='utf-8');raise
    out,weights,count,offset=compose(images,transforms)
    save_overlap_diagnostics(images,transforms,pairs,args.output/'overlap_checks',pos,args.channel_fov)
    mosaic=Image.fromarray(out);mosaic.save(args.output/'mosaic.png')
    rgba=mosaic.convert('RGBA');rgba.putalpha(Image.fromarray(np.uint8(weights>1e-5)*255));rgba.save(args.output/'mosaic_rgba.png')
    np.save(args.output/'coverage_count.npy',count)
    palette=np.array([[0,0,0],[90,130,190],[240,180,70],[255,255,255]],np.uint8)
    Image.fromarray(palette[np.minimum(count,3)]).save(args.output/'coverage.png')
    for n in CHANNELS:meta[n]['image_to_mosaic']=(offset@transforms[n]).tolist()
    info={'method':'Coarse overlap search, dense ECC affine, NCC acceptance gates and robust global affine registration',
        'reference_channel':'G','registration_size':args.size,'minimum_pair_ncc':args.min_ncc,
        'input_kind':args.input_kind,
        'content_check':'Check whether the files contain Source Bitmap or Simulated Image. This program does not classify export content automatically.',
        'channels':meta,'pairs':pairs,'covered_pixels':int((count>0).sum()),'overlap_pixels':int((count>1).sum()),
        'limitations':'Locally affine model initialized by approximate field geometry. No absolute celestial WCS, lens-distortion correction, super-resolution or radiometric calibration.'}
    (args.output/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print('Saved registered mosaic:',args.output.resolve())

if __name__=='__main__':main()
