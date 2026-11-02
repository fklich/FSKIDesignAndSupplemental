"""Meaningful checks for crop detection, false-match rejection and error recovery."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import cv2
from zemax_images import detect_plot,centers
from registration import register,pair_register,H

def main():
    root=Path(__file__).resolve().parent
    baseline=json.loads((root/'registered_output/manifest.json').read_text())
    patches={n:Image.open(root/f'registered_output/crops/Channel_{n}.png').convert('RGB').resize((652,652)) for n in 'ABCDEFG'}
    # Detection must adapt to different frame padding, rather than returning fixed pixels.
    square=Image.new('RGB',(200,200),'black')
    for pad in [12,37]:
        frame=ImageOps.expand(square,border=pad,fill='white')
        assert detect_plot(frame)==(pad,pad,pad+200,pad+200)
    try:detect_plot(Image.new('RGB',(300,300),'white'))
    except ValueError:pass
    else:raise AssertionError('Blank export must be rejected')
    rng=np.random.default_rng(2026);s=256;y,x=np.mgrid[:s,:s]
    mask=np.uint8(np.hypot(x-127.5,y-127.5)<s*.475)*255
    noise1=rng.normal(0,.1,(s,s)).astype(np.float32);noise2=rng.normal(0,.1,(s,s)).astype(np.float32)
    info,result=pair_register(noise1,noise2,mask,np.eye(2,3),search=12)
    assert result is None,'Unrelated images must not be accepted'
    # Inject independent errors into the actual exported fields. G remains the reference.
    settings={'A':(.5,4,-3,1.002),'B':(-.4,-3,3,.998),'C':(.35,3,2,1.001),
        'D':(-.3,-4,-2,.999),'E':(.4,2,-4,1.002),'F':(-.25,-3,-2,.999),'G':(0,0,0,1)}
    changed={};injected={}
    for n,(angle,dx,dy,scale) in settings.items():
        M=cv2.getRotationMatrix2D((325.5,325.5),angle,scale);M[:,2]+=[dx,dy]
        a=cv2.warpAffine(np.asarray(patches[n]),M,(652,652))
        changed[n]=Image.fromarray(a);injected[n]=H(M)
    _,T,pairs=register(changed,centers(),652,search=30)
    old={n:np.array(v['image_to_mosaic']) for n,v in baseline['channels'].items()}
    old={n:np.linalg.inv(old['G'])@mat for n,mat in old.items()}
    # Compare coordinates after undoing the injected transform. Test interior and edge locations.
    points=np.array([[100,100,1],[325.5,325.5,1],[550,550,1],[100,550,1],[550,100,1.]])
    errors={}
    for n in 'ABCDEFG':
        estimate=np.linalg.inv(T['G'])@T[n]@injected[n]
        d=points@estimate.T-points@old[n].T
        errors[n]=float(np.sqrt(np.mean(np.sum(d[:,:2]**2,axis=1))))
    report={'crop_padding_invariance':'passed','blank_export_rejection':'passed',
      'unrelated_image_rejection':'passed','injected_errors':settings,'recovery_rms_registration_pixels':errors,
      'accepted_pairs':sum(i['accepted'] for i in pairs)}
    print(json.dumps(report,indent=2))
    (root/'validation_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert max(errors.values())<3.0,errors
    print('All checks passed.')

if __name__=='__main__':main()
