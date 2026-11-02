"""Content-based overlap registration. Nominal geometry only initializes the search."""
import itertools, math
import cv2
import numpy as np

cv2.setNumThreads(1)
cv2.setRNGSeed(2026)

def H(M):
    h=np.eye(3);h[:2]=M;return h

def ncc(a,b,mask):
    m=mask>0
    if m.sum()<200:return -1.
    x=a[m].astype(float);y=b[m].astype(float);x-=x.mean();y-=y.mean()
    return float((x*y).sum()/max(1e-12,np.sqrt((x*x).sum()*(y*y).sum())))

def score(a,b,M,mask):
    s=a.shape[0]
    wa=cv2.warpAffine(a,M,(s,s))
    valid=(cv2.warpAffine(mask,M,(s,s),flags=cv2.INTER_NEAREST)>0)&(mask>0)
    valid=cv2.erode(valid.astype(np.uint8),np.ones((5,5),np.uint8))
    return ncc(wa,b,valid),valid

def plausible(M,nominal,size,search):
    sv=np.linalg.svd(M[:,:2],compute_uv=False)
    angle=abs(math.degrees(math.atan2(M[1,0],M[0,0])))
    cp=np.array([(size-1)/2]*2+[1.])
    return (np.linalg.det(M[:,:2])>0 and sv.min()>.94 and sv.max()<1.06
        and angle<4 and np.linalg.norm((M-nominal)@cp)<search*1.7)

def pair_register(a,b,mask,nominal,search=24,min_ncc=.45):
    size=a.shape[0];before,_=score(a,b,nominal,mask)
    # Translation search on reduced images gives ECC a wider capture range.
    small=size//2;f=small/size
    sa=cv2.resize(a,(small,small));sb=cv2.resize(b,(small,small))
    sm=cv2.resize(mask,(small,small),interpolation=cv2.INTER_NEAREST)
    base=nominal.copy();base[:,2]*=f
    grid=np.unique(np.r_[np.arange(-search,search+.1,6.),0.])
    coarse=[]
    for dx in grid:
        for dy in grid:
            M=base.copy();M[:,2]+=[dx*f,dy*f]
            v,_=score(sa,sb,M,sm);coarse.append((v,dx,dy))
    coarse.sort(reverse=True,key=lambda x:x[0]);starts=[nominal.copy()];selected=[]
    for v,dx,dy in coarse:
        if all(np.hypot(dx-x,dy-y)>7 for x,y in selected):
            M=nominal.copy();M[:,2]+=[dx,dy];starts.append(M);selected.append((dx,dy))
        if len(selected)==2:break
    candidates=[]
    for initial in starts:
        z=cv2.warpAffine(a,initial,(size,size))
        vm=cv2.bitwise_and(cv2.warpAffine(mask,initial,(size,size),flags=cv2.INTER_NEAREST),mask)
        vm=cv2.erode(vm,np.ones((5,5),np.uint8))
        if np.count_nonzero(vm)<300:continue
        try:
            ecc,W=cv2.findTransformECC(b,z,np.eye(2,3,dtype=np.float32),cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,150,1e-5),inputMask=vm,gaussFiltSize=3)
            M=(np.linalg.inv(H(W))@H(initial))[:2]
            if plausible(M,nominal,size,search):
                v,valid=score(a,b,M,mask);candidates.append((v,M,int(valid.sum())))
        except (cv2.error,np.linalg.LinAlgError):continue
    candidates.sort(reverse=True,key=lambda x:x[0])
    info={'ncc_before':before,'accepted':False,'method':'coarse overlap search and ECC affine'}
    if not candidates:info['reason']='No plausible converged image registration';return info,None
    v,M,area=candidates[0];info.update(ncc_after=v,overlap_pixels=area,matrix_a_to_b=M.tolist())
    if v<min_ncc or area<500:info['reason']='Insufficient overlap similarity or support';return info,None
    cp=np.array([(size-1)/2]*2+[1.])
    for other,N,_ in candidates[1:]:
        if v-other<.03 and np.linalg.norm((M-N)@cp)>4:
            info['reason']='Competing overlap matches are ambiguous';return info,None
    _,valid=score(a,b,M,mask);yy,xx=np.where(valid>0)
    # These are constraints sampled from a measured dense transform, not independent star matches.
    sel=np.arange(0,len(xx),max(1,len(xx)//40))[:40]
    q=np.c_[xx[sel],yy[sel]].astype(float)
    p=(np.c_[q,np.ones(len(q))]@np.linalg.inv(H(M)).T)[:,:2]
    info['accepted']=True
    return info,{'p':p,'q':q,'confidence':max(.05,v)**2}

def connected(edges):
    seen={'G'}
    while True:
        new=seen.copy()
        for e in edges:
            if e['a'] in seen or e['b'] in seen:new.update([e['a'],e['b']])
        if seen==new:return seen==set('ABCDEFG')
        seen=new

def solve_global(edges,size):
    if not connected(edges):raise ValueError('Accepted overlap graph is disconnected. No mosaic can be produced.')
    idx={n:i for i,n in enumerate('ABCDEF')};A=[];B=[];weights=[];center=(size-1)/2
    for e in edges:
        for p,q in zip(e['p'],e['q']):
            row=np.zeros(18);rhs=np.zeros(2)
            for name,xy,sign in [(e['a'],p,1),(e['b'],q,-1)]:
                if name=='G':rhs-=sign*(xy-center)
                else:row[3*idx[name]:3*idx[name]+3]+=sign*np.r_[xy-center,1.]
            A.append(row);B.append(rhs);weights.append(e['confidence']/len(e['p']))
    A=np.array(A);B=np.array(B);base=np.sqrt(weights);w=base.copy()
    if np.linalg.matrix_rank(A)<18:raise ValueError('Overlap constraints are degenerate.')
    if np.linalg.cond(A)>1e6:raise ValueError('Overlap constraints are poorly conditioned.')
    for _ in range(8):
        sol=np.linalg.lstsq(A*w[:,None],B*w[:,None],rcond=None)[0]
        residual=np.linalg.norm(A@sol-B,axis=1)
        w=base*np.sqrt(np.minimum(1,1.5/np.maximum(residual,1e-6)))
    C=np.array([[1,0,-center],[0,1,-center],[0,0,1.]])
    transforms={'G':C}
    for n,i in idx.items():
        mat=np.eye(3);mat[:2]=sol[3*i:3*i+3].T;transforms[n]=mat@C
    for e in edges:
        d=np.c_[e['p'],np.ones(len(e['p']))]@transforms[e['a']].T-np.c_[e['q'],np.ones(len(e['q']))]@transforms[e['b']].T
        e['global_rms_px']=float(np.sqrt(np.mean(np.sum(d[:,:2]**2,axis=1))))
    return transforms

def register(patches,pos,size=652,channel_fov=3.5,search=24,min_ncc=.45):
    images={};signals={};yy,xx=np.mgrid[:size,:size]
    mask=np.uint8(np.hypot(xx-(size-1)/2,yy-(size-1)/2)<size*.475)*255
    for name,im in patches.items():
        images[name]=np.asarray(im.resize((size,size)))
        f=cv2.cvtColor(images[name],cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        signals[name]=cv2.GaussianBlur(f,(0,0),.9)-cv2.GaussianBlur(f,(0,0),5)
    infos=[];edges=[]
    for a,b in itertools.combinations('ABCDEFG',2):
        if np.linalg.norm(np.array(pos[a])-pos[b])>channel_fov+2*search*channel_fov/size:continue
        dx=(pos[a][0]-pos[b][0])*size/channel_fov;dy=-(pos[a][1]-pos[b][1])*size/channel_fov
        nominal=np.array([[1.,0,dx],[0,1.,dy]])
        info,edge=pair_register(signals[a],signals[b],mask,nominal,search,min_ncc)
        info.update(a=a,b=b);infos.append(info)
        print(f'{a}-{b}: {info["ncc_before"]:.3f} -> {info.get("ncc_after",-1):.3f}, {"accepted" if edge else "rejected"}',flush=True)
        if edge:edge.update(a=a,b=b);edges.append(edge)
    transforms=solve_global(edges,size)
    bad=[e for e in edges if e['global_rms_px']>2.5]
    if bad:
        bad_ids={id(e) for e in bad}
        edges=[e for e in edges if id(e) not in bad_ids];transforms=solve_global(edges,size)
        for e in bad:
            info=next(i for i in infos if (i['a'],i['b'])==(e['a'],e['b']))
            info.update(accepted=False,reason='Global loop inconsistency')
    for e in edges:
        if e['global_rms_px']>2.5:raise ValueError('Global loop residual exceeds 2.5 registration pixels.')
        relative=(np.linalg.inv(transforms[e['b']])@transforms[e['a']])[:2]
        v,_=score(signals[e['a']],signals[e['b']],relative,mask)
        info=next(i for i in infos if (i['a'],i['b'])==(e['a'],e['b']))
        info.update(global_ncc=v,global_rms_px=e['global_rms_px'])
        if v<min_ncc-.08:raise ValueError(f'Global overlap similarity failed for {e["a"]}-{e["b"]}.')
    for n,mat in transforms.items():
        sv=np.linalg.svd(mat[:2,:2],compute_uv=False)
        fitted=(mat@np.array([(size-1)/2]*2+[1]))[:2]
        nominal=np.array(pos[n])*[size/channel_fov,-size/channel_fov]
        angle=abs(math.degrees(math.atan2(mat[1,0],mat[0,0])))
        if sv.min()<.92 or sv.max()>1.08 or angle>8 or np.linalg.norm(fitted-nominal)>search*2:
            raise ValueError(f'Global transform for {n} exceeds the search bounds.')
    return images,transforms,infos

def compose(images,transforms):
    size=next(iter(images.values())).shape[0]
    corners=np.array([[0,0,1],[size,0,1],[0,size,1],[size,size,1.]])
    pts=np.vstack([corners@mat.T for mat in transforms.values()]);lo=np.floor(pts[:,:2].min(0))-8;hi=np.ceil(pts[:,:2].max(0))+8
    w,h=(hi-lo).astype(int);offset=np.eye(3);offset[:2,2]=-lo
    yy,xx=np.mgrid[:size,:size];r=np.hypot(xx-(size-1)/2,yy-(size-1)/2)/(size/2)
    alpha=np.clip((.988-r)/.06,0,1).astype(np.float32)
    alpha[:3]=0;alpha[-3:]=0;alpha[:,:3]=0;alpha[:,-3:]=0
    sums=np.zeros((h,w,3),np.float32);weights=np.zeros((h,w),np.float32);count=np.zeros((h,w),np.uint8)
    for n,a in images.items():
        mat=offset@transforms[n]
        weight=cv2.warpAffine(alpha,mat[:2],(w,h))
        sums+=cv2.warpAffine(a.astype(np.float32)*alpha[...,None],mat[:2],(w,h))
        weights+=weight;count+=(weight>1e-4).astype(np.uint8)
    out=np.zeros_like(sums);np.divide(sums,weights[...,None],out=out,where=weights[...,None]>1e-5)
    return np.uint8(np.clip(np.rint(out),0,255)),weights,count,offset

def save_overlap_diagnostics(images,transforms,pairs,output,positions,channel_fov=3.5):
    """Show the same overlap before and after fitting; red/green separation reveals errors."""
    from PIL import Image,ImageDraw
    import csv
    output.mkdir(exist_ok=True);size=next(iter(images.values())).shape[0]
    yy,xx=np.mgrid[:size,:size]
    mask=np.uint8(np.hypot(xx-(size-1)/2,yy-(size-1)/2)<size*.475)*255
    for pair in pairs:
        a,b=pair['a'],pair['b']
        dx=(positions[a][0]-positions[b][0])*size/channel_fov
        dy=-(positions[a][1]-positions[b][1])*size/channel_fov
        before=np.array([[1.,0,dx],[0,1.,dy]])
        after=(np.linalg.inv(transforms[b])@transforms[a])[:2]
        valid=(mask>0)&(cv2.warpAffine(mask,after,(size,size),flags=cv2.INTER_NEAREST)>0)
        valid&=cv2.warpAffine(mask,before,(size,size),flags=cv2.INTER_NEAREST)>0
        y,x=np.where(valid)
        if len(x)<50:continue
        bounds=(max(0,x.min()-5),max(0,y.min()-5),min(size,x.max()+6),min(size,y.max()+6))
        panels=[]
        for mat in [before,after]:
            one=cv2.warpAffine(images[a],mat,(size,size))
            ga=cv2.cvtColor(one,cv2.COLOR_RGB2GRAY).astype(float)
            gb=cv2.cvtColor(images[b],cv2.COLOR_RGB2GRAY).astype(float)
            # Identical display stretch on both channels; not a photometric correction.
            ga=np.uint8(np.clip(np.sqrt(ga/255)*255,0,255));gb=np.uint8(np.clip(np.sqrt(gb/255)*255,0,255))
            overlay=np.stack([ga,gb,np.minimum(ga,gb)],axis=2);overlay[~valid]=0
            im=Image.fromarray(overlay).crop(bounds)
            scale=min(480/im.width,450/im.height)
            im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))))
            panels.append(im)
        height=max(im.height for im in panels)+88
        sheet=Image.new('RGB',(1000,height),'white');draw=ImageDraw.Draw(sheet)
        status='accepted constraint' if pair['accepted'] else 'rejected constraint; final view uses other connections'
        draw.text((12,8),f'{a}-{b}: {status}',fill='black')
        for i,im in enumerate(panels):
            draw.text((i*500+12,30),'Nominal positions' if i==0 else 'Global registration',fill='black')
            sheet.paste(im,(i*500+(500-im.width)//2,58))
        draw.text((12,height-18),'Same common overlap and display stretch. Coincident signal is neutral; red/green separation indicates disagreement.',fill='black')
        sheet.save(output/f'overlap_{a}{b}.png')
    with (output/'pair_scores.csv').open('w',newline='',encoding='utf-8-sig') as f:
        keys=['a','b','accepted','ncc_before','ncc_after','global_ncc','global_rms_px','overlap_pixels','reason']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(pairs)
