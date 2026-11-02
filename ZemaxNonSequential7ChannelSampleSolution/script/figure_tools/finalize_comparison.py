"""Keep the two selected regions and label them Region 1 and Region 2."""
from pathlib import Path
import os
from PIL import Image,ImageDraw,ImageFont

root=Path(__file__).resolve().parents[1]
figures=Path(os.environ.get('TELESCOPE_FIGURE_DIR',str(root/'comparison_figures')))
folder=figures/'local_details'
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',29)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21)
bold=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',23)
rows=[Image.open(folder/f'Region_{n}_Overlap_Comparison.png').convert('RGB') for n in [2,4]]
width=rows[0].width;header=112;gap=20;footer=84
height=header+sum(im.height for im in rows)+gap+footer
sheet=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(sheet)
for j,(title,subtitle) in enumerate(zip(['Original sky reference','Direct overlap','Mosaic'],['Corresponding input region','Equal weights; no alignment','Aligned and blended'])):
    x=28+j*404
    draw.text((x,20),title,font=font,fill='black');draw.text((x,62),subtitle,font=small,fill='#444444')
y=header
for n,(im,label) in enumerate(zip(rows,['Channels D and E','Channels F and G']),1):
    sheet.paste(im,(0,y));draw.rectangle((0,y,width-1,y+43),fill='white')
    draw.text((28,y+8),f'Region {n}   {label}',font=bold,fill='black');y+=im.height+gap
draw.text((28,height-66),'48 x 48 pixel regions enlarged 8x. Identical square root display stretch in every panel.',font=small,fill='#444444')
draw.text((28,height-34),'Same input images. No artificial displacement, sharpening or extra blur was added.',font=small,fill='#444444')
sheet.save(folder/'Local_Regions_1_and_2_Alignment_Comparison.png',dpi=(250,250))
