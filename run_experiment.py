import cv2,numpy as np,csv,time,os,platform
from pathlib import Path
root=Path(__file__).resolve().parent
for d in ['dataset','outputs','results']: (root/d).mkdir(exist_ok=True)
rng=np.random.default_rng(52)
rows=[]
for i in range(30):
    img=np.full((480,640,3),230,dtype=np.uint8)
    token=f'{100+i}'
    cv2.rectangle(img,(170,145),(470,325),(255,255,255),-1)
    cv2.rectangle(img,(170,145),(470,325),(30,30,30),2)
    cv2.putText(img,token,(225,255),cv2.FONT_HERSHEY_SIMPLEX,2,(20,20,20),4,cv2.LINE_AA)
    if i%3==1: img=cv2.GaussianBlur(img,(5,5),1.2)
    if i%3==2: img=np.clip(img.astype(np.int16)+rng.normal(0,12,img.shape),0,255).astype('uint8')
    cv2.imwrite(str(root/'dataset'/f'token_{i:02d}.png'),img)
    rows.append([f'token_{i:02d}.png',token,170,145,470,325])
with open(root/'dataset'/'labels.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['file','token','x1','y1','x2','y2']);w.writerows(rows)
records=[]
for file,token,x1,y1,x2,y2 in rows:
    img=cv2.imread(str(root/'dataset'/file))
    t=time.perf_counter()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,binary=cv2.threshold(gray,180,255,cv2.THRESH_BINARY_INV)
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[cv2.boundingRect(c) for c in contours if cv2.contourArea(c)>500]
    elapsed=(time.perf_counter()-t)*1000
    detected=any(abs(x-x1)<15 and abs(y-y1)<15 and abs((x+w)-x2)<15 and abs((y+h)-y2)<15 for x,y,w,h in boxes)
    annotated=img.copy()
    for x,y,w,h in boxes: cv2.rectangle(annotated,(x,y),(x+w,y+h),(0,120,0),2)
    cv2.imwrite(str(root/'outputs'/file),annotated)
    records.append([file,token,int(detected),round(elapsed,4),len(boxes)])
with open(root/'results'/'measurements.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['file','ground_truth_token','document_box_detected','processing_ms','contours']);w.writerows(records)
with open(root/'results'/'summary.txt','w') as f:
    f.write('Synthetic controlled dataset; NOT clinic data.\n')
    f.write(f'Python: {platform.python_version()} OpenCV: {cv2.__version__}\n')
    f.write(f'Document boundary recall: {sum(r[2] for r in records)}/{len(records)} = {sum(r[2] for r in records)/len(records):.2%}\n')
    f.write(f'Mean processing ms/image: {np.mean([r[3] for r in records]):.4f}\n')
    f.write('This is NOT OCR accuracy or a three-package comparison. Run separately in each venv to compare builds.\n')
print((root/'results'/'summary.txt').read_text())
