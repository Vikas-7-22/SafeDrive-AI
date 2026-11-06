from pathlib import Path
import pandas as pd,torch
from torch import nn
from torchvision import models,transforms
from PIL import Image
from sklearn.metrics import classification_report,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]; D=ROOT/"dataset"
ck=torch.load(ROOT/"models/mobilenetv2_safedrive.pth",map_location="cpu",weights_only=False); names=ck["classes"]
m=models.mobilenet_v2(weights=None); m.classifier[1]=nn.Linear(m.last_channel,len(names)); m.load_state_dict(ck["state_dict"]); m.eval()
tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
df=pd.read_csv(D/"classification_labels.csv"); df=df[df.split=="test"]; y=[]; p=[]
with torch.no_grad():
 for _,r in df.iterrows():
  x=tf(Image.open(D/r.image).convert("RGB")).unsqueeze(0); p.append(m(x).argmax(1).item()); y.append(int(r.class_id))
print(classification_report(y,p,target_names=names,zero_division=0)); print(confusion_matrix(y,p))
