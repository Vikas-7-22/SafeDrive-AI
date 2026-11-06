import argparse, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"dataset"
CLASSES=["safe_alert","mobile_distraction","visual_distraction","drowsiness","yawning","smoking","drinking_container"]
class DS(Dataset):
 def __init__(self,split,tf):
  self.df=pd.read_csv(DATA/"classification_labels.csv"); self.df=self.df[self.df.split==split].reset_index(drop=True); self.tf=tf
 def __len__(self): return len(self.df)
 def __getitem__(self,i):
  r=self.df.iloc[i]; return self.tf(Image.open(DATA/r.image).convert("RGB")),int(r.class_id)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=8); p.add_argument("--batch-size",type=int,default=32); a=p.parse_args()
 torch.manual_seed(42); np.random.seed(42); random.seed(42); device="cuda" if torch.cuda.is_available() else "cpu"
 norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
 tr=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(.12,.12,.12,.04),transforms.ToTensor(),norm])
 ev=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),norm])
 train=DataLoader(DS("train",tr),batch_size=a.batch_size,shuffle=True,num_workers=0); val=DataLoader(DS("val",ev),batch_size=a.batch_size,num_workers=0)
 try: model=models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
 except Exception: model=models.mobilenet_v2(weights=None)
 model.classifier[1]=nn.Linear(model.last_channel,len(CLASSES)); model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); lossfn=nn.CrossEntropyLoss()
 for ep in range(a.epochs):
  model.train(); total=correct=0; loss_sum=0
  for x,y in train:
   x,y=x.to(device),y.to(device); opt.zero_grad(); z=model(x); loss=lossfn(z,y); loss.backward(); opt.step()
   loss_sum+=loss.item()*len(y); correct+=(z.argmax(1)==y).sum().item(); total+=len(y)
  model.eval(); vc=vt=0
  with torch.no_grad():
   for x,y in val:
    x,y=x.to(device),y.to(device); z=model(x); vc+=(z.argmax(1)==y).sum().item(); vt+=len(y)
  print(f"Epoch {ep+1}/{a.epochs} loss={loss_sum/total:.4f} train_acc={correct/total:.3f} val_acc={vc/vt:.3f}")
 torch.save({"state_dict":model.cpu().state_dict(),"classes":CLASSES},ROOT/"models/mobilenetv2_safedrive.pth")
 print("Saved model weights.")
if __name__=="__main__": main()
