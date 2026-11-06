from pathlib import Path
import streamlit as st,torch,pandas as pd
from torch import nn
from torchvision import models,transforms
from PIL import Image
R=Path(__file__).resolve().parent; names=["safe_alert","mobile_distraction","visual_distraction","drowsiness","yawning","smoking","drinking_container"]
st.set_page_config(page_title="SafeDrive AI",page_icon="🚗",layout="wide")
st.title("🚗 SafeDrive AI — Driver Monitoring Prototype")
st.caption("Synthetic-data research demonstration — not for real-world safety decisions.")
st.warning("The included images are computer-generated illustrations. Predictions are experimental and should not be used to assess a real driver's fitness or impairment.")
w=R/"models/mobilenetv2_safedrive.pth"; f=st.file_uploader("Upload an image",type=["jpg","jpeg","png"])
if not w.exists(): st.info("Train the classifier first: python src/train_classifier.py")
elif f:
 ck=torch.load(w,map_location="cpu",weights_only=False); m=models.mobilenet_v2(weights=None); m.classifier[1]=nn.Linear(m.last_channel,len(names)); m.load_state_dict(ck["state_dict"]); m.eval()
 tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 im=Image.open(f).convert("RGB"); st.image(im,use_container_width=True)
 with torch.no_grad(): probs=torch.softmax(m(tf(im).unsqueeze(0)),1)[0]
 ix=int(probs.argmax()); a,b=st.columns(2); a.metric("Predicted class",names[ix].replace("_"," ").title()); b.metric("Confidence",f"{float(probs[ix])*100:.1f}%")
 st.bar_chart(pd.DataFrame({"score":probs.numpy()},index=names))
d=pd.read_csv(R/"dataset/classification_labels.csv"); st.subheader("Dataset summary"); st.write(f"{len(d)} images · {d.label.nunique()} classes"); st.dataframe(d.groupby(["label","split"]).size().unstack(fill_value=0))
st.caption("A bottle or cup is not proof of alcohol consumption or intoxication.")
