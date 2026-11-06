from pathlib import Path
import pandas as pd
from PIL import Image
d=Path(__file__).resolve().parents[1]/"dataset"
x=pd.read_csv(d/"classification_labels.csv")
print("Total:",len(x))
print(x.groupby(["label","split"]).size().unstack(fill_value=0))
bad=[]
for p in x.image:
 try:
  with Image.open(d/p) as im: im.verify()
 except Exception as e: bad.append((p,str(e)))
print("Unreadable:",len(bad))
