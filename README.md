# SafeDrive AI — synthetic-data prototype

Includes 1,400 procedurally generated 640x640 illustrative images: 200 per class, with 140 train / 40 validation / 20 test per class. Seven classification classes: safe_alert, mobile_distraction, visual_distraction, drowsiness, yawning, smoking, drinking_container.

**Important:** these are schematic computer-generated illustrations, not photographs. They are suitable for validating code, data loaders, annotation formats and end-to-end integration, but not for demonstrating real-world accuracy or safety. A drink container does not establish alcohol use or impairment.

## Windows setup
Use Python 3.11 or 3.12 64-bit:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
If Torch installation fails, install the appropriate CPU/CUDA build from the official PyTorch installation selector, then install remaining requirements.

## Check dataset
```powershell
python scripts/check_dataset.py
```
## Train MobileNetV2
```powershell
python src/train_classifier.py --epochs 8 --batch-size 32
```
## Train YOLOv8
```powershell
yolo detect train data=dataset/dataset.yaml model=yolov8n.pt imgsz=640 epochs=30 batch=8 project=runs name=safedrive
```
## Evaluate and launch
```powershell
python src/evaluate_classifier.py
streamlit run app.py
```
Classifier weights are saved to models/mobilenetv2_safedrive.pth. YOLO labels cover driver, phone, cigarette and drink_container. No pretrained detector weights are bundled.

## Live webcam monitoring

Install the requirements in a supported Python environment, then run:

```powershell
python live_webcam.py
```

Press `Q` in the webcam window to stop. The live module uses MediaPipe Face Mesh landmarks with OpenCV to estimate eye closure (EAR), mouth opening (MAR), and head yaw. Thresholds are adjustable, for example `python live_webcam.py --eye-seconds 1.7 --yaw 28`.

If a trained YOLO weights file exists at `runs/detect/safedrive/weights/best.pt`, the script also attempts optional object detection. The included synthetic training images are illustrative and are not reliable real-world training data; object detection trained only on them may not recognize real phones, cigarettes, or containers. Webcam face-state estimates are a research prototype, not a validated driver-safety or medical system.
