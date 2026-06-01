# pip uninstall torch torchvision torchaudio -y
# pip cache purge
# pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu

import cv2
import os
from ultralytics import YOLO

current_path = os.path.join(os.getcwd(), 'deep_learning', 'yolo')

# Load image
img_path = os.path.join(current_path, 'cars.jpg')
img = cv2.imread(img_path)

# Load pretrained model
model = YOLO("yolo11n.pt")  # nano model
# Run inference
results = model(img)
for box in results[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    print(model.names[cls], conf)

# Draw detections
annotated = results[0].plot()

cv2.imshow("YOLO11 Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
