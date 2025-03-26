import cv2
from ultralytics import YOLO
import cvzone
import math
import torch


device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")


model = YOLO("yolov8l.pt").to(device)  
cap = cv2.VideoCapture("../videos/video2.mp4")
cap.set(3, 1440)
cap.set(4, 900)


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
frame_skip = 2
frame_count = 0

while True:
    success, img = cap.read()
    if not success:
        break
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue
    results = model(img, stream=True, device=device, half=True)  
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (x1, max(40, y1)), scale=2, thickness=1)
    display_img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
    cv2.imshow("YOLOv8 (Optimized)", display_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
