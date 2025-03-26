from ultralytics import YOLO
import cv2
model=YOLO("../yolo-weights/yolov8l.pt")
result = model("images/3.jpg",show=True)
cv2.waitKey(0)