from ultralytics import YOLO
import cv2
import os

model = YOLO("runs/detect/train8/weights/best.pt")

input_folder = "data/raw/images/test"
output_folder = "outputs/crops"
os.makedirs(output_folder, exist_ok=True)

results = model(input_folder)

i = 0
for r in results:
    img = r.orig_img
    for box in r.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(f"{output_folder}/plate_{i}.jpg", crop)
        i += 1

print("Plates cropped successfully")
