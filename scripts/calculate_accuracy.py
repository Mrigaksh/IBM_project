import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

print("===================================")
print("YOLOv8 MODEL ACCURACY REPORT")
print("===================================\n")

# FULL absolute paths (IMPORTANT)
model_path = r"C:\Users\Admin\runs\detect\train8\weights\best.pt"
data_yaml = r"C:\Users\Admin\Documents\Clg_stuff\SEM8\IBM_proj\data\data.yaml"

# Load model
model = YOLO(model_path)
print("Model loaded successfully.\n")

# Run validation
metrics = model.val(data=data_yaml)

# Extract metrics (convert array → single value)
precision = metrics.box.p.mean()
recall = metrics.box.r.mean()
map50 = metrics.box.map50

# Print results
print("\n===================================")
print("FINAL RESULTS")
print("===================================")

print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"mAP50 (Accuracy): {map50*100:.2f}%")

print("\n===================================")
print("Validation completed successfully.")
print("===================================")