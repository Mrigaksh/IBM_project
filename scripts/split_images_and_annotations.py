import os
import shutil
import random

# -------- PATHS --------
SOURCE_DIR = "data/raw/images"        # where jpeg + xml are mixed
IMAGE_OUT = "data/raw/images"
ANN_OUT = "data/raw/annotations"

TRAIN_RATIO = 0.8   # 80% train, 20% test

# -------- CREATE FOLDERS --------
for split in ["train", "test"]:
    os.makedirs(os.path.join(IMAGE_OUT, split), exist_ok=True)
    os.makedirs(os.path.join(ANN_OUT, split), exist_ok=True)

# -------- COLLECT IMAGE FILES --------
images = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith((".jpg", ".jpeg"))]
random.shuffle(images)

split_index = int(len(images) * TRAIN_RATIO)
train_images = images[:split_index]
test_images = images[split_index:]

def move_files(image_list, split):
    for img in image_list:
        xml = img.replace(".jpg", ".xml").replace(".jpeg", ".xml")

        img_src = os.path.join(SOURCE_DIR, img)
        xml_src = os.path.join(SOURCE_DIR, xml)

        if not os.path.exists(xml_src):
            continue

        shutil.move(img_src, os.path.join(IMAGE_OUT, split, img))
        shutil.move(xml_src, os.path.join(ANN_OUT, split, xml))

# -------- MOVE FILES --------
move_files(train_images, "train")
move_files(test_images, "test")

print("✅ Images and XML files separated into train and test folders.")
