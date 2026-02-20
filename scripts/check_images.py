import os

images_path = "../data/raw/images"

print("Total images:", len(os.listdir(images_path)))
print("Sample images:", os.listdir(images_path)[:5])
