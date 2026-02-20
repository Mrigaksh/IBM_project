import os
import xml.etree.ElementTree as ET

def convert_folder(folder_path):
    print(f"Converting annotations in: {folder_path}")

    for xml_file in os.listdir(folder_path):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(folder_path, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        w = float(size.find("width").text)
        h = float(size.find("height").text)

        txt_filename = xml_file.replace(".xml", ".txt")
        txt_path = os.path.join(folder_path, txt_filename)

        with open(txt_path, "w") as f:
            for obj in root.findall("object"):
                cls_id = 0  # license plate class

                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                x_center = ((xmin + xmax) / 2) / w
                y_center = ((ymin + ymax) / 2) / h
                width = (xmax - xmin) / w
                height = (ymax - ymin) / h

                f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

    print(f"Done: {folder_path}\n")


# Convert both folders
convert_folder("data/raw/labels/train")
convert_folder("data/raw/labels/test")

print("All conversions completed!")
