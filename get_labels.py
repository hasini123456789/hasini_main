import pandas as pd
from PIL import Image
import os
from collections import defaultdict

# === Files & settings ===
classname_file = "class-descriptions-boxable.csv"
annotations_file = "oidv6-train-annotations-bbox.csv"
image_ids_file = "image_ids.txt"  # image IDs without "train/" prefix
filtered_classes_file = "filtered_class_names.txt"
image_folder = "dataset/train/images"
labels_dir = "dataset/train/labels"
os.makedirs(labels_dir, exist_ok=True)

# === Step 1: Load filtered class names ===
with open(filtered_classes_file, "r") as f:
    filtered_class_names = [line.strip() for line in f.readlines()]

# === Step 2: Map class name to index ===
class_name_to_index = {name: i for i, name in enumerate(filtered_class_names)}

# === Step 3: Load mappings and data ===
classes_df = pd.read_csv(classname_file, header=None, names=['LabelName', 'ClassName'])
labels_to_class = dict(zip(classes_df["LabelName"], classes_df["ClassName"]))

annots_df = pd.read_csv(annotations_file, usecols=["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"])

with open(image_ids_file, "r") as file:
    image_ids = [line.replace("train/", "").strip() for line in file.readlines()]
image_ids_set = set(image_ids)

annots_filtered = annots_df[annots_df["ImageID"].isin(image_ids_set)]

# === Image size cache ===
image_size_cache = {}

def get_image_size(image_id):
    if image_id not in image_size_cache:
        img_path = os.path.join(image_folder, image_id + ".jpg")
        with Image.open(img_path) as img:
            image_size_cache[image_id] = img.size
    return image_size_cache[image_id]

def bbox_to_yolo(xmin_pixel, ymin_pixel, xmax_pixel, ymax_pixel, image_width, image_height):
    x_center = ((xmin_pixel + xmax_pixel) / 2) / image_width
    y_center = ((ymin_pixel + ymax_pixel) / 2) / image_height
    width = (xmax_pixel - xmin_pixel) / image_width
    height = (ymax_pixel - ymin_pixel) / image_height
    return x_center, y_center, width, height

# === Process annotations ===
to_enter = []

for _, row in annots_filtered.iterrows():
    image_id = row["ImageID"]
    class_name = labels_to_class.get(row["LabelName"])

    # Only include classes in the filtered list
    class_index = class_name_to_index.get(class_name)
    if class_index is None:
        continue

    image_width, image_height = get_image_size(image_id)

    xmin_pixel = row["XMin"] * image_width
    xmax_pixel = row["XMax"] * image_width
    ymin_pixel = row["YMin"] * image_height
    ymax_pixel = row["YMax"] * image_height

    x_c, y_c, w, h = bbox_to_yolo(xmin_pixel, ymin_pixel, xmax_pixel, ymax_pixel, image_width, image_height)

    to_enter.append({
        "image_id": image_id,
        "class_index": class_index,
        "x_center": x_c,
        "y_center": y_c,
        "width": w,
        "height": h
    })

# === Group and write labels ===
labels_by_image = defaultdict(list)
for item in to_enter:
    labels_by_image[item["image_id"]].append(item)

for image_id, annotations in labels_by_image.items():
    lines = []
    for ann in annotations:
        line = f"{ann['class_index']} {ann['x_center']} {ann['y_center']} {ann['width']} {ann['height']}"
        lines.append(line)
    file_path = os.path.join(labels_dir, f"{image_id}.txt")
    with open(file_path, "w") as f:
        f.write("\n".join(lines))

print(f"✅ Labels generated for {len(labels_by_image)} images using filtered_class_names.txt.")
