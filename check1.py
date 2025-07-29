import pandas as pd

# Load files
classname_file = "class-descriptions-boxable.csv"
annotations_file = "oidv6-train-annotations-bbox.csv"

# Load class descriptions
classes_df = pd.read_csv(classname_file, header=None, names=['LabelName', 'ClassName'])

# Load bounding box annotations (only needed column)
annots_df = pd.read_csv(annotations_file, usecols=["LabelName"])

# Get labelnames that actually have images (i.e. appear in annotation file)
labelnames_with_images = set(annots_df["LabelName"])

# Filter classes to only those with image IDs
filtered_df = classes_df[classes_df["LabelName"].isin(labelnames_with_images)]

# Get final filtered class names
filtered_class_names = filtered_df["ClassName"].tolist()

# Optional: Save to check
with open("filtered_class_names.txt", "w") as f:
    f.write("\n".join(filtered_class_names))

print(f"✅ Found {len(filtered_class_names)} classes with bounding boxes.")


