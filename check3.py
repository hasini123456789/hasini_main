import os

# Path to label files folder
labels_dir = "dataset/train/labels"

# Load data.yaml info
import yaml
with open("data.yaml") as f:
    data = yaml.safe_load(f)

num_classes = data['nc']

# Check all label files
invalid_files = []
for fname in os.listdir(labels_dir):
    if fname.endswith(".txt"):
        with open(os.path.join(labels_dir, fname)) as f:
            for line in f:
                if line.strip():
                    cls_index = int(line.split()[0])
                    if cls_index < 0 or cls_index >= num_classes:
                        invalid_files.append((fname, cls_index))

if invalid_files:
    print("⚠️ Found label files with invalid class indices:")
    for fname, cls_idx in invalid_files:
        print(f"  - {fname}: class index {cls_idx} out of range (0 to {num_classes - 1})")
else:
    print("✅ All label class indices are compatible with data.yaml")
