import yaml

# Read class names from the filtered_class_names.txt file
with open("filtered_class_names.txt", "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

data = {
    'train': 'dataset_split/train/images',
    'val': 'dataset_split/val/images',
    'nc': len(class_names),
    'names': class_names
}

with open("data.yaml", "w") as f:
    yaml.dump(data, f, default_flow_style=False)

print(f"✅ data.yaml created with {len(class_names)} classes.")

