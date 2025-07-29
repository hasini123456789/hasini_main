import pandas as pd
import os
import cv2

index=12
with open("filtered_class_names.txt", "r") as f:
    filtered_class_names = [line.strip() for line in f if line.strip()]

print(filtered_class_names[index])






def draw_yolo_bboxes(image_path, label_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    with open(label_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())

            # Convert normalized coordinates to pixel values
            x_center *= w
            y_center *= h
            width *= w
            height *= h

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, str(int(class_id)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

label_path="/Users/hasini/Downloads/MAIN/dataset/train/labels/00a70c466ad1fe24.txt"
image_path="/Users/hasini/Downloads/MAIN/dataset/train/images/00a70c466ad1fe24.jpg"
draw_yolo_bboxes(image_path,label_path)