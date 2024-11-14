
import cv2
import numpy as np
import model
import utility as su


# Load a model
y8 = model.YoloV8('weights/yolov8l.pt')  # load an official model
colors = su.generate_distinct_colors(len(y8.class_names.keys()))

img_path = "zidane.jpg"
batch_regions = y8.process(img_path)
img = cv2.imread(img_path)

for regions in batch_regions[0]:
    for region in regions:
        label_idx = y8.class_idx_by_name[region.label]
        color = colors[label_idx]
        img = su.draw_predict(region, img, color, bbox=True)

cv2.imshow("0", img)
key = cv2.waitKey(0)
cv2.destroyWindow("0")






