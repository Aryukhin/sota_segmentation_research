yaml_content = """
# Ultralytics YOLO 🚀, AGPL-3.0 license
# Lyft Segmentation dataset
# Documentation: https://docs.ultralytics.com/datasets/segment/
# Example usage: yolo train data=lyft_dataset.yaml

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: /Users/sasa/PycharmProjects/pythonProject/lyft_segmentation  # dataset root dir
train: /Users/sasa/PycharmProjects/pythonProject/lyft_segmentation/train/images      # train images (relative to 'path')
val: /Users/sasa/PycharmProjects/pythonProject/lyft_segmentation/val/images             # val images (relative to 'path')
test:                       # test images (optional)

# Classes
names:
  0: Unlabeled
  1: Building
  2: Fence
  3: Other
  4: Pedestrian
  5: Pole
  6: Roadline
  7: Road
  8: Sidewalk
  9: Vegetation
  10: Car
  11: Wall
  12: Traffic sign
"""

with open("lyft_segmentation/lyft_dataset.yaml", "w") as f:
    f.write(yaml_content)
