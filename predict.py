from ultralytics import YOLO
from pathlib import Path

model = YOLO('runs/detect/train/weights/last.pt')

paths = tuple(Path('../dataset/images/val').iterdir())

print(paths)

results = model(paths)

for path, result in zip(paths, results, strict=True):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen
    result.save(filename=f'predictions/{path.stem}.jpg')  # save to disk
