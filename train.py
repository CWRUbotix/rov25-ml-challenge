from ultralytics import YOLO

model = YOLO('yolov8n.pt')
#model = YOLO('runs/detect/train3/weights/last.pt')

model.train(data='config.yaml', epochs=50, imgsz=640, batch=16, device=0)
