from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.train(data='config.yaml', epochs=50, imgsz=640, batch=16, device=0)
