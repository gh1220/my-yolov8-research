from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="../datasets/data.yaml",
    device="cpu",
    imgsz=416,
    epochs=50,
    batch=1
)