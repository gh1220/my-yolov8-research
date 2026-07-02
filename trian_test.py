from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")

backbone = model.model.model
expected_rcc_layers = [0, 1, 3, 5, 7]
actual_layers = {i: backbone[i].__class__.__name__ for i in expected_rcc_layers}
print(actual_layers)
assert all(backbone[i].__class__.__name__ == "RCC" for i in expected_rcc_layers), "Backbone is not using RCC"

model.train(
    data="../datasets/data.yaml",
    device="cpu",
    imgsz=416,
    epochs=50,
    batch=1
)