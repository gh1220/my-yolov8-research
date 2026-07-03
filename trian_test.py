from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")

backbone = model.model.model
expected_rcc_layers = [0, 1, 3, 5, 7]
expected_cfc_layers = [2, 4, 6, 8]
expected_sba_layers = [12, 16, 20, 24]
actual_layers = {i: backbone[i].__class__.__name__ for i in expected_rcc_layers + expected_cfc_layers + expected_sba_layers}
print(actual_layers)
assert all(backbone[i].__class__.__name__ == "RCC" for i in expected_rcc_layers), "Backbone is not using RCC"

def _is_cfc_stage(module):
    return module.__class__.__name__ == "CFC" or (
        module.__class__.__name__ == "Sequential" and all(child.__class__.__name__ == "CFC" for child in module)
    )


assert all(_is_cfc_stage(backbone[i]) for i in expected_cfc_layers), "Backbone is not using CFC"


def _is_sba(module):
    return module.__class__.__name__ == "SBA"


assert all(_is_sba(backbone[i]) for i in expected_sba_layers), "Neck is not using SBA"

model.train(
    data="../datasets/data.yaml",
    device="cpu",
    imgsz=640,
    epochs=100,
    batch=1
)