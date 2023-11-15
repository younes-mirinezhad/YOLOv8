from ultralytics import YOLO

modelPath = 'path/to/model.pt' # Custoum yolo models
model = YOLO(modelPath)

model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640) # used for c++ 

