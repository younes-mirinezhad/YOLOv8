from ultralytics import YOLO

model = YOLO('path/to/model.pt')

# model.export(format="onnx") # used for python
model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640) # used for c++ 
# model.export(format='torchscript')
