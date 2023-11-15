from ultralytics import YOLO

#Load detection model
#It can auto download models if you dont have it
#modelPath = 'yolov8n.pt' # detection models (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
#modelPath = 'yolov8n-seg.pt' # segmentation models (yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt)
modelPath = 'path/to/model.pt' # Custoum yolo models

model = YOLO(modelPath)

yamlPath = '/path/t0/coco8-seg.yaml'

# Train the model with CPU
# results = model.train(data=yamlPath, epochs=500, imgsz=640, device='cpu')

# Train the model with 1 GPUs
results = model.train(data=yamlPath, epochs=500, imgsz=640, device=0)

# Train the model with 2 GPUs
# results = model.train(data=yamlPath, epochs=500, imgsz=640, device=[0, 1])


# Evaluate the model's performance on the validation set
results = model.val()
