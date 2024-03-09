from ultralytics import YOLO

# Load detection model
# model = YOLO('yolov8n.pt')

# Load segmentation model
model = YOLO('yolov8n-seg.pt')
yamlPath = '/media/chiko/HDD_1/Work/Pars_AI/Femur/Data/FemurDataset/3_femur-seg_Part3_4Cto2C.yaml'

# Train the model with CPU
# results = model.train(data=yamlPath, epochs=500, imgsz=640, device='cpu')

# Train the model with 1 GPUs
# results = model.train(data=yamlPath, epochs=500, imgsz=640, device=0)
# results = model.train(data=yamlPath, epochs=500, imgsz=640, device=0, overlap_mask=True)
results = model.train(data=yamlPath, epochs=500, imgsz=640, batch=32, device=0)

# Train the model with 2 GPUs
# results = model.train(data=yamlPath, epochs=500, imgsz=640, device=[0, 1])


# Evaluate the model's performance on the validation set
results = model.val()
