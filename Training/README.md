# YOLOv8
YOLOv8 detection and segmentation sample training script in python  
#
### install ultralytics
pip install ultralytics  
#
### Make dataset
I put sample dataset structure in training folder that contain both detection and segmentation sample annotation.  
One *.txt file per image.  
If there are no objects in an image, no *.txt file is required.  
One row per object:  
All x and y must be normalized (from 0 to 1)  
---detection sample object : <classID> <x_center> <y_center> <width> <height>
---detection sample object : <classID> <x1> <y1> <x2> <y2> ... <xn> <yn>
#
### Start training
1: Set Dataset Path and class names in Yaml file  
2: Set modelPath and yamlPath on trainYOLOv8.py  
3: Run trainYOLOv8 in python: python trainYOLOv8.py  
Your output model is in run folder
#
### Export model to ONNX
1: Set modelPath on ExportModels.py  
2: Run ExportModels in python: python ExportModels.py  
#
### Export model to TensorRT
1: Set weight_pt(your pytorch trained weight) on ExportForTensorRT.py  
2: Run ExportForTensorRT in python: python3 ExportForTensorRT.py  
