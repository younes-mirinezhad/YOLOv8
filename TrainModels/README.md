# YOLOv8
Detection and Segmentation sample training script in python  
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
---segmentation sample object : <classID> <x1> <y1> <x2> <y2> ... <xn> <yn>  
#
### Start training
1: Set Dataset Path and class names in Yaml file  
2: Set modelPath and yamlPath on trainYOLOv8.py  
3: Run trainYOLOv8 in python: python trainYOLOv8.py  
Your output model is in run folder  
 
