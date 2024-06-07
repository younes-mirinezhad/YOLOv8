# YOLOv8
Export models in python  
#
### install ultralytics
pip install ultralytics  
#
### Export.py
1: Set modelPath on Export.py  
2: Run Export in python: python3 Export.py  
#
### ExportOnnxEnd2End.py
1: Set pytorch model path (weight_pt="path/to/model.pt")  
2: Set model input shape (input_shape=[1, 3, 640, 640])  
3: Set model CONF threshoud (conf_thres = 0.25)    
4: Set model IOU threshoud (iou_thres = 0.65)  
5: Run Export in python: python3 ExportOnnxEnd2End.py  
#
### ExportForTensorRT.py
1: Set pytorch model path (weight_pt="path/to/model.pt")  
2: Set model input shape (input_shape=[1, 3, 640, 640])  
3: Set model CONF threshoud (conf_thres = 0.25)    
4: Set model IOU threshoud (iou_thres = 0.65)  
5: Run Export in python: python3 ExportForTensorRT.py  
