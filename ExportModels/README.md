# YOLOv8
YOLOv8 detection and segmentation sample training script in python  
#
### install ultralytics
pip install ultralytics  
#
### Export model to ONNX
1: Set modelPath on ExportModels.py  
2: Run ExportModels in python: python ExportModels.py  
#
### Export model to TensorRT
1: Set weight_pt(your pytorch trained weight) on ExportForTensorRT.py  
2: Run ExportForTensorRT in python: python3 ExportForTensorRT.py  
