# YOLOv8
YOLOv8 detection and segmentation sample in C++  

#
Build OpenCV: https://github.com/younes-mirinezhad/Scrips/tree/main/OpenCV  
Download ONNX Runtime: https://github.com/microsoft/onnxruntime  
Install TensorRT: https://github.com/younes-mirinezhad/Scrips/tree/main/TensorRT  

#
## Detection  
----- Based on OpenCV DNN  
----- Based on ONNX Runtime  
----- Based on TensorRT  

----- Output:  
---------- Detected object ClassID(int)  
---------- Detected object Confidence(float)  
---------- Detected object box(cv::Rect)  

#
## Segmentation  
----- Based on OpenCV DNN  
----- Based on ONNX Runtime  

----- Output:  
---------- Detected object ClassID(int)  
---------- Detected object Confidence(float)  
---------- Detected object box(cv::Rect)  
---------- Mask of detected box(cv::Mat)  
---------- Contours of detected mask(std::vector<std::vector<cv::Point>>)  

#
## Train and export models  
https://github.com/younes-mirinezhad/YOLOv8/tree/main/Training  

