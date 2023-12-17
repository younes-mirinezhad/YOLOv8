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
TODO: Working on batchSize > 1  

### References
1: <a href="https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp">Github: UNeedCryDear</a>
1: <a href="https://github.com/triple-Mu/YOLOv8-TensorRT">Github: triple-Mu</a>
