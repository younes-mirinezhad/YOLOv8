# YOLOv8
YOLOv8 detection and segmentation sample in C++  

#
Build OpenCV: https://github.com/younes-mirinezhad/Scrips/tree/main/OpenCV  
Download ONNX Runtime: https://github.com/microsoft/onnxruntime  
Download TensorRT: https://developer.nvidia.com/tensorrt-download  
convert ONNX to TensorRT: trtexec --onnx=model.onnx --saveEngine=engine.trt  

#
## Detection  
----- Based on OpenCV DNN  
----- Based on ONNX Runtime  

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
1: <a href="https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp">yolov8-opencv-onnxruntime-cpp</a>
