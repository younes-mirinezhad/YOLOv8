# YOLOv8
YOLOv8 detection and segmentation sample in C++  
#
### YOLOv8 Detection OpenCV DNN
Detection based on "OpenCV DNN".  
----- Only need to OpenCV  
----- Output contains:  
---------- Detected object ClassID(int)  
---------- Detected object Confidence(float)  
---------- Detected object box(cv::Rect)  
TODO: Working on batchSize > 1  
#
### YOLOv8 Segmentation OpenCV DNN
Segmentation based on "OpenCV DNN".  
----- Only need to OpenCV  
----- Output contains:  
---------- Detected object ClassID(int)  
---------- Detected object Confidence(float)  
---------- Detected object box(cv::Rect)  
---------- Mask of detected box(cv::Mat)  
---------- Contours of detected mask(std::vector<std::vector<cv::Point>>)  
TODO: Working on batchSize > 1  
#
#
### References
1: <a href="https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp">yolov8-opencv-onnxruntime-cpp</a>
