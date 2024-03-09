# YOLOv8
Detection sample in C++  

#
Select detector type (TensorRT in line 11 ro OpenCV_DNN in line 14) and set detectorModelPath in line 10 or 13  
Set class names in line 23  
Set image path in line 39  

In ProcessingThread, You can choose one of cv::Mat (line 42) or cv::cuda::GpuMat (line 45)

Note: OpenCV_DNN detection has problem. In detections box all of x,y,w,h are 0  
