# YOLOv8
1. Use TrainModels to train your model on custom data  
2. Use ExportModels to export trained pytorch model (model.pt) to:  
	1. Export: Normal onnx file  
	2. ExportOnnxEnd2End: End2End onnx file to use in c++ engine builder  
	3: ExportForTensorRT: Make tensorRT engine file from trained model  
3. Use TensorRT_EngineBuilder to automatically make engine file from end2end onnx file in your destination system.  
4. Use Infrence_CPP to see sample of infrence class in c++.  
