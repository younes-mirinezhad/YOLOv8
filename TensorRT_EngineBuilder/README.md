# TensorRT_EngineBuilder
Make TensorRT engine file from an End2End Onnx file  

add EngineBuilder folder to your project (check its dependencies in CmakeList.txt)  
1: #include "EngineBuilder/enginebuilder.h"  
2: EngineBuilder engineBuilder("path/to/sample_End2End.onnx", "path/to/sample_End2End.engine");  
3: engineBuilder.buildEngine();  
