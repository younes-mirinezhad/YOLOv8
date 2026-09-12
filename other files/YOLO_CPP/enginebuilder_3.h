#ifndef ENGINEBUILDER_3_H
#define ENGINEBUILDER_3_H

#include <string>
#include <NvInfer.h>
#include <iostream>
#include <cassert>
#include <memory>

class EngineBuilder_3
{
public:
    EngineBuilder_3();
    bool build(std::string onnxModelPath, std::string enginePath);
    std::unique_ptr<nvinfer1::INetworkDefinition> network;
    void addPlugin();
    int _backGroundClass{-1};
    int _maxNumOfDetection{100};
    int _scoreActivation{0};
    int _boxCoding{1}; // 0 : (x0,y0) , (x1,y1) ---- 1: cx, cy, w, h
    float _confidenceThreshold{0.2};
    float _iouThreshold{0.4};
    std::string getTensorShape(nvinfer1::ITensor *tensor);
};

class Logger_3 : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "***** [kINTERNAL_ERROR] [TRT] " << msg << std::endl;
                break;
            case Severity::kERROR:
                std::cerr << "***** [kERROR] [TRT] " << msg << std::endl;
                break;
            case Severity::kWARNING:
                std::cout << "+++++ [kWARNING] [TRT] " << msg << std::endl;
                break;
            case Severity::kINFO:
                std::cout << "----- [kINFO] [TRT] " << msg << std::endl;
                break;
            case Severity::kVERBOSE:
                std::cout << "[kVERBOSE] [TRT] " << msg << std::endl;
                break;
            default:
                assert(false && "Invalid log level");
                break;
            }
        }
    }
};

#endif // ENGINEBUILDER_3_H
