#pragma once

#include "detector.h"
#include "NvInfer.h"

struct Binding {
    size_t size = 1;
    size_t dsize = 1;
    nvinfer1::Dims dims;
    std::string name;
};
struct PreParam {
    float ratio  = 1.0f;
    float dw     = 0.0f;
    float dh     = 0.0f;
    float height = 0;
    float width  = 0;
};

class Logger: public nvinfer1::ILogger {
public:
    nvinfer1::ILogger::Severity reportableSeverity;

    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO):
        reportableSeverity(severity)
    {
    }

    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity > reportableSeverity) {
            return;
        }
        switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};

class Detector_TensorRT_End2End : public Detector
{
public:
    explicit Detector_TensorRT_End2End(QObject *parent = nullptr);
    ~Detector_TensorRT_End2End();

    bool LoadModel(QString& modelPath) override;
    BatchDetectedObject Run(MatVector& srcImgList) override;

private:
    void copy_from_Mat(const cv::Mat& image);
    void copy_from_Mat(const cv::Mat& image, cv::Size& size);
    void letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    ImagesDetectedObject postprocess();

    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine  = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream  = nullptr;
    int num_bindings;
    int num_inputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    int num_outputs = 0;
    std::vector<void*> device_ptrs;
    std::vector<void*> host_ptrs;
    PreParam pparam;
};

