#ifndef DETECTOR_TENSORRT_H
#define DETECTOR_TENSORRT_H

#include "detector.h"
#include <NvInferRuntime.h>

struct Binding {
    size_t size = 1;
    size_t dsize = 1;
    nvinfer1::Dims dims;
    char* name;
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

    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO): reportableSeverity(severity) { }

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

class Detector_TensorRT : public Detector
{
public:
    explicit Detector_TensorRT(QObject *parent = nullptr);
    ~Detector_TensorRT();

    bool LoadModel(QString &modelPath) override;
    ImagesDetectedObject detect(cv::Mat &srcImg) override;
    ImagesDetectedObject detect(cv::cuda::GpuMat &srcImg) override;

private:
    void copy_from_Mat(cv::Mat &img, cv::Size &size);
    void letterbox(cv::Mat &img_input, cv::Size &size);
    void copy_from_Mat(cv::cuda::GpuMat &gImg, cv::Size &size);
    void letterbox(cv::cuda::GpuMat& gImg_input, cv::Size &size);
    void blobFromGpuMat(const cv::cuda::GpuMat& gImg_input, const std::array<float, 3>& std,
                        const std::array<float, 3>& mean, bool swapBR, bool normalize);
    ImagesDetectedObject postprocess();

    nvinfer1::IExecutionContext* context{nullptr};
    cudaStream_t stream{nullptr};
    int num_inputs{0}, num_outputs{0};
    std::vector<Binding> input_bindings, output_bindings;
    std::vector<void*> device_ptrs, host_ptrs;
    PreParam pparam;
    cv::cuda::GpuMat _gBlob;
    cv::Mat _blob;
};

#endif // DETECTOR_TENSORRT_H
