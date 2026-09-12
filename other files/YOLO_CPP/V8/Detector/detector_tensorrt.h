#ifndef DETECTOR_TENSORRT_H
#define DETECTOR_TENSORRT_H

#include <QObject>
#include <NvInferRuntime.h>
#include <opencv2/core/cuda.hpp>

struct Binding {
    size_t size = 1;
    size_t dsize = 1;
    nvinfer1::Dims dims;
    const char* name;
};

struct PreParam {
    float ratio  = 1.0f;
    float dw     = 0.0f;
    float dh     = 0.0f;
    float height = 0;
    float width  = 0;
};

struct DetectedObject {
    int classID;
    QString className;
    float confidence;
    cv::Rect box;
};

using Severity = nvinfer1::ILogger::Severity;
class Logger : public nvinfer1::ILogger {
public:
    void setLogSeverity(Severity severity);
private:
    void log(Severity severity, const char *msg) noexcept override;
    Severity m_severity = Severity::kINFO;
};

class Detector_TensorRT
{
public:
    Detector_TensorRT();
    ~Detector_TensorRT();

    void setInputSize(cv::Size newInputSize);
    void setClassNames(QStringList newClassNames);
    bool loadModel(std::string& modelPath);
    QList<QList<DetectedObject>> detect(std::vector<cv::Mat> &src_imgs);

private:
    cv::cuda::GpuMat preprocess(std::vector<cv::cuda::GpuMat> &src_imgs);
    cv::cuda::GpuMat blob_from_GpuMats(std::vector<cv::cuda::GpuMat> &batchInput);
    void inference(cv::cuda::GpuMat &model_input);
    QList<QList<DetectedObject>> postprocess();

    QStringList _classNames{};
    cv::Size _inputSize = cv::Size(960, 960);
    bool _modelIsLoaded = false;
    bool use_gpumat{true};
    bool add_pad = false;
    int top_pad;
    int left_pad;
    QList<PreParam> pparams;

    nvinfer1::IExecutionContext* context{nullptr};
    cudaStream_t stream{nullptr};
    int num_inputs{0};
    int num_outputs{0};
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*> device_ptrs;
    std::vector<void*> host_ptrs;
    int _batch{0};
};

#endif // DETECTOR_TENSORRT_H







