#pragma once

#include "segmentor.h"
#include<onnxruntime_cxx_api.h>

class Segmentor_ONNXRUNTIME : public Segmentor
{
public:
    explicit Segmentor_ONNXRUNTIME(QObject *parent = nullptr);

    bool LoadModel(QString& modelPath) override;
    BatchSegmentedObject Run(MatVector& srcImgList) override;

private:
    void Preprocessing(const std::vector<cv::Mat>& SrcImgs, std::vector<cv::Mat>& OutSrcImgs, std::vector<cv::Vec4d>& params);
    void LetterBox(const cv::Mat& image,
                   cv::Mat& outImage,
                   cv::Vec4d& params,
                   const cv::Size& newShape = cv::Size(640, 640),
                   bool autoShape = false,
                   bool scaleFill = false,
                   bool scaleUp = true,
                   int stride = 32,
                   const cv::Scalar& color = cv::Scalar(114, 114, 114));
    void GetMask(const cv::Mat& maskProposals,
                 const cv::Mat& maskProtos,
                 ImagesSegmentedObject& output,
                 const MaskParams& maskParams);
    void GetMask2(const cv::Mat& maskProposals,
                  const cv::Mat& maskProtos,
                  SegmentedObject& output,
                  const MaskParams& maskParams);
    void calcContours(ImagesSegmentedObject& output);
    template <typename T>
    T VectorProduct(const std::vector<T>& v)
    {
        return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
    };
    int _cudaID = 0;
    float _classThreshold = 0.25;
    float _nmsThreshold = 0.45;
    float _maskThreshold = 0.5;
    Ort::Session* _OrtSession = nullptr;
    Ort::Env _OrtEnv = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Yolov5-Seg");
    std::shared_ptr<char> _inputName, _output_name0, _output_name1;
    std::vector<char*> _inputNodeNames, _outputNodeNames;
    std::vector<int64_t> _inputTensorShape, _outputTensorShape, _outputMaskTensorShape;
    bool _isDynamicShape = false; //onnx support dynamic shape
    Ort::MemoryInfo _OrtMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPUOutput);
};
