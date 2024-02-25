#pragma once
#include <QObject>
#include <opencv2/opencv.hpp>
#include <QDebug>

struct DetectedObject {
    int classID;
    std::string className;
    float confidence;
    cv::Rect box;
};
using ImagesDetectedObject = std::vector<DetectedObject>;
class Detector : public QObject
{
public:
    explicit Detector(QObject *parent = nullptr);

    virtual bool LoadModel(QString &modelPath) = 0;
    virtual ImagesDetectedObject detect(cv::Mat &srcImg) = 0;
    virtual ImagesDetectedObject detect(cv::cuda::GpuMat &srcImg) = 0;

    std::vector<std::string> _classNamesList;
    void setClassNames(std::vector<std::string> newClassNamesList);

    cv::Size _inputSize = cv::Size(960, 960);
    void setInputSize(cv::Size newInputSize);
};
