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
struct Frames_Detection
{
    int detectionTime_ms{-1};
    std::vector<DetectedObject> detections;
};

class Detector : public QObject
{
public:
    explicit Detector(QObject *parent = nullptr);

    virtual bool LoadModel(std::string &modelPath) = 0;
    virtual Frames_Detection detect(cv::Mat &srcImg) = 0;
    virtual Frames_Detection detect(cv::cuda::GpuMat &srcImg) = 0;

    void setClassNames(std::vector<std::string> newClassNamesList);
    std::vector<std::string> _classNamesList;

    void setInputSize(cv::Size newInputSize);
    cv::Size _inputSize = cv::Size(640, 640);
};
