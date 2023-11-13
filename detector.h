#pragma once
#include <QObject>
#include "opencv2/core/types.hpp"
#include <QDebug>

using MatVector = std::vector<cv::Mat>;
struct DetectedObject {
    int classID;
    float confidence;
    cv::Rect box;
};
using ImagesDetectedObject = std::vector<DetectedObject>;
using BatchDetectedObject = std::vector<ImagesDetectedObject>;
class Detector : public QObject
{
public:
    explicit Detector(QObject *parent = nullptr);

    virtual bool LoadModel(QString& modelPath) = 0;
    virtual BatchDetectedObject Run(MatVector& srcImgList) = 0;

    std::vector<std::string> _classNamesList;
    void setClassNames(std::vector<std::string> newClassNamesList);

    int _batchSize = 1;
    void setBatchSize(int newBatch);

    cv::Size _inputSize = cv::Size(640, 640);
    void setInputSize(cv::Size newInputSize);
};
