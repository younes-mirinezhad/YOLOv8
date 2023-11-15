#pragma once
#include <QObject>
#include "opencv2/core/mat.hpp"
#include <QDebug>

using MatVector = std::vector<cv::Mat>;
struct SegmentedObject {
    int classID;
    float confidence;
    cv::Rect box;
    cv::Mat boxMask;
    std::vector<std::vector<cv::Point>> maskContoursList;
};
using ImagesSegmentedObject = std::vector<SegmentedObject>;
using BatchSegmentedObject = std::vector<ImagesSegmentedObject>;

struct MaskParams {
    //int segChannels = 32;
    //int segWidth = 160;
    //int segHeight = 160;
    int netWidth = 640;
    int netHeight = 640;
    float maskThreshold = 0.5;
    cv::Size srcImgShape;
    cv::Vec4d params;
};

class Segmentor : public QObject
{
public:
    explicit Segmentor(QObject *parent = nullptr);

    virtual bool LoadModel(QString& modelPath) = 0;
    virtual BatchSegmentedObject Run(MatVector& srcImgList) = 0;

    std::vector<std::string> _classNamesList;
    void setClassNames(std::vector<std::string> newClassNamesList);

    int _batchSize = 1;
    void setBatchSize(int newBatch);

    cv::Size _inputSize = cv::Size(640, 640);
    void setInputSize(cv::Size newInputSize);
};
