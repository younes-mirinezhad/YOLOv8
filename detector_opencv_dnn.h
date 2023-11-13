#pragma once

#include "detector.h"
#include <opencv2/opencv.hpp>

class Detector_OpenCV_DNN : public Detector
{
public:
    explicit Detector_OpenCV_DNN(QObject *parent = nullptr);

    bool LoadModel(QString& modelPath) override;
    virtual BatchDetectedObject Run(MatVector& srcImgList) override;

private:
    cv::dnn::Net model;
};
