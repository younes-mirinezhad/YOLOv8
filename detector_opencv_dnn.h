#pragma once

#include "detector.h"
#include <opencv2/opencv.hpp>

class Detector_OpenCV_DNN : public Detector
{
public:
    explicit Detector_OpenCV_DNN(QObject *parent = nullptr);

    bool LoadModel(QString& modelPath) override;
    BatchDetectedObject Run(MatVector& srcImgList) override;

private:
    cv::dnn::Net model;
    float _classThreshold = 0.25;
    float _nmsThreshold = 0.45;

    void LetterBox(const cv::Mat& image,
                   cv::Mat& outImage,
                   cv::Vec4d& params,
                   const cv::Size& newShape = cv::Size(640, 640),
                   bool autoShape = false,
                   bool scaleFill = false,
                   bool scaleUp = true,
                   int stride = 32,
                   const cv::Scalar& color = cv::Scalar(114, 114, 114));
};
