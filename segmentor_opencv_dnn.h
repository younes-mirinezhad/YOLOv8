#pragma once
#include "segmentor.h"
#include <opencv2/opencv.hpp>

struct MaskParams {
    int netWidth = 640;
    int netHeight = 640;
    float maskThreshold = 0.5;
    cv::Size srcImgShape;
    cv::Vec4d params;
};

class Segmentor_OpenCV_DNN : public Segmentor
{
public:
    explicit Segmentor_OpenCV_DNN(QObject *parent = nullptr);

    bool LoadModel(QString& modelPath) override;
    virtual BatchSegmentedObject Run(MatVector& srcImgList) override;

private:
    cv::dnn::Net model;

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
