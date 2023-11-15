#pragma once
#include "segmentor.h"
#include <opencv2/opencv.hpp>

class Segmentor_OpenCV_DNN : public Segmentor
{
public:
    explicit Segmentor_OpenCV_DNN(QObject *parent = nullptr);

    bool LoadModel(QString& modelPath) override;
    BatchSegmentedObject Run(MatVector& srcImgList) override;

private:
    cv::dnn::Net model;
    float _classThreshold = 0.25;
    float _nmsThreshold = 0.45;
    float _maskThreshold = 0.5;

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
};
