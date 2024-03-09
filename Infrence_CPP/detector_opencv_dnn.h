#ifndef DETECTOR_OPENCV_DNN_H
#define DETECTOR_OPENCV_DNN_H

#include "detector.h"

class Detector_OpenCV_DNN : public Detector
{
public:
    explicit Detector_OpenCV_DNN(QObject *parent = nullptr);
    ~Detector_OpenCV_DNN();

    bool LoadModel(QString &modelPath) override;
    ImagesDetectedObject detect(cv::Mat &srcImg) override;
    ImagesDetectedObject detect(cv::cuda::GpuMat &srcImg) override;

private:
    cv::dnn::Net net;
    bool letterBoxForSquare = true;
    cv::Mat formatToSquare(const cv::Mat &source);
    float modelScoreThreshold{0.45};
    float modelNMSThreshold{0.50};
};

#endif // DETECTOR_OPENCV_DNN_H
