#include "detector_tensorrt_end2end.h"

Detector_TensorRT_End2End::Detector_TensorRT_End2End(QObject *parent) : Detector{parent} {
    qDebug() << Q_FUNC_INFO;
}

bool Detector_TensorRT_End2End::LoadModel(QString &modelPath)
{
    qDebug() << Q_FUNC_INFO;
    return true;
}

BatchDetectedObject Detector_TensorRT_End2End::Run(MatVector &srcImgList)
{
    qDebug() << Q_FUNC_INFO;

    auto model_path = "/media/chiko/HDD_1/Work/Training_Scripts/YOLOv8/yolov8n.trt";
    cv::Mat img = srcImgList[0];

    return{};
}
