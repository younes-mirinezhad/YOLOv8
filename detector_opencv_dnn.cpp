#include "detector_opencv_dnn.h"
#include <QFileInfo>

#if CUDA_STATUS
#define CUDA_Availability true
#else
#define CUDA_Availability false
#endif

Detector_OpenCV_DNN::Detector_OpenCV_DNN(QObject *parent) : Detector{parent} {
    qDebug() << Q_FUNC_INFO;
}

bool Detector_OpenCV_DNN::LoadModel(QString &modelPath)
{
    qDebug() << Q_FUNC_INFO << modelPath;

#if CUDA_Availability
    qDebug() << "Founded CUDA device info";
    int cuda_devices_count = cv::cuda::getCudaEnabledDeviceCount();
    for (int dev = 0; dev < cuda_devices_count; ++dev) {
        cv::cuda::printCudaDeviceInfo(dev);
        qDebug() << " -------------------------------------------------- ";
    }
#endif

    try
    {
        if (!(QFileInfo::exists(modelPath) && QFileInfo(modelPath).isFile())) {
            qDebug() << "----- Model path does not exist,  please check " << modelPath;
            return false;
        }
        model = cv::dnn::readNet(modelPath.toStdString());

#if CV_VERSION_MAJOR==4 && CV_VERSION_MINOR==7 && CV_VERSION_REVISION==0
        model.enableWinograd(false); //bug of opencv4.7.x in AVX only platform ,https://github.com/opencv/opencv/pull/23112 and https://github.com/opencv/opencv/issues/23080
        model.enableWinograd(true); //If your CPU supports AVX2, you can set it true to speed up
#endif

#if CUDA_Availability
        qDebug() << "----- Inference device: CUDA";
        model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA); //DNN_TARGET_CUDA or DNN_TARGET_CUDA_FP16
        qDebug() << "---------- Model is loaded ";
#else
        qDebug() << "----- Inference device: CPU";
        model.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        qDebug() << "---------- Model is loaded ";
#endif
    }
    catch (const std::exception&) {
        qDebug() << "----- Can't load model:" << modelPath;
        return false;
    }
    return true;
}

BatchDetectedObject Detector_OpenCV_DNN::Run(MatVector &srcImgList)
{
    qDebug() << Q_FUNC_INFO;

    // TODO: just work with bachNumber=1
    if(_batchSize > 1 || srcImgList.size() > 1) {
        qDebug() <<"This class just work with batchNumber=1";
        return {};
    }

    BatchDetectedObject batchOutput;
    ImagesDetectedObject imageOutput;

    return batchOutput;
}
