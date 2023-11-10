#include "segmentor_opencv_dnn.h"
#include <QFileInfo>

#if CUDA_STATUS
#define CUDA_Availability true
#else
#define CUDA_Availability false
#endif

Segmentor_OpenCV_DNN::Segmentor_OpenCV_DNN(QObject *parent) : Segmentor{parent} {
    qDebug() << Q_FUNC_INFO;
}

bool Segmentor_OpenCV_DNN::LoadModel(QString &modelPath)
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

BatchSegmentedObject Segmentor_OpenCV_DNN::Run(MatVector &srcImgList)
{
    qDebug() << Q_FUNC_INFO;

    // TODO: just work with bachNumber=1
    if(_batchSize > 1) {
        qDebug() <<"This class just work with batchNumber=1";
        return {};
    }

    BatchSegmentedObject batchOutput;

    auto srcImg = srcImgList[0];
    cv::Mat netInputImg;
    cv::Vec4d params;
    LetterBox(srcImg, netInputImg, params, _inputSize);

    cv::Mat blob;
    cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, _inputSize, cv::Scalar(0, 0, 0), true, false);

    return batchOutput;
}

void Segmentor_OpenCV_DNN::LetterBox(const cv::Mat &image, cv::Mat &outImage, cv::Vec4d &params, const cv::Size &newShape, bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar &color)
{
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{ r, r };
    int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

    auto dw = (float)(newShape.width - new_un_pad[0]);
    auto dh = (float)(newShape.height - new_un_pad[1]);

    if (autoShape) {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill) {
        dw = 0.0f;
        dh = 0.0f;
        new_un_pad[0] = newShape.width;
        new_un_pad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1]) {
        cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
    }
    else {
        outImage = image.clone();
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    params[0] = ratio[0];
    params[1] = ratio[1];
    params[2] = left;
    params[3] = top;

    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}
