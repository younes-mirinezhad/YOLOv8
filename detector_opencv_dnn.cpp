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
//        model.enableWinograd(true); //If your CPU supports AVX2, you can set it true to speed up
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

    auto srcImg = srcImgList[0];
    cv::Mat netInputImg;
    cv::Vec4d params;
    LetterBox(srcImg, netInputImg, params, _inputSize);

    cv::Mat blob;
    cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, _inputSize, cv::Scalar(0, 0, 0), true, false);
    //************************************
    // If there is no problem with other settings, but results are a lot different from  Python-onnx , you can try to use the following two sentences
    //cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, _inputSize, cv::Scalar(104, 117, 123), true, false);
    //cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, _inputSize, cv::Scalar(114, 114,114), true, false);
    //************************************
    model.setInput(blob);

    std::vector<cv::Mat> net_output_img;
    model.forward(net_output_img, model.getUnconnectedOutLayersNames()); //get outputs

    std::vector<int> class_ids;// res-class_id
    std::vector<float> confidences;// res-conf
    std::vector<cv::Rect> boxes;// res-box
    cv::Mat output0=cv::Mat(cv::Size(net_output_img[0].size[2], net_output_img[0].size[1]), CV_32F, (float*)net_output_img[0].data).t();  //[bs,116,8400]=>[bs,8400,116]
    int net_width = output0.cols;
    int rows = output0.rows;
    float* pdata = (float*)output0.data;

    for (int r = 0; r < rows; ++r) {
        cv::Mat scores(1, _classNamesList.size(), CV_32FC1, pdata + 4);
        cv::Point classIdPoint;
        double max_class_socre;
        minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
        max_class_socre = (float)max_class_socre;
        if (max_class_socre >= _classThreshold) {
            //rect [x,y,w,h]
            float x = (pdata[0] - params[2]) / params[0];
            float y = (pdata[1] - params[3]) / params[1];
            float w = pdata[2] / params[0];
            float h = pdata[3] / params[1];
            int left = MAX(int(x - 0.5 * w + 0.5), 0);
            int top = MAX(int(y - 0.5 * h + 0.5), 0);
            class_ids.push_back(classIdPoint.x);
            confidences.push_back(max_class_socre);
            boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
        }
        pdata += net_width;//next line
    }
    //NMS
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
    std::vector<std::vector<float>> temp_mask_proposals;
    cv::Rect holeImgRect(0, 0, srcImg.cols, srcImg.rows);
    for (int i = 0; i < nms_result.size(); ++i) {
        int idx = nms_result[i];
        DetectedObject result;
        result.classID = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx] & holeImgRect;
        imageOutput.push_back(result);
    }

    batchOutput.push_back(imageOutput);
    return batchOutput;
}

void Detector_OpenCV_DNN::LetterBox(const cv::Mat &image, cv::Mat &outImage, cv::Vec4d &params, const cv::Size &newShape, bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar &color)
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
