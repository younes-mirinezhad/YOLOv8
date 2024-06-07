// TODO: Use cv::cuda::GpuMat
// TODO: pre-allocation

#include "detector_opencv_dnn.h"
#include <QFileInfo>
#include "spdlog/spdlog.h"

Detector_OpenCV_DNN::Detector_OpenCV_DNN(QObject *parent) : Detector{parent} {}

Detector_OpenCV_DNN::~Detector_OpenCV_DNN() {}

bool Detector_OpenCV_DNN::LoadModel(std::string &modelPath)
{
    spdlog::info("{} -----> Loading model: {}", Q_FUNC_INFO, modelPath);

    if (!(QFileInfo::exists(QString::fromStdString(modelPath))
          && QFileInfo(QString::fromStdString(modelPath)).isFile())) {
        spdlog::info("{} -----> Model path does not exist", Q_FUNC_INFO);
        return false;
    }

    try
    {
        net = cv::dnn::readNetFromONNX(modelPath);

        spdlog::info("{} -----> Inference device: CUDA", Q_FUNC_INFO);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        spdlog::info("{} ----------> Model is loaded.", Q_FUNC_INFO);

        // spdlog::info("{} -----> Inference device: CPU", Q_FUNC_INFO);
        // net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        // net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        // spdlog::info("{} ----------> Model is loaded.", Q_FUNC_INFO);
    } catch (const std::exception&) {
        spdlog::info("{} ----------> Can't load model.", Q_FUNC_INFO);
        return false;
    }
    return true;
}

Frames_Detection Detector_OpenCV_DNN::detect(cv::Mat &srcImg)
{
    cv::Mat modelInput = srcImg;

    if (letterBoxForSquare && _inputSize.width == _inputSize.height)
        modelInput = formatToSquare(modelInput);

    cv::Mat blob = cv::dnn::blobFromImage(modelInput, 1.0/255.0, _inputSize, cv::Scalar(), true);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    int rows = outputs[0].size[2];
    int dimensions = outputs[0].size[1];

    outputs[0] = outputs[0].reshape(1, dimensions);
    cv::transpose(outputs[0], outputs[0]);

    float *data = (float *)outputs[0].data;

    float x_factor = modelInput.cols / _inputSize.width;
    float y_factor = modelInput.rows / _inputSize.height;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {
        float *classes_scores = data+4;

        cv::Mat scores(1, _classNamesList.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;

        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > modelScoreThreshold)
        {
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);

            //TODO: Why all time box elemets are 0? (I guess it is Opencv problem)
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);

            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
        }

        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

    Frames_Detection detections{};
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        DetectedObject result;
        result.classID = class_ids[idx];
        result.confidence = confidences[idx];
        result.className = _classNamesList[result.classID];
        result.box = boxes[idx];

        detections.detections.push_back(result);
    }

    return detections;
}
cv::Mat Detector_OpenCV_DNN::formatToSquare(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    // int _max = MAX(col, row);
    // cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    cv::Mat result = cv::Mat::zeros(_inputSize.width, _inputSize.height, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

Frames_Detection Detector_OpenCV_DNN::detect(cv::cuda::GpuMat &srcImg)
{
    spdlog::error("{} ***** Need to fix.", Q_FUNC_INFO);
    return {};
}

