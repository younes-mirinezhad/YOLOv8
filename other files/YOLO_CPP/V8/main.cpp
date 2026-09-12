#include <QCoreApplication>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "Detector/detector_tensorrt.h"
#include "EngineBuilder/enginebuilder.h"

void batch_TRT_Detector(std::string modelPath) {
    auto inputSize = cv::Size(640, 640);
    QString imgPath1= "Files/Images/000000000009.jpg";
    QString imgPath2= "Files/Images/000000000025.jpg";
    QString imgPath3= "Files/Images/000000000030.jpg";
    QString imgPath4= "Files/Images/000000000034.jpg";
    QString imgPath5= "Files/Images/000000000036.jpg";
    QString imgPath6= "Files/Images/000000000042.jpg";
    QString imgPath7= "Files/Images/000000000049.jpg";
    QString imgPath8= "Files/Images/000000000061.jpg";
    QStringList classes = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                           "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                           "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                           "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                           "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
                           "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                           "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                           "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                           "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

    auto detector = new Detector_TensorRT;
    detector->setInputSize(inputSize);
    detector->setClassNames(classes);
    detector->loadModel(modelPath);

    // I use bach 5
    std::vector<cv::Mat> imgVec;
    imgVec.push_back(cv::imread(imgPath1.toStdString()));
    imgVec.push_back(cv::imread(imgPath2.toStdString()));
    imgVec.push_back(cv::imread(imgPath3.toStdString()));
    imgVec.push_back(cv::imread(imgPath4.toStdString()));
    imgVec.push_back(cv::imread(imgPath5.toStdString()));
    // imgVec.push_back(cv::imread(imgPath6.toStdString()));
    // imgVec.push_back(cv::imread(imgPath7.toStdString()));
    // imgVec.push_back(cv::imread(imgPath8.toStdString()));

    bool showResults{true};
    for (int i = 0; i < 50; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();

        auto resDetector = detector->detect(imgVec);

        auto t2 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        qDebug() << "----------> detection time: " << dt << " (ms)";

        if(!showResults)
            continue;
        for (int idx = 0; idx < imgVec.size(); ++idx) {
            auto img = imgVec[idx].clone();
            auto det = resDetector[idx];
            for (int i = 0; i < det.size(); ++i) {
                cv::rectangle(img, det[i].box, cv::Scalar(0, 0, 255), 2, 8);
                cv::putText(img, det[i].className.toStdString(), cv::Point(det[i].box.x, det[i].box.y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
            }
            cv::imshow("Detection Box " + std::to_string(i), img);
            cv::waitKey(0);
            cv::destroyAllWindows();
            img.release();
        }
    }
}
void build_ONNX_to_TRT_Engine(std::string modelPath_e2eONNX, std::string modelPath_e2eEngine) {
    EngineBuilder engineBuilder(modelPath_e2eONNX, modelPath_e2eEngine);
    engineBuilder.buildEngine();
}
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    std::string modelPath_e2eONNX = "Files/Models/Detection/yolov8n_b5_640_end2end.onnx";
    std::string modelPath_e2eEngine = "Files/Models/Detection/yolov8n_b5_640_end2end.engine";

    build_ONNX_to_TRT_Engine(modelPath_e2eONNX, modelPath_e2eEngine);
    batch_TRT_Detector(modelPath_e2eEngine);

    return a.exec();
}
