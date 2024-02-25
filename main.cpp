#include <QCoreApplication>
#include "processingthread.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    auto _imgPath = "/media/chiko/HDD_1/Work/Training_Scripts/CocoImage/img.jpeg";
    auto inputSize = cv::Size(640, 640);
    QString detectorModelPath = "/media/chiko/HDD_1/Work/Training_Scripts/YOLOv8/Weights/Detection/TensorRT/yolov8n_640.engine";

    std::vector<std::string> classNamesList = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    ProcessingThread *_processor = new ProcessingThread;
    auto detectorStatus = _processor->loadDetector(detectorModelPath, inputSize, classNamesList);
    if(!detectorStatus)
        return{};

    _processor->startProcess();

    for (int i = 0; i < 1000; ++i) {
        _processor->process();

        QThread::msleep(100);

        auto res = _processor->getResult();

        auto img = cv::imread(_imgPath);
        auto color_box = cv::Scalar(0, 0, 255);
        for (int i = 0; i < res.size(); ++i) {
            cv::rectangle(img, res[i].box, color_box, 2, 8);
            cv::putText(img, res[i].className, cv::Point(res[i].box.x, res[i].box.y),
                        cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }
        cv::imshow("Detection Box", img);
        cv::waitKey(0);
    }

    _processor->stopProcess();

    return a.exec();
}
