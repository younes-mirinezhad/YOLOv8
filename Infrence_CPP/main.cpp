#include <QCoreApplication>
#include "processingthread.h"
#include "detector_tensorrt.h"
#include "detector_opencv_dnn.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    QString detectorModelPath = "/media/chiko/HDD_1/W/Pars_AI/Projects/Archive/YOLOv8/Models/Detection/TensorRT/yolov8n_640.engine";
    auto detector = new Detector_TensorRT;

    // QString detectorModelPath = "/media/chiko/HDD_1/W/Pars_AI/Projects/Archive/YOLOv8/Models/Detection/yolov8n.onnx";
    // auto detector = new Detector_OpenCV_DNN; // It has problem in detection boxes

    auto inputSize = cv::Size(640, 640);
    detector->setInputSize(inputSize);

    auto detectorStatus = detector->LoadModel(detectorModelPath);
    if(!detectorStatus)
        return{};

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
    detector->setClassNames(classNamesList);

    ProcessingThread *_processor = new ProcessingThread;
    _processor->setDetector(detector);
    auto imgPath = "/media/chiko/HDD_2/Dataset/Coco/CocoImage/img.jpeg";
    _processor->setImagePath(imgPath);
    _processor->startProcess();

    for (int i = 0; i < 1000; ++i) {
        _processor->process();

        QThread::msleep(100);

        auto res = _processor->getResult();

        auto img = cv::imread(imgPath);
        auto color_box = cv::Scalar(0, 0, 255);
        for (int i = 0; i < res.size(); ++i) {
            qDebug() << "---------->"
                     << "- Class:" << res[i].className
                     << "- Conf:"<< res[i].confidence
                     << "- [x, y]: [" << res[i].box.x <<","<< res[i].box.y << "]"
                     << "- [w, h]: [" << res[i].box.width <<","<< res[i].box.height << "]";

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
