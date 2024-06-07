#include <QCoreApplication>
#include "detector_tensorrt.h"
#include "spdlog/spdlog.h"
#include "qthread.h"
// #include "detector_opencv_dnn.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    std::string detectorModelPath = "/media/chiko/HDD_1/Work/Pars_AI/Projects/InProgress/YOLOv8/Models/Detection/yolov8n_end2end.engine";
    auto detector = new Detector_TensorRT;

    // std::string detectorModelPath = "/media/chiko/HDD_1/W/Pars_AI/Projects/Archive/YOLOv8/Models/Detection/yolov8n.onnx";
    // auto detector = new Detector_OpenCV_DNN; // It has problem in detection boxes

    auto inputSize = cv::Size(640, 640);
    detector->setInputSize(inputSize);

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

    auto detectorStatus = detector->LoadModel(detectorModelPath);
    if(!detectorStatus)
        return{};

    auto imgPath = "/media/chiko/HDD_2/Dataset/Coco/CocoImage/img.jpeg";
    for (int idx = 0; idx < 100; ++idx) {
        QThread::msleep(50);

        auto t1 = std::chrono::high_resolution_clock::now();

        auto img = cv::imread(imgPath);

        // -------------------- Use cv::Mat
        // auto res = detector->detect(img);

        // -------------------- Use cv::cuda::GpuMat
        cv::cuda::GpuMat gImg;
        gImg.upload(img);
        auto res = detector->detect(gImg);

        auto t2 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        res.detectionTime_ms = dt;

        img = cv::imread(imgPath);
        auto color_box = cv::Scalar(0, 0, 255);
        for (int i = 0; i < res.detections.size(); ++i) {
            spdlog::info("----------> Class: {} - Conf: {} - Box: [{}x{}] , [{}x{}]",
                          res.detections[i].className,
                          res.detections[i].confidence,
                          res.detections[i].box.x,
                          res.detections[i].box.y,
                          res.detections[i].box.width,
                          res.detections[i].box.height,
                          dt);

            cv::rectangle(img, res.detections[i].box, color_box, 2, 8);
            cv::putText(img, res.detections[i].className,
                        cv::Point(res.detections[i].box.x, res.detections[i].box.y),
                        cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }
        spdlog::info("---------------> detection time: {} ms", res.detectionTime_ms);
        spdlog::info("------------------------------------------------------------");
        cv::imshow("Detection Box", img);
    }
    cv::waitKey(0);

    return a.exec();
}
