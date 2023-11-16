#include <QCoreApplication>
#include "detector.h"
#include "detector_opencv_dnn.h"
#include "segmentor.h"
#include "segmentor_opencv_dnn.h"
#include "segmentor_onnxruntime.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    std::vector<std::string> _classNamesList = {"person", "bicycle", "car", "motorcycle",
                                                "airplane", "bus", "train", "truck", "boat",
                                                "traffic light", "fire hydrant", "stop sign",
                                                "parking meter", "bench", "bird", "cat", "dog",
                                                "horse", "sheep", "cow", "elephant", "bear",
                                                "zebra", "giraffe", "backpack", "umbrella",
                                                "handbag", "tie", "suitcase", "frisbee", "skis",
                                                "snowboard", "sports ball", "kite", "baseball bat",
                                                "baseball glove", "skateboard", "surfboard",
                                                "tennis racket", "bottle", "wine glass", "cup",
                                                "fork", "knife", "spoon", "bowl", "banana", "apple",
                                                "sandwich", "orange", "broccoli", "carrot", "hot dog",
                                                "pizza", "donut", "cake", "chair", "couch", "potted plant",
                                                "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                                                "remote", "keyboard", "cell phone", "microwave", "oven",
                                                "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                                "scissors", "teddy bear", "hair drier", "toothbrush"};
    auto imgPath = "/media/chiko/HDD_1/Work/Training_Scripts/YOLOv8/img.jpeg";
    auto batchSize = 1;
    auto inputSize = cv::Size(640, 640);

    auto img = cv::imread(imgPath);
    std::vector<cv::Mat> imgList = {img};

    //##################### Segmentor
    {
        Segmentor *segmentor{nullptr};
//        segmentor = new Segmentor_OpenCV_DNN;
        segmentor = new Segmentor_ONNXRUNTIME;

        QString modelPath = "/media/chiko/HDD_1/Work/Training_Scripts/YOLOv8/yolov8m-seg.onnx";
        auto modelStatus = segmentor->LoadModel(modelPath);

        if(modelStatus) {
            segmentor->setClassNames(_classNamesList);
            segmentor->setBatchSize(batchSize);
            segmentor->setInputSize(inputSize);

            auto result = segmentor->Run(imgList);

            auto color_box = cv::Scalar(0, 0, 255);
            auto color_mask = cv::Scalar(0, 255, 0);
            auto color_contours = cv::Scalar(255, 0, 0);
            cv::Mat maskImg = img.clone();
            cv::Mat boxImg = img.clone();
            cv::Mat contoursImg = img.clone();
            for (int i = 0; i < result[0].size(); ++i) {
                maskImg(result[0][i].box).setTo(color_mask, result[0][i].boxMask);

                cv::rectangle(boxImg, result[0][i].box, color_box, 2, 8);
                cv::putText(boxImg, _classNamesList[result[0][i].classID],
                            cv::Point(result[0][i].box.x, result[0][i].box.y),
                            cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

                for(size_t c = 0; c< result[0][i].maskContoursList.size(); c++)
                    drawContours(contoursImg, result[0][i].maskContoursList, (int)c, color_contours, 2, cv::LINE_8, {}, 0 );
            }
            cv::imshow("Segmentation Mask", maskImg);
            cv::imshow("Segmentation Box", boxImg);
            cv::imshow("Segmentation Contours", contoursImg);
        }
    }

    //##################### Detector
    {
//        Detector *detector{nullptr};
//        detector = new Detector_OpenCV_DNN;

//        QString modelPath = "/media/chiko/HDD_1/Work/Training_Scripts/YOLOv8/yolov8m.onnx";
//        auto modelStatus = detector->LoadModel(modelPath);

//        if(modelStatus){
//            detector->setClassNames(_classNamesList);
//            detector->setBatchSize(batchSize);
//            detector->setInputSize(inputSize);

//            auto result = detector->Run(imgList);

//            auto color_box = cv::Scalar(0, 0, 255);
//            cv::Mat boxImg = img.clone();
//            for (int i = 0; i < result[0].size(); ++i) {
//                cv::rectangle(boxImg, result[0][i].box, color_box, 2, 8);
//                cv::putText(boxImg, _classNamesList[result[0][i].classID],
//                            cv::Point(result[0][i].box.x, result[0][i].box.y),
//                            cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
//            }
//            cv::imshow("Detection Box", boxImg);
//        }
    }

    return a.exec();
}
