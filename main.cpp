#include <QCoreApplication>
#include "detector.h"
#include "segmentor.h"
#include "detector_opencv_dnn.h"
#include "segmentor_opencv_dnn.h"
#include "detector_onnxruntime.h"
#include "segmentor_onnxruntime.h"
#include "detector_tensorrt_end2end.h"

void detectorFunc(Detector *detector, std::vector<cv::Mat> imgList, std::vector<std::string> _classNamesList)
{
    std::vector<cv::Mat> imgBatch;

    for (int imgIDX = 0; imgIDX < imgList.size(); ++imgIDX) {
        // make batch of images = 1
        imgBatch.clear();
        imgBatch.push_back(imgList[imgIDX]);

        auto result = detector->Run(imgBatch);

        //        auto img = imgList[imgIDX];
        //        auto color_box = cv::Scalar(0, 0, 255);
        //        cv::Mat boxImg = img.clone();
        //        for (int i = 0; i < result[0].size(); ++i) {
        //            cv::rectangle(boxImg, result[0][i].box, color_box, 2, 8);
        //            cv::putText(boxImg, _classNamesList[result[0][i].classID],
        //                        cv::Point(result[0][i].box.x, result[0][i].box.y),
        //                        cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        //        }
        //        cv::imshow("Detection Box " + QString::number(imgIDX).toStdString(), boxImg);
    }
}
void segmentorFunc(Segmentor *segmentor, std::vector<cv::Mat> imgList, std::vector<std::string> _classNamesList)
{
    std::vector<cv::Mat> imgBatch;
    for (int imgIDX = 0; imgIDX < imgList.size(); ++imgIDX) {
        // make batch of images = 1
        imgBatch.clear();
        imgBatch.push_back(imgList[imgIDX]);

        auto result = segmentor->Run(imgBatch);

        //        auto img = imgList[imgIDX];
        //        auto color_box = cv::Scalar(0, 0, 255);
        //        auto color_mask = cv::Scalar(0, 255, 0);
        //        auto color_contours = cv::Scalar(255, 0, 0);
        //        cv::Mat maskImg = img.clone();
        //        cv::Mat boxImg = img.clone();
        //        cv::Mat contoursImg = img.clone();
        //        for (int i = 0; i < result[0].size(); ++i) {
        //            maskImg(result[0][i].box).setTo(color_mask, result[0][i].boxMask);

        //            cv::rectangle(boxImg, result[0][i].box, color_box, 2, 8);
        //            cv::putText(boxImg, _classNamesList[result[0][i].classID],
        //                        cv::Point(result[0][i].box.x, result[0][i].box.y),
        //                        cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

        //            for(size_t c = 0; c< result[0][i].maskContoursList.size(); c++)
        //                drawContours(contoursImg, result[0][i].maskContoursList, (int)c, color_contours, 2, cv::LINE_8, {}, 0 );
        //        }
        //        cv::imshow("Segmentation Mask " + QString::number(imgIDX).toStdString(), maskImg);
        //        cv::imshow("Segmentation Box " + QString::number(imgIDX).toStdString(), boxImg);
        //        cv::imshow("Segmentation Contours " + QString::number(imgIDX).toStdString(), contoursImg);
    }
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    std::vector<std::string> _classNamesList = {
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
    auto imgPathList = {
        "/media/chiko/HDD_1/Work/Training_Scripts/CocoImage/000000000009.jpg",
        "/media/chiko/HDD_1/Work/Training_Scripts/CocoImage/000000000025.jpg",
        "/media/chiko/HDD_1/Work/Training_Scripts/CocoImage/000000000030.jpg",
        "/media/chiko/HDD_1/Work/Training_Scripts/CocoImage/000000000034.jpg",
        "/media/chiko/HDD_1/Work/Training_Scripts/CocoImage/000000000036.jpg",
        "/media/chiko/HDD_1/Work/Training_Scripts/CocoImage/000000000042.jpg",
        "/media/chiko/HDD_1/Work/Training_Scripts/CocoImage/000000000049.jpg",
        "/media/chiko/HDD_1/Work/Training_Scripts/CocoImage/000000000061.jpg",
        "/media/chiko/HDD_1/Work/Training_Scripts/CocoImage/img.jpeg"
    };
    std::vector<cv::Mat> imgList;
    for(auto imgPath : imgPathList){
        auto img = cv::imread(imgPath);
        imgList.push_back(img);
    }

    auto batchSize = 1;
    auto inputSize = cv::Size(640, 640);

    //--------------------------------------------------Detector
    {
        Detector *detector{nullptr}; // Detector_OpenCV_DNN Or Detector_ONNXRUNTIME Or Detector_TensorRT_End2End
        detector = new Detector_TensorRT_End2End;

//        QString detectorModelPath = "/media/chiko/HDD_1/Work/Training_Scripts/YOLOv8/Weights/Detection/yolov8n.onnx";
        QString detectorModelPath = "/media/chiko/HDD_1/Work/Training_Scripts/YOLOv8/Weights/Detection/End2End/yolov8n.engine";
        auto detectorStatus = detector->LoadModel(detectorModelPath);
        if(!detectorStatus)
            return{};

        detector->setClassNames(_classNamesList);
        detector->setBatchSize(batchSize);
        detector->setInputSize(inputSize);

        detectorFunc(detector, imgList, _classNamesList);
    }

    //--------------------------------------------------Segmentor
    {
        Segmentor *segmentor{nullptr}; // Or Segmentor_ONNXRUNTIME Or Segmentor_ONNXRUNTIME
        segmentor = new Segmentor_OpenCV_DNN;

        QString segmentorModelPath = "/media/chiko/HDD_1/Work/Training_Scripts/YOLOv8/Weights/Segmentation/yolov8n-seg.onnx";
        auto segmentorStatus = segmentor->LoadModel(segmentorModelPath);
        if(!segmentorStatus)
            return{};

        segmentor->setClassNames(_classNamesList);
        segmentor->setBatchSize(batchSize);
        segmentor->setInputSize(inputSize);

        segmentorFunc(segmentor, imgList, _classNamesList);
    }

    return a.exec();
}
