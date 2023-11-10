#include <QCoreApplication>
#include "segmentor.h"
#include "segmentor_opencv_dnn.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    QString modelPath = "/media/chiko/HDD_1/Work/Training_Scripts/YOLOv8/yolov8n-seg.onnx";

    Segmentor *segmentor{nullptr};
    segmentor = new Segmentor_OpenCV_DNN;

    auto modelStatus = segmentor->LoadModel(modelPath);

    return a.exec();
}
