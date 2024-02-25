#include "processingthread.h"
#include "qdebug.h"
#include "detector.h"
#include "detector_tensorrt.h"

ProcessingThread::ProcessingThread() {}

bool ProcessingThread::loadDetector(QString detectorModelPath, cv::Size inputSize, std::vector<std::string> classNamesList)
{
    _detector = new Detector_TensorRT;
    _detector->setClassNames(classNamesList);
    _detector->setInputSize(inputSize);

    auto detectorStatus = _detector->LoadModel(detectorModelPath);

    return detectorStatus;
}

void ProcessingThread::setDetector(Detector *newDetector)
{
    qDebug() << Q_FUNC_INFO;

    _detector = newDetector;
}

void ProcessingThread::startProcess()
{
    qDebug() << Q_FUNC_INFO;

    _running = 1;
    start();
}
void ProcessingThread::stopProcess()
{
    qDebug() << Q_FUNC_INFO;

    _running = 0;
}

void ProcessingThread::run()
{
    qDebug() << Q_FUNC_INFO;
    auto _imgPath = "/media/chiko/HDD_1/Work/Training_Scripts/CocoImage/img.jpeg";

    while (_running) {
        if(_process) {
            auto Start_time = std::chrono::high_resolution_clock::now();

            auto img = cv::imread(_imgPath);

            // Use cv::Mat
            // _result = _detector->detect(img);

            // Use cv::cuda::GpuMat
            cv::cuda::GpuMat gImg;
            gImg.upload(img);
            _result = _detector->detect(gImg);

            auto Current_time = std::chrono::high_resolution_clock::now();
            auto Elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(Current_time - Start_time).count();
            qDebug() << "----- Detection time:" << Elapsed_time;

            _process = 0;
        }
    }
}

void ProcessingThread::process()
{
    if(_process) {
        qDebug() << "***** Detection is in not finished";
    } else {
        _process = 1;
    }
}

ImagesDetectedObject ProcessingThread::getResult()
{
    return _result;
}
