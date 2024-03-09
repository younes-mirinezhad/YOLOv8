#include "processingthread.h"
#include "qdebug.h"

ProcessingThread::ProcessingThread() {}

void ProcessingThread::setDetector(Detector *newDetector)
{
    qDebug() << Q_FUNC_INFO;

    _detector = newDetector;
}

void ProcessingThread::setImagePath(std::string newPath)
{
    _imgPath = newPath;
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

    while (_running) {
        if(_process) {
            auto Start_time = std::chrono::high_resolution_clock::now();

            auto img = cv::imread(_imgPath);

            // Use cv::Mat
            _result = _detector->detect(img);

            // Use cv::cuda::GpuMat
            // cv::cuda::GpuMat gImg;
            // gImg.upload(img);
            // _result = _detector->detect(gImg);

            auto Current_time = std::chrono::high_resolution_clock::now();
            auto Elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(Current_time - Start_time).count();
            auto dt = double(Elapsed_time / 100)/10;
            qDebug() << "----- Detection time:" << dt;

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
