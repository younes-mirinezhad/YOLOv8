#ifndef PROCESSINGTHREAD_H
#define PROCESSINGTHREAD_H

#include <QThread>
#include "detector.h"

class ProcessingThread : public QThread
{
public:
    ProcessingThread();

    void setDetector(Detector *newDetector);
    Detector *_detector{nullptr};

    void setImagePath(std::string newPath);
    std::string _imgPath{};

    void startProcess();
    void stopProcess();

    void run();
    QAtomicInt _running{0};
    QAtomicInt _process{0};

    void process();
    ImagesDetectedObject getResult();
    ImagesDetectedObject _result;
};

#endif // PROCESSINGTHREAD_H
