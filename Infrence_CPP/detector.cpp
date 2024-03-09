#include "detector.h"

Detector::Detector(QObject *parent) : QObject{parent} {}

void Detector::setClassNames(std::vector<std::string> newClassNamesList)
{
    qDebug() << Q_FUNC_INFO;

    _classNamesList = newClassNamesList;
}

void Detector::setInputSize(cv::Size newInputSize)
{
    qDebug() << Q_FUNC_INFO;

    _inputSize = newInputSize;
}
